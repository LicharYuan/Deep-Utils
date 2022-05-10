from contextlib import contextmanager
from torch import distributed as dist
import torch
import torch.nn as nn
import time
import functools
import os
from collections import OrderedDict
import pickle
import warnings
ASYNC_NORM = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
_LOCAL_PROCESS_GROUP = None

# ----------- all reduce norm -----------------
def get_async_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, ASYNC_NORM):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states


def pyobj2tensor(pyobj, device="cuda"):
    """serialize picklable python object to tensor"""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)

def tensor2pyobj(tensor):
    """deserialize tensor to picklable python object"""
    return pickle.loads(tensor.cpu().numpy().tobytes())

def _get_reduce_op(op_name):
    return {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
    }[op_name.lower()]

def all_reduce(py_dict, op="sum", group=None):
    """
    Apply all reduce function for python dict object.
    NOTE: make sure that every py_dict has the same keys and values are in the same shape.

    Args:
        py_dict (dict): dict to apply all reduce op.
        op (str): operator, could be "sum" or "mean".
    """
    world_size = get_world_size()
    if world_size == 1:
        return py_dict
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    py_key_tensor = pyobj2tensor(py_key)
    dist.broadcast(py_key_tensor, src=0)
    py_key = tensor2pyobj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])
    dist.all_reduce(flatten_tensor, op=_get_reduce_op(op))
    if op == "mean":
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape)
        for x, shape in zip(torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    return OrderedDict({k: v for k, v in zip(py_key, split_tensors)})


def all_reduce_norm(module):
    """
    All reduce norm statistics in different devices.
    """
    states = get_async_norm_states(module)
    if len(states.keys()) != 0:
        states = all_reduce(states, op="mean")
        module.load_state_dict(states, strict=False)
    else:
        print("module without Batch Norm")


# --------------- utils ------------------
def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

@contextmanager
def wait_for_the_master(local_rank):
    # Make all processes waiting for the master to do some task.
    if local_rank > 0:
        dist.barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            dist.barrier()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        warnings.warn(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor