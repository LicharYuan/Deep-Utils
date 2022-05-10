from datetime import timedelta
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dutils.torch.dist as udist

DEFAULT_TIMEOUT = timedelta(minutes=30)
__all__ = ["launch"]

def _find_free_port():

    # Find an available port of current machine / node.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):  
    # support multi-machine
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"        

        start_method = "spawn"

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        print(" *** Run Process in Single GPU *** ")
        main_func(*args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    print("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        print("Process group URL: {}".format(dist_url))
        exit("fail init process group")
        raise

    assert  udist._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            udist._LOCAL_PROCESS_GROUP = pg

    udist.synchronize()
    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    main_func(*args)
    