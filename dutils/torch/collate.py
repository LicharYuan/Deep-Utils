import torch

class CData:
    def __init__(self, data, expand=True, to_tensor=True):
        self._data = data
        self._expand = expand
        self._tt = to_tensor
        self._tl = to_list

    @property
    def to_tensor(self):
        # or to list
        return self._tt

    @property
    def expand_dim(self):
        return self._expand


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(type(data))

def collate_fn(batch):
    # concat batch > 1 results
    imgs_pair_batch = []
    target_batch = []
    for d in batch:
        if d is not None:
            imgs_pair_batch.append(np.expand_dims(d[0], 0))
            target_batch.append(np.expand_dims(d[1], 0))

    imgs_pair_batch = to_tensor(np.concatenate(imgs_pair_batch, axis=0))
    target_batch = to_tensor(np.concatenate(target_batch, axis=0))
    return imgs_pair_batch, target_batch

def cdata_collate_fn(batch, data_keys=None):
    collate_datas = [[] for _ in len(batch[0])]
    for datas in batch:
        for i, d in datas:
            if d.expand_dim:
                collate_datas[i].append(np.expand_dims(d, 0))
            else:
                collate_datas[i].append(d)

    _try_data = batch[0]
    for i,d in try_data:
        if d.to_tensor:
            collate_datas[i] = to_tensor(np.concatenate(collate_datas[i], axis=0))
        else:
            collate_datas[i] = np.concatenate(collate_datas[i], axis=0)
    
    if data_keys is not None:
        return_data = {}
        assert len(data_keys) == len(batch[0]), "key mismatch"
        for key, collate_data in zip(data_keys, collate_datas):
            return_data[key] = collate_data
        return return_data

    return collate_data

