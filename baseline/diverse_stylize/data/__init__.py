import numpy as np
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from data.data_loader_my import SourceDataset, ReferenceDataset


def make_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def create_data_loader(opt, which='source', type=None):
    print('Preparing %s dataset during %s phase...' % (which, type))
    if which == 'source':
        dataset = SourceDataset(opt, type)
    elif which == 'reference':
        dataset = ReferenceDataset(opt, type)
    else:
        raise NotImplementedError

    if type == 'train':
        sampler = make_weighted_sampler(dataset.targets)
        return data.DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            sampler=sampler,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True)
    elif type is not None:
        return data.DataLoader(dataset=dataset, batch_size=opt.batch_size)
    else: 
        raise NotImplementedError('Please specify dataset type!')