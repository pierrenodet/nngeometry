import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)


class InMemoryDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size=32, shuffle=False, drop_last=False, device=None
    ):
        assert hasattr(dataset, "tensors")
        self.tensors = tuple([t.to(device=device) for t in dataset.tensors])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = dataset
        self.dataset_length = len(self.dataset)
        self.sampler = (
            SequentialSampler(self.dataset)
            if not shuffle
            else RandomSampler(self.dataset)
        )  # for compat

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(
                self.dataset_length, device=self.tensors[0].device
            )
        else:
            self.indices = None

        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_length:
            raise StopIteration

        end_idx = self.current_idx + self.batch_size

        if end_idx > self.dataset_length:
            if self.drop_last:
                raise StopIteration
            end_idx = self.dataset_length

        if self.shuffle:
            batch_idx = self.indices[self.current_idx : end_idx]
            batch = tuple(t[batch_idx] for t in self.tensors)
        else:
            batch = tuple(t[self.current_idx : end_idx] for t in self.tensors)

        self.current_idx = end_idx

        return batch

    def __len__(self):
        if self.drop_last:
            return self.dataset_length // self.batch_size
        return (self.dataset_length + self.batch_size - 1) // self.batch_size
