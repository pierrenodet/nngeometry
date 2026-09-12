import tempfile
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn.functional as F
from deep_datasets import DATASETS
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    TensorDataset,
)
from tqdm import tqdm


def tensorize(dataset, device=None):
    inputs, targets = [], []
    for i, t in dataset:
        inputs.append(i)
        targets.append(t)
    inputs = torch.stack(inputs, dim=0)
    if isinstance(targets[0], (int, float)) or targets[0].ndim == 0:
        targets = torch.tensor(targets)
    else:
        targets = torch.stack(targets, dim=0)
    if device is not None:
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
    return TensorDataset(inputs, targets)


def subset(dataset, indices):
    if isinstance(dataset, TensorDataset):
        return tensorize(Subset(dataset, indices))
    else:
        return Subset(dataset, indices)


def concat(*datasets):
    if isinstance(datasets[0], TensorDataset):
        inputs, targets = [], []
        for dataset in datasets:
            inputs.append(dataset.tensors[0])
            targets.append(dataset.tensors[1])
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        return TensorDataset(inputs, targets)
    else:
        return ConcatDataset(datasets)


def make_loss_fn(_):
    return F.cross_entropy


# def loop(fn, model, loader, device):
#     outputs = []
#     if isinstance(loader.dataset, TensorDataset):  # in memory dataset
#         inputs = deepcopy(loader.dataset.tensors[0])
#         labels = deepcopy(loader.dataset.tensors[1])

#         if inputs.device != device:
#             inputs = inputs.to(device=device)
#             labels = labels.to(device=device)
#         if isinstance(loader.sampler, RandomSampler):
#             shuffle = torch.randperm(inputs.shape[0], device=device)
#             inputs = inputs[shuffle]
#             labels = labels[shuffle]
#         for batch in range(0, inputs.shape[0], loader.batch_size):
#             # last_batch = batch > inputs.shape[0] - loader.batch_size
#             outputs.append(
#                 fn(
#                     model,
#                     inputs[batch : (batch + loader.batch_size)],
#                     labels[batch : (batch + loader.batch_size)],
#                 )
#             )
#     else:  # use dataloader api
#         for inputs, labels in loader:
#             inputs = inputs.to(device=device)
#             labels = labels.to(device=device)
#             outputs.append(fn(model, inputs, labels))
#     return outputs


def loop(fn, model, loader, device):
    outputs = []
    for inputs, labels in loader:
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        outputs.append(fn(model, inputs, labels))
    return outputs


def fit(
    model,
    optimizer,
    train,
    epochs,
    device,
    verbose=True,
    forget=False,
    augmentations=[],
):
    model.train()
    loss_fn = make_loss_fn(None)
    if verbose:
        epochs = tqdm(range(epochs))
        # bar = epochs
    else:
        epochs = range(epochs)
        # bar = None

    def fn(model, inputs, labels):
        optimizer.zero_grad()
        for aug in augmentations:
            inputs = aug(inputs)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # if bar is not None:
        #     bar.set_description(f"{round(loss.item(), 2)}")
        if forget:
            loss *= -1
        loss.backward()
        optimizer.step()

    for _ in epochs:
        loop(fn, model, train, device)
    model.eval()
    return model


class InMemoryDataLoader(DataLoader):
    def __init__(
        self, *tensors, batch_size=32, shuffle=False, drop_last=False, device=None
    ):
        self.tensors = tuple([t.to(device=device) for t in tensors])
        self.dataset_len = tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = TensorDataset(*self.tensors)
        self.sampler = (
            SequentialSampler(self.dataset)
            if not shuffle
            else RandomSampler(self.dataset)
        )

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(
                self.dataset_len, device=self.tensors[0].device
            )
        else:
            self.indices = None

        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_len:
            raise StopIteration

        end_idx = self.current_idx + self.batch_size

        if end_idx > self.dataset_len:
            if self.drop_last:
                raise StopIteration
            end_idx = self.dataset_len

        if self.shuffle:
            batch_idx = self.indices[self.current_idx : end_idx]
            batch = tuple(t[batch_idx] for t in self.tensors)
        else:
            batch = tuple(t[self.current_idx : end_idx] for t in self.tensors)

        self.current_idx = end_idx

        return batch

    def __len__(self):
        if self.drop_last:
            return self.dataset_len // self.batch_size
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


def eval(model, test, device):
    loss_fn = make_loss_fn(None)
    model.eval()

    def fn(model, inputs, labels):
        outputs = model(inputs)
        accuracy = (outputs.argmax(dim=1) == labels).sum()
        loss = loss_fn(outputs, labels, reduction="sum")
        count = inputs.shape[0]
        return accuracy, loss, count

    with torch.no_grad():
        outputs = loop(fn, model, test, device)

    accuracy = 0
    count = 0
    loss = 0
    for a, l, c in outputs:  # noqa: E741
        accuracy += a
        loss += l
        count += c

    return (accuracy / count).item(), (loss / count).item()


def embeddings(model, loader, device):
    model = deepcopy(model)
    if hasattr(model, "fc2"):
        model.fc2 = nn.Identity()
    elif hasattr(model, "fc1"):
        model.fc1 = nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()
    else:
        model[-1] = nn.Identity()
    embeddings = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device=device)
            embeddings.append(model(inputs).detach())
    return torch.cat(embeddings)


def add_default_arguments(parser, exp_name):
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="mnist-1d")
    parser.add_argument("--output_dir", type=str, default=f"{exp_name}-output")
    parser.add_argument("--data_dir", type=str, default=tempfile.gettempdir())
    parser.add_argument("--seed", type=int, default=2340320)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--repr", type=str, default="F_mix")
    parser.add_argument("--regul_type", type=str, default="trace")
    parser.add_argument("--regul", type=float, default=1e-2)
    parser.add_argument("--solve_type", type=str, default="default")
    parser.add_argument("--svd_rank", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser


def result_file_name(args):
    return f"{args.dataset}-{datetime.now().date()}-{args.seed}-{args.epochs}-{args.batch_size}-{args.repr}-{args.regul}-{args.regul_type}-{args.solve_type}-data.csv"


def mislabeled_dataset(
    dataset, ratio, n_classes=None, target_transform_fn=None, **kwargs
):
    noisy_dataset = deepcopy(dataset)

    if isinstance(dataset, Subset):
        indices = dataset.indices
        dataset = dataset.dataset
        subset = True
    else:
        indices = torch.arange(len(dataset))
        subset = False

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif isinstance(dataset, TensorDataset):
        targets = dataset.tensors[1]
    else:
        raise ValueError(f"dataset {dataset} not supported for mislabeled")

    if isinstance(targets, list):
        targets = torch.tensor(targets)

    targets = targets[indices]

    dataset_size = len(targets)

    n_noisy_examples = int(dataset_size * ratio)
    noisy_examples = torch.randperm(dataset_size)[:n_noisy_examples]
    is_noisy = torch.zeros(dataset_size, dtype=bool)
    is_noisy[noisy_examples] = True
    if n_classes is None:
        n_classes = targets.max().item() + 1
    noisy_targets = torch.randint(0, n_classes, (n_noisy_examples,))
    if target_transform_fn is not None:
        if isinstance(dataset, TensorDataset):
            noisy_targets = target_transform_fn(noisy_targets)
    targets[noisy_examples] = noisy_targets

    if subset:
        noisy_dataset = noisy_dataset.dataset

    if hasattr(dataset, "targets"):
        for i, idx in enumerate(indices):
            noisy_dataset.targets[idx] = targets[i].item()

    elif isinstance(dataset, TensorDataset):
        for i, idx in enumerate(indices):
            t = targets[i]
            if t.numel() == 1:
                t = t.item()
            noisy_dataset.tensors[1][idx] = t

    if subset:
        noisy_dataset = Subset(noisy_dataset, indices)

    return noisy_dataset, is_noisy
