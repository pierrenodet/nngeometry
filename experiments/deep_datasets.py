from functools import partial
from typing import Callable, Dict, NamedTuple, Optional

import torch
from deep_models import ResNet18, ResNet34
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms


def mnist1d(root, train=True, transform=None):
    from mnist1d.data import get_dataset_args, make_dataset

    args = get_dataset_args()
    args.num_samples = 10000
    args.train_split = 0.1
    args.padding = [24, 40]
    args.max_translation = 34
    args.final_seq_length = 28
    # args.iid_noise_scale = 1e-3
    # args.corr_noise_scale = 0.1
    # args.shear_scale = 0
    data = make_dataset(args)
    X_train, y_train = (
        torch.from_numpy(data["x"]).to(dtype=torch.float32),
        torch.tensor(data["y"]),
    )
    X_test, y_test = (
        torch.from_numpy(data["x_test"]).to(dtype=torch.float32),
        torch.tensor(data["y_test"]),
    )
    if train:
        return TensorDataset(X_train, y_train)
    else:
        return TensorDataset(X_test, y_test)


def cifar10(root, train=True, transform=None):
    from datasets import load_dataset

    split = "train" if train else "test"
    dataset = load_dataset("uoft-cs/cifar10", split=split)

    if transform is None:
        transform = lambda x: x  # noqa: E731

    X = torch.stack([transform(img.convert("RGB")) for img in dataset["img"]])
    y = torch.tensor(dataset["label"])

    return TensorDataset(X, y)


def cifar100(root, train=True, transform=None):
    from datasets import load_dataset

    split = "train" if train else "test"
    dataset = load_dataset("uoft-cs/cifar100", split=split)

    if transform is None:
        transform = lambda x: x  # noqa: E731

    X = torch.stack([transform(img.convert("RGB")) for img in dataset["img"]])
    y = torch.tensor(dataset["fine_label"])

    return TensorDataset(X, y)


class Experiment(NamedTuple):
    transform: Callable | None
    dataset: Callable[[str, bool, Callable], Dataset]
    model: nn.Module
    optimizer: Callable[[Dict[str, Optional[nn.Parameter]]], optim.Optimizer]
    classes: list[str] = []


class Residuals(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


DATASETS = {
    "mnist-1d": Experiment(
        None,
        mnist1d,
        # lambda outputs=10, c=32: nn.Sequential(
        #     nn.Unflatten(-1, (1, -1)),
        #     nn.Conv1d(1, c, 5, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool1d(2),
        #     nn.Conv1d(c, c * 2, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool1d(2),
        #     nn.Conv1d(c * 2, c * 4, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.Linear(c * 4, outputs),
        # ),
        # partial(optim.AdamW, weight_decay=1e-2),
        # partial(
        #     torch.optim.SGD,
        #     lr=1e-2,
        #     momentum=0.9,
        #     weight_decay=1e-2,
        #     nesterov=True,
        # ),
        lambda outputs=10, c=64, h=4, d=2, k=9: nn.Sequential(
            # nn.Unflatten(-1, (1, -1)),
            # nn.Conv1d(1, c, kernel_size=9, padding=9 // 2),
            # nn.GELU(),
            # nn.AdaptiveAvgPool1d(1),
            # nn.Flatten(),
            # nn.Linear(c, c),
            # nn.Conv1d(1, c, kernel_size=k, padding=k // 2),
            # nn.GELU(),
            # nn.MaxPool1d(2),  # 28 -> 14, first bit of real invariance
            # nn.Conv1d(c, c * 2, kernel_size=k, padding=k // 2),
            # nn.GELU(),
            # nn.AdaptiveAvgPool1d(1),  # global pool -> true shift invariance
            # nn.Flatten(),
            # nn.Linear(c * 2, c),
            nn.Linear(28, c),
            *[
                Residuals(
                    nn.Sequential(
                        nn.LayerNorm(c),
                        nn.Linear(c, c * h),
                        nn.GELU(),
                        nn.Linear(c * h, c),
                    )
                )
                for _ in range(d)
            ],
            nn.LayerNorm(c),
            nn.Linear(c, outputs),
        ),
        partial(
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=True,
        ),
        list(map(str, range(10))),
    ),
    "mnist": Experiment(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        partial(
            datasets.MNIST,
            download=True,
        ),
        # lambda outputs=10, c=64: nn.Sequential(
        #     nn.Conv2d(1, c, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(c, c * 2, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(c * 2, c * 4, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(c * 4, outputs),
        # ),
        lambda outputs=10, c=1024: nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, c),
            nn.ReLU(),
            nn.Linear(c, outputs),
        ),
        # partial(optim.SGD, lr=1e-3, weight_decay=1e-2, momentum=0.9, nesterov=True),
        partial(optim.AdamW, weight_decay=1e-2),
        list(map(str, range(10))),
    ),
    "k-mnist": Experiment(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1918,), (0.3483,)),
            ]
        ),
        partial(datasets.KMNIST, download=True),
        lambda outputs=10, c=64: nn.Sequential(
            nn.Conv2d(1, c, 3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(c, c * 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 4, outputs),
        ),
        # lambda outputs=10: MLP(784, outputs, (512, 128)).insert(0, nn.Flatten()),
        partial(optim.SGD, lr=0.01, weight_decay=1e-3, momentum=0.9, nesterov=True),
        # partial(optim.Adam, lr=1e-3, weight_decay=1e-3),
        [
            "お",
            "き",
            "す",
            "つ",
            "な",
            "は",
            "ま",
            "や",
            "れ",
            "を",
        ],
        # [
        #     "あ",
        #     "い",
        #     "う",
        #     "え",
        #     "お",
        #     "か",
        #     "き",
        #     "く",
        #     "け",
        #     "こ",
        #     "さ",
        #     "し",
        #     "す",
        #     "せ",
        #     "そ",
        #     "た",
        #     "ち",
        #     "つ",
        #     "て",
        #     "と",
        #     "な",
        #     "に",
        #     "ぬ",
        #     "ね",
        #     "の",
        #     "は",
        #     "ひ",
        #     "ふ",
        #     "へ",
        #     "ほ",
        #     "ま",
        #     "み",
        #     "む",
        #     "め",
        #     "も",
        #     "や",
        #     "ゆ",
        #     "よ",
        #     "ら",
        #     "り",
        #     "る",
        #     "れ",
        #     "ろ",
        #     "わ",
        #     "ゐ",
        #     "ゑ",
        #     "を",
        #     "ん",
        #     "ゝ",
        # ],
    ),
    "fashion-mnist": Experiment(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        ),
        partial(datasets.FashionMNIST, download=True),
        lambda outputs=10, c=64: nn.Sequential(
            nn.Conv2d(1, c, 3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(c, c * 2, 3, stride=1, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, stride=1, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 4, outputs),
        ),
        partial(optim.AdamW, weight_decay=1e-2),
        # partial(optim.SGD, lr=1e-3, weight_decay=1e-2, momentum=0.9, nesterov=True),
        [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle Boot",
        ],
    ),
    "cifar-10": Experiment(
        transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)
                ),
            ],
        ),
        cifar10,
        # partial(datasets.CIFAR10, download=True),
        partial(ResNet18, batch_norm=False),
        # partial(VGG, "VGG19"),
        # partial(ResNet18, batch_norm=False),
        # lambda outputs=10: VisionTransformer(outputs, num_layers=1, hidden_size=512),
        # lambda outputs=10, c=64: nn.Sequential(
        #     nn.Conv2d(3, c, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(c, c * 2, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(c * 2, c * 4, 3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(c * 4, c * 8, 3, stride=1, padding=1),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(c * 8, outputs),
        # ),
        # nn.Conv2d(3, c, 3, stride=1, padding=1),
        # nn.GELU(),
        # nn.Conv2d(c, c, 3, stride=1, padding=1),
        # nn.GELU(),
        # nn.AvgPool2d(2),
        # nn.Conv2d(c, c * 2, 3, stride=1, padding=1),
        # nn.GELU(),
        # nn.Conv2d(c * 2, c * 2, 3, stride=1, padding=1),
        # nn.GELU(),
        # nn.AvgPool2d(2),
        # nn.Conv2d(c * 2, c * 4, 3, stride=1, padding=1),
        # nn.GELU(),
        # nn.Conv2d(c * 4, c * 4, 3, stride=1, padding=1),
        # nn.GELU(),
        # # nn.AvgPool2d(2),
        # nn.AdaptiveAvgPool2d(1),
        # nn.Flatten(),
        # nn.Linear(c * 4, outputs),
        # # nn.GELU(),
        # # nn.Linear(c * 4, outputs),
        # ),
        # partial(optim.SGD, lr=1e-2, weight_decay=1e-4, momentum=0.9, nesterov=True),
        partial(optim.AdamW, weight_decay=1e-2),
        [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    ),
    "cifar-100": Experiment(
        transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4866, 0.4409), std=(0.2673, 0.2564, 0.2762)
                ),
            ],
        ),
        cifar100,
        # partial(VGG, "VGG19"),
        partial(ResNet34, batch_norm=False),
        # partial(optim.SGD, lr=0.01, weight_decay=1e-3, momentum=0.9, nesterov=True),
        partial(optim.AdamW, weight_decay=1e-2),
        [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ],
    ),
}
