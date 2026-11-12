# /// script
# dependencies = [
#   "nngeometry"
# ]
# [tool.uv.sources]
# nngeometry = { path = "..", editable = true }
# ///


# %%
import argparse
import pprint
import time
from functools import partial

import torch
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50
from utils import InMemoryDataLoader, to_tensor_dataset

from nngeometry.backend import torch_hooks as nnj
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM_MonteCarlo
from nngeometry.object import PMatDiag, PMatEKFAC, PMatImplicit, PMatKFAC, PMatQuasiDiag
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import PVector, random_pvector

# # ResNet50 on CIFAR10
parser = argparse.ArgumentParser(description="resnet benchmark on cifar10")
parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--image_size", choices=["imagenet", "cifar10"], default="cifar10")
args = parser.parse_args()

torch.manual_seed(0)
# %%
transform = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]
if args.image_size == "imagenet":
    transform = [
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
        )
    ] + transform
transform = transforms.Compose(transform)

datasets.CIFAR10.url = (
    "https://zenodo.org/records/15602362/files/cifar-10-python.tar.gz?download=1"
)
trainset = datasets.CIFAR10(
    root="/tmp/data", train=True, download=True, transform=transform
)
trainset = to_tensor_dataset(Subset(trainset, range(1000)), device=args.device)
trainloader = InMemoryDataLoader(trainset, batch_size=50, shuffle=False)

# %%
model = resnet50(num_classes=10, norm_layer=lambda c: nn.Identity())
if args.image_size == "cifar10":
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool == nn.Identity()
model = model.to(args.device)
model = model.eval()


layer_collection = LayerCollection.from_model(model)
pvector = random_pvector(
    LayerCollection.from_model(model), device=args.device, dtype=torch.float32
)

print(f"{layer_collection.numel()} parameters")

# %%
# compute timings and display FIMs


def perform_timing():
    timings = dict()

    for repr in [
        PMatImplicit,
        PMatDiag,
        PMatEKFAC,
        PMatKFAC,
        partial(PMatKFAC, strategy="one_iter_kpsvd"),
        PMatQuasiDiag,
        PMatDense,  # last layer
    ]:
        print("Timing representation:")
        pprint.pprint(repr)

        timings[repr] = dict()

        if repr == PMatDense:
            for i, (layer_id, layer) in enumerate(layer_collection.layers.items()):
                if i == len(layer_collection.layers) - 1:
                    lc = LayerCollection({layer_id: layer})
                    v = PVector(
                        lc, dict_repr={layer_id: pvector.to_torch_layer(layer_id)}
                    )
        else:
            lc = layer_collection
            v = pvector

        time_start = time.time()
        F = FIM_MonteCarlo(
            model=model,
            loader=trainloader,
            representation=repr,
            device=args.device,
            layer_collection=lc,
        )
        time_end = time.time()
        timings[repr]["init"] = time_end - time_start

        if repr == PMatEKFAC:
            time_start = time.time()
            F.update_diag(examples=trainloader)
            time_end = time.time()
            timings[repr]["update_diag"] = time_end - time_start

        time_start = time.time()
        F.mv(v)
        time_end = time.time()
        timings[repr]["Mv"] = time_end - time_start

        time_start = time.time()
        F.vTMv(v)
        time_end = time.time()
        timings[repr]["vTMv"] = time_end - time_start

        time_start = time.time()
        F.trace()
        time_end = time.time()
        timings[repr]["tr"] = time_end - time_start

        try:
            time_start = time.time()
            F.norm()
            time_end = time.time()
            timings[repr]["frob"] = time_end - time_start
        except NotImplementedError:
            pass

        try:
            time_start = time.time()
            F.solve(v)
            time_end = time.time()
            timings[repr]["solve"] = time_end - time_start
        except:
            pass

        del F

    pprint.pprint(timings)


# %%

with nnj.use_unfold_impl_for_convs():
    perform_timing()

with nnj.use_conv_impl_for_convs():
    perform_timing()
