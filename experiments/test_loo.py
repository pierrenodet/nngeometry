# /// script
# dependencies = [
#  "nngeometry",
#  "seaborn",
#  "scipy",
#  "pandas",
#  "joblib",
#  "scikit-learn",
#  "requests",
#  "mnist1d",
# ]
# [tool.uv.sources]
# nngeometry = { path = "..", editable = true }
# ///


# Example of command uv run removal.py --dataset fashion-mnist --output_dir output --epochs 10000 --runs 10
# %%
#!%env PYTORCH_ENABLE_MPS_FALLBACK=1

# %%
import argparse
import os
import random
import sys
from copy import deepcopy
from functools import partial

import requests
import torch
import torch.nn.functional as tF
from deep_augments import random_crop, random_erase, random_flip
from deep_datasets import DATASETS
from deep_utils import (
    InMemoryDataLoader,
    add_default_arguments,
    eval,
    fit,
    result_file_name,
    subset,
    tensorize,
)
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from experiments.nng_utils import last_layer, supported_kfac_layers
from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.jacobian import Jacobian
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM, FIM_MonteCarlo, sqrt_var_classif_logits
from nngeometry.object.pspace import PMatBlockDiag, PMatDense, PMatEKFAC, PMatKFAC

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def is_interactive():
    return "ipykernel" in sys.modules


if is_interactive():
    sys.argv = [""]

parser = argparse.ArgumentParser()
add_default_arguments(parser, "mislabeled")
parser.add_argument("--noise_ratio", type=float, default=0.25)
parser.add_argument("--correction_ratio", type=float, default=0.15)

args = parser.parse_args()

args.dataset = "mnist-1d"
args.runs = 1
args.regul = 1e-4
# args.ft = True
args.epochs = 10
args.repr = "F_kfac"
args.competitors = True
args.seed = 1337
args.ft = False
args.output_dir = "mislabeled-output"
args.device = "cuda"


os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)

transform, dataset, model_fn, optimizer_fn, classes = DATASETS[args.dataset]

# model_fn = partial(model_fn, c=16)
model_fn = partial(model_fn, c=32, h=2, d=2)

if transform is not None:
    if hasattr(transform, "transforms"):
        deterministic_transform = transforms.Compose(transform.transforms[-2:])
    else:
        deterministic_transform = transform
else:
    deterministic_transform = None

train_set = dataset(root=args.data_dir, train=True, transform=deterministic_transform)
test_set = dataset(root=args.data_dir, train=False, transform=deterministic_transform)

train_set = tensorize(train_set)
test_set = tensorize(test_set)

DataLoader = lambda dataset, **kwargs: InMemoryDataLoader(  # noqa: E731
    *dataset.tensors, device=args.device, **kwargs
)


test = DataLoader(test_set, batch_size=1024)

strategies = [
    "margin",
    "influence",
    "cook",
    "cook_bar",
    "cook_I",
    "aloo",
    "random",
    "loss",
]
records = []


if "cifar" in args.dataset:
    max_batch_size = 128
    augmentations = [random_flip(0.5), random_crop(4)]  # , random_erase(0.5, 3)]
    # augmentations = []
else:
    max_batch_size = 128
    augmentations = []


if args.dataset == "cifar-10":
    fname = "CIFAR-10_human.pt"
    if not os.path.exists(os.path.join(args.data_dir, fname)):
        r = requests.get(
            f"https://github.com/UCSC-REAL/cifar-10-100n/raw/refs/heads/main/data/{fname}"
        )
        open(os.path.join(args.data_dir, fname), "wb").write(r.content)
    noises = torch.load(os.path.join(args.data_dir, fname), weights_only=False)

train = DataLoader(train_set, batch_size=128, shuffle=True)
model = deepcopy(model_fn()).to(device=args.device)
optimizer = optimizer_fn(model.parameters())

fit(
    model,
    optimizer,
    train,
    args.epochs,
    args.device,
    augmentations=augmentations,
)
print(eval(model, train, args.device))
print(eval(model, test, args.device))
# %%
for i in range(args.runs):
    noisy_train_set = deepcopy(train_set)
    noisy_probability = torch.rand(noisy_train_set.tensors[1].shape[0])
    noisy_examples = noisy_probability <= args.noise_ratio
    noisy_labels = torch.randint(0, len(classes), (torch.sum(noisy_examples),))
    noisy_train_set.tensors[1][noisy_examples] = noisy_labels

    noisy_train = DataLoader(noisy_train_set, batch_size=128, shuffle=True)

    model = deepcopy(model_fn()).to(device=args.device)
    optimizer = optimizer_fn(model.parameters())

    fit(
        model,
        optimizer,
        noisy_train,
        args.epochs,
        args.device,
        augmentations=augmentations,
    )
    none_accuracy, none_test_loss = eval(model, test, args.device)

    record = {}
    record["strategy"] = "none"
    record["correction"] = "none"
    record["test_accuracy"] = none_accuracy
    record["test_loss"] = none_test_loss
    record["run"] = i
    record["auc"] = 0
    print(record)
    records.append(record)
    N = noisy_examples.shape[0]
    index_loo = random.randint(0, N)
    # %%
    args.repr = "F_kfac"
    if "last" in args.repr:
        repr = PMatDense
        lc = last_layer(LayerCollection.from_model(model))
    elif "kfac" in args.repr:
        repr = partial(PMatKFAC, strategy="one_iter_kpsvd")
        # repr = PMatKFAC
        # repr = PMatEKFAC
        # repr = partial(PMatEKFAC, strategy="one_iter_kpsvd")
        lc = supported_kfac_layers(LayerCollection.from_model(model))
    elif "bd" in args.repr:
        repr = PMatBlockDiag
        lc = supported_kfac_layers(LayerCollection.from_model(model))

    F = FIM(
        model,
        noisy_train,
        repr,
        "classif_logits",
        device=args.device,
        layer_collection=lc,
        verbose=True,
    )

    def loss(inputs, targets):
        return tF.cross_entropy(model(inputs), targets, reduction="none")

    def func(inputs, targets):
        return sqrt_var_classif_logits(model(inputs))

    def logits(inputs, targets):
        return model(inputs)

    if hasattr(F, "update_diag"):
        F.update_diag(noisy_train)

    N = noisy_examples.shape[0]
    if hasattr(F, "__rmul__"):
        F = N * F
    else:  # kfac
        for layer_id, layer in tqdm(lc.layers.items()):
            a, g = F.data[layer_id]
            a *= N**0.5
            g *= N**0.5
    noisy_train_loo = DataLoader(
        subset(noisy_train_set, torch.argwhere(torch.arange(N) != index_loo).squeeze()),
        batch_size=128,
        shuffle=True,
    )
    noisy_train_removed = DataLoader(
        subset(
            noisy_train_set, torch.argwhere(torch.arange(N) == index_loo).squeeze(0)
        ),
        batch_size=128,
        shuffle=True,
    )

    F_loo = FIM(
        model,
        noisy_train_loo,
        repr,
        "classif_logits",
        device=args.device,
        layer_collection=lc,
        verbose=True,
    )

    if hasattr(F_loo, "update_diag"):
        F_loo.update_diag(noisy_train)

    N = noisy_examples.shape[0] - 1
    if hasattr(F_loo, "__rmul__"):
        F_loo = N * F_loo
    else:  # kfac
        for layer_id, layer in tqdm(lc.layers.items()):
            a, g = F_loo.data[layer_id]
            a *= N**0.5
            g *= N**0.5
    # %%
    args.regul = 1e0
    if isinstance(F, PMatKFAC):
        cache = {}
        cache_loo = {}

        def solve(A, B, scale=1.0):
            evals, evecs = A
            projected = torch.einsum("de,ondk->onek", evecs, B)
            projected /= scale * evals[None, None, :, None] + args.regul**0.5
            return torch.einsum("de,onek->ondk", evecs, projected)

        def woodbury_downdate_solve(solve, A, B, U):
            FinvG = solve(A, B)
            FinvJ = solve(A, U)
            leverage = torch.einsum("onij, Onij->noO", U, FinvJ)
            cross = torch.einsum("onij, Onij->noO", U, FinvG)
            schur = (
                torch.eye(
                    leverage.shape[-1],
                    dtype=leverage.dtype,
                    device=leverage.device,
                )[None, ...]
                - leverage
            )
            effect = torch.linalg.solve(schur, cross)
            return FinvG + torch.einsum("onij,noO->Onij", FinvJ, effect)

        for layer_id, layer in tqdm(lc.layers.items()):
            a, g = F.data[layer_id]
            trace = torch.trace(g)

            if layer.transposed:
                a, g = g, a

            cache[layer_id] = {
                "eig_a": torch.linalg.eigh(a),
                "eig_g": torch.linalg.eigh(g),
                "trace": trace,
            }

            a, g = F_loo.data[layer_id]
            if layer.transposed:
                a, g = g, a
            cache_loo[layer_id] = {
                "eig_a": torch.linalg.eigh(a),
                "eig_g": torch.linalg.eigh(g),
            }

        for inputs, targets in tqdm(noisy_train_removed):
            pfmap_func = Jacobian(
                model,
                (inputs, targets),
                function=func,
                layer_collection=lc,
            )
            pfmap_grad = Jacobian(
                model,
                (inputs, targets),
                function=loss,
                layer_collection=lc,
            )
            for layer_id, layer in lc.layers.items():
                if layer.has_bias():
                    Gw, Gb = pfmap_grad.to_torch_layer(layer_id)
                else:
                    Gw = pfmap_grad.to_torch_layer(layer_id)[0]
                sG = Gw.size()
                Gw = Gw.reshape(sG[0], sG[1], sG[2], -1)
                if layer.has_bias():
                    G = torch.cat([Gw, Gb.unsqueeze(-1)], dim=-1)

                if layer.has_bias():
                    Jw, Jb = pfmap_func.to_torch_layer(layer_id)
                else:
                    Jw = pfmap_func.to_torch_layer(layer_id)[0]
                sJ = Jw.size()
                Jw = Jw.reshape(sJ[0], sJ[1], sJ[2], -1)
                if layer.has_bias():
                    J = torch.cat([Jw, Jb.unsqueeze(-1)], dim=-1)

                solve_g = solve(cache[layer_id]["eig_g"], G)
                solve_ga = solve(
                    cache[layer_id]["eig_a"], solve_g.transpose(-1, -2)
                ).transpose(-1, -2)
                si = torch.einsum("onij, Onij->noO", G, solve_ga)

                trace = cache[layer_id]["trace"]
                trace_loo = torch.sqrt(
                    cache[layer_id]["trace"] ** 2
                    - (J**2).sum(dim=(0, 2, 3), keepdim=True)
                )
                solve_g_loo = woodbury_downdate_solve(
                    partial(solve, scale=trace / trace_loo),
                    cache[layer_id]["eig_g"],
                    G,
                    J * trace_loo.rsqrt(),
                )
                solve_ga_loo = woodbury_downdate_solve(
                    partial(solve, scale=trace / trace_loo),
                    cache[layer_id]["eig_a"],
                    solve_g_loo.transpose(-1, -2),
                    J.transpose(-1, -2) * trace_loo.rsqrt(),
                ).transpose(-1, -2)
                si_loo = torch.einsum("onij, Onij->noO", G, solve_ga_loo)

                solve_g_loo_true = solve(cache_loo[layer_id]["eig_g"], G)
                solve_ga_loo_true = solve(
                    cache_loo[layer_id]["eig_a"],
                    solve_g_loo_true.transpose(-1, -2),
                ).transpose(-1, -2)
                si_loo_true = torch.einsum("onij, Onij->noO", G, solve_ga_loo_true)

                print(
                    si.squeeze(),
                    si_loo.squeeze(),
                    si_loo_true.squeeze(),
                )
                torch.testing.assert_close(si_loo, si_loo_true)
                # %%

    elif isinstance(F, PMatBlockDiag):
        cache = {}
        cache_loo = {}

        for layer_id, layer in tqdm(lc.layers.items()):
            block = F.data[layer_id]

            block_reg = block + args.regul * torch.eye(
                block.shape[0], dtype=block.dtype, device=block.device
            )

            cache[layer_id] = torch.linalg.cholesky(block_reg)

            def solve(A, B):
                return torch.cholesky_solve(B, A)

            block = F_loo.data[layer_id]

            block_reg = block + args.regul * torch.eye(
                block.shape[0], dtype=block.dtype, device=block.device
            )

            cache_loo[layer_id] = torch.linalg.cholesky(block_reg)

        for inputs, targets in tqdm(noisy_train_removed):
            pfmap_func = Jacobian(
                model,
                (inputs, targets),
                function=func,
                layer_collection=lc,
            )
            pfmap_grad = Jacobian(
                model,
                (inputs, targets),
                function=loss,
                layer_collection=lc,
            )
            for layer_id, layer in lc.layers.items():
                if layer.has_bias():
                    Gw, Gb = pfmap_grad.to_torch_layer(layer_id)
                else:
                    Gw = pfmap_grad.to_torch_layer(layer_id)[0]
                sG = Gw.size()
                G = Gw.view(sG[0], sG[1], -1)
                if layer.has_bias():
                    G = torch.cat([G, Gb.view(sG[0], sG[1], -1)], dim=2)

                if layer.has_bias():
                    Jw, Jb = pfmap_func.to_torch_layer(layer_id)
                else:
                    Jw = pfmap_func.to_torch_layer(layer_id)[0]
                sJ = Jw.size()
                J = Jw.view(sJ[0], sJ[1], -1)
                if layer.has_bias():
                    J = torch.cat([J, Jb.view(sJ[0], sJ[1], -1)], dim=2)

                G = G.permute(1, 0, 2)
                J = J.permute(1, 0, 2)

                block_si = torch.einsum(
                    "nop, nOp->noO",
                    G,
                    solve(cache[layer_id], G.transpose(1, 2)).transpose(1, 2),
                )
                block_FinvJ = solve(cache[layer_id], J.transpose(1, 2)).transpose(1, 2)
                block_leverage = torch.einsum("nop, nOp->noO", J, block_FinvJ)
                block_cross = torch.einsum("nop, nOp->noO", G, block_FinvJ)
                block_schur = (
                    torch.eye(block_leverage.shape[-1], device=args.device)
                    - block_leverage
                )
                loo_effect = torch.linalg.solve(
                    block_schur, block_cross.transpose(1, 2)
                )
                block_si_loo = block_si + block_cross @ loo_effect
                block_si_loo_true = torch.einsum(
                    "nop, nOp->noO",
                    G,
                    solve(cache_loo[layer_id], G.transpose(1, 2)).transpose(1, 2),
                )

                print(
                    block_si.squeeze(),
                    block_si_loo.squeeze(),
                    block_si_loo_true.squeeze(),
                )
                torch.testing.assert_close(block_si_loo, block_si_loo_true)

    elif isinstance(F, PMatDense):
        for inputs, targets in tqdm(noisy_train_removed):
            pfmap_func = Jacobian(
                model,
                (inputs, targets),
                function=func,
                layer_collection=lc,
            )
            pfmap_grad = Jacobian(
                model,
                (inputs, targets),
                function=loss,
                layer_collection=lc,
            )
            si = torch.einsum(
                "onp, Onp->noO",
                pfmap_grad.to_torch(),
                F.solve(pfmap_grad, regul=args.regul).to_torch(),
            )
            FinvJ = F.solve(pfmap_func, regul=args.regul).to_torch()
            leverage = torch.einsum(
                "onp, Onp->noO",
                pfmap_func.to_torch(),
                FinvJ,
            )
            cross = torch.einsum(
                "onp, Onp->noO",
                pfmap_grad.to_torch(),
                FinvJ,
            )

            schur = torch.eye(leverage.shape[-1], device=args.device) - leverage
            loo_effect = torch.linalg.solve(schur, cross.transpose(1, 2))
            torch.testing.assert_close(
                torch.einsum(
                    "onp, Onp->noO",
                    pfmap_grad.to_torch(),
                    F_loo.solve(pfmap_grad, regul=args.regul).to_torch(),
                ),
                si + cross @ loo_effect,
            )

# %%
