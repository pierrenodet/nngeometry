# /// script
# dependencies = [
#  "deepgnostics",
#  "seaborn",
#  "scipy",
#  "pandas",
#  "joblib",
#  "scikit-learn",
#  "requests",
#  "mnist1d",
# ]
# [tool.uv.sources]
# deepgnostics = { path = "../.." , editable = true}
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

import pandas as pd
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
from nngeometry.metrics import FIM, FIM_MonteCarlo
from nngeometry.object.pspace import PMatDense, PMatEKFAC, PMatKFAC

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
args.regul = 1e-2
# args.ft = True
args.epochs = 50
args.repr = "F_kfac"
args.competitors = True
args.seed = 1337
args.ft = False
args.output_dir = "mislabeled-output"
args.device = "cuda:0"


os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)

transform, dataset, model_fn, optimizer_fn, classes = DATASETS[args.dataset]

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
    # %%

    if "last" in args.repr:
        repr = PMatDense
        lc = last_layer(LayerCollection.from_model(model))
    elif "kfac" in args.repr:
        repr = partial(PMatKFAC, strategy="one_iter_kpsvd")
        # repr = PMatKFAC
        # repr = PMatEKFAC
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

    # F = repr(
    #     lc,
    #     TorchHooksJacobianBackend(
    #         model,
    #         lambda inputs, targets: tF.cross_entropy(
    #             model(inputs), targets, reduction="none"
    #         ),
    #         centering=False,
    #         verbose=True,
    #     ),
    #     examples=noisy_train,
    # )
    def loss(inputs, targets):
        return tF.cross_entropy(model(inputs), targets, reduction="none")

    def func(inputs, targets):
        log_prob = tF.log_softmax(model(inputs), dim=1)
        prob = torch.exp(log_prob).detach()
        return torch.sqrt(prob) * log_prob

    def logits(inputs, targets):
        return model(inputs)

    if hasattr(F, "update_diag"):
        F.update_diag(noisy_train)

    N = noisy_examples.shape[0]
    # if hasattr(F, "__rmul__"):
    F = N * F
    # else:  # kfac
    #     for layer_id, layer in tqdm(lc.layers.items()):
    #         a, g = F.data[layer_id]
    #         a *= N
    #         g *= N

    # %%
    args.regul = 1e-5
    light_noisy_noaug = DataLoader(noisy_train_set, batch_size=32)

    self_influence = []
    self_influence_loo = []

    if isinstance(F, PMatEKFAC):
        evecs, evals = F.data

        for inputs, targets in tqdm(light_noisy_noaug):
            si = 0
            si_loo = 0
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
            # pfmap_logits = Jacobian(
            #     model,
            #     (inputs, targets),
            #     function=logits,
            #     layer_collection=lc,
            # )
            # with torch.no_grad():
            #     p = torch.softmax(model(inputs), dim=1)
            #     y = tF.one_hot(targets, num_classes=len(classes)).float()
            for layer_id, layer in lc.layers.items():
                # Jl = pfmap_logits.to_torch_layer(layer_id)
                # Jlkfe = F._proj_to_kfe_batched(Jl, evecs[layer_id], layer)

                # mean_Jlkfe = torch.einsum("nc,cnij->nij", p, Jlkfe)
                # Jlkfe_centered = Jlkfe - mean_Jlkfe.unsqueeze(0)
                # Jkfe = torch.einsum("nc,cnij->cnij", torch.sqrt(p), Jlkfe_centered)
                # Gkfe = torch.einsum("nc, cnij->nij", p - y, Jlkfe).unsqueeze(0)

                J = pfmap_func.to_torch_layer(layer_id)
                Jkfe = F._proj_to_kfe_batched(J, evecs[layer_id], layer)
                G = pfmap_grad.to_torch_layer(layer_id)
                Gkfe = F._proj_to_kfe_batched(G, evecs[layer_id], layer)

                sG = Gkfe.size()

                Gkfe_scaled = Gkfe / (evals[layer_id].view(1, 1, *sG[2:]) + args.regul)

                si += torch.einsum("onij, Onij->noO", Gkfe_scaled, Gkfe)

                Gkfe_loo_scaled = Gkfe / (
                    (evals[layer_id].view(1, 1, *sG[2:]) + args.regul)
                    - (Jkfe**2).sum(dim=0)
                )
                si_loo += torch.einsum("onij, Onij->noO", Gkfe_loo_scaled, Gkfe)

            self_influence.append(si.detach().cpu())
            self_influence_loo.append(si_loo.detach().cpu())
    elif isinstance(F, PMatKFAC):
        cache = {}

        for layer_id, layer in tqdm(lc.layers.items()):
            a, g = F.data[layer_id]

            if layer.transposed:
                a, g = g, a

            a_reg = a + args.regul**0.5 * torch.eye(
                a.shape[0], dtype=a.dtype, device=a.device
            )
            g_reg = g + args.regul**0.5 * torch.eye(
                g.shape[0], dtype=g.dtype, device=g.device
            )

            cache[layer_id] = {
                "La": torch.linalg.cholesky(a_reg),
                "Lg": torch.linalg.cholesky(g_reg),
            }

        for inputs, targets in tqdm(light_noisy_noaug):
            si = 0
            si_loo = 0
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
                bs = sG[0] * sG[1]
                G = Gw.view(bs, sG[2], -1)
                if layer.has_bias():
                    G = torch.cat([G, Gb.view(bs, -1, 1)], dim=2)

                solve_g = torch.cholesky_solve(G, cache[layer_id]["Lg"])
                solve_ga = torch.cholesky_solve(
                    solve_g.transpose(1, 2), cache[layer_id]["La"]
                )
                solve_ga = solve_ga.transpose(1, 2)
                si += torch.einsum(
                    "onp, Onp->noO",
                    G.view(sG[0], sG[1], -1),
                    solve_ga.view(sG[0], sG[1], -1),
                )

                solve_a = torch.cholesky_solve(G.transpose(1, 2), cache[layer_id]["La"])

                schur_g = torch.eye(G.size(2), device=args.device).unsqueeze(
                    0
                ) - torch.bmm(G.transpose(1, 2), solve_g)
                schur_a = torch.eye(G.size(1), device=args.device).unsqueeze(
                    0
                ) - torch.bmm(G, solve_a)

                print(torch.vmap(torch.det)(schur_a),torch.vmap(torch.det)(schur_g))

                si_loo += torch.einsum(
                    "onp, Onp->noO",
                    torch.linalg.solve(schur_a, solve_a.transpose(1, 2)).reshape(
                        sG[0], sG[1], -1
                    ),
                    torch.linalg.solve(schur_g, solve_g.transpose(1, 2))
                    .transpose(1, 2)
                    .reshape(sG[0], sG[1], -1),
                )

            self_influence.append(si.detach().cpu())
            self_influence_loo.append(si_loo.detach().cpu())
    elif isinstance(F, PMatDense):
        for inputs, targets in tqdm(light_noisy_noaug):
            pfmap = Jacobian(
                model,
                (inputs, targets),
                function=func,
                layer_collection=lc,
            )
            si = (pfmap.to_torch() * F.solve(pfmap, regul=args.regul).to_torch()).sum(
                dim=(0, 2)
            )
            si_loo = si / (1 - si)
        self_influence.append(si)
        self_influence_loo.append(si_loo)
    # %%
    self_influence = torch.cat(self_influence)
    self_influence_loo = torch.cat(self_influence_loo)

    # %%
    loss = []
    for inputs, targets in tqdm(light_noisy_noaug):
        with torch.no_grad():
            loss.append(tF.cross_entropy(model(inputs), targets, reduction="none"))
    loss = torch.cat(loss)

    # %%
    tr_self_influence = torch.vmap(torch.trace)(self_influence)
    tr_self_influence_loo = torch.vmap(torch.trace)(self_influence_loo)

    # %%
    print(
        roc_auc_score(noisy_examples, tr_self_influence.numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence_loo.numpy(force=True)),
        roc_auc_score(noisy_examples, loss.numpy(force=True)),
    )
    # %%
    print(
        kendalltau(
            tr_self_influence.numpy(force=True), tr_self_influence_loo.numpy(force=True)
        ).statistic
    )


# %%
# %%
import matplotlib.pyplot as plt

plt.scatter(
    torch.argsort(torch.argsort(tr_self_influence_loo)),
    torch.argsort(torch.argsort(tr_self_influence)),
    s=1,
    # c=noisy_train_set.tensors[1]
    c=noisy_examples,
)
# %%
plt.scatter(tr_self_influence, tr_self_influence_loo)
plt.axline((0, 0), slope=1)
# %%
self_influence[0].diag()
# %%
noisy_train_set.tensors[1][0]
# %%
self_influence_loo[0].diag()

# %%
