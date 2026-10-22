# /// script
# dependencies = [
#  "nngeometry",
#  "scikit-learn",
#  "mnist1d",
#  "opt-einsum",
#  "h5py",
#  "datasets"
# ]
# [tool.uv.sources]
# nngeometry = { path = ".." , editable = true}
# ///

# %%
import argparse
import json
import os
import random
import sys
from copy import deepcopy
from functools import partial
from itertools import chain, product

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import h5py
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from experiments.nng_utils import last_layer, supported_kfac_layers
from nngeometry.backend import TorchFuncJacobianBackend, TorchHooksJacobianBackend
from nngeometry.jacobian import Jacobian
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM, FIM_MonteCarlo, sqrt_var_classif_logits
from nngeometry.object.fspace import FMatDense
from nngeometry.object.logra import PMatLoGRA
from nngeometry.object.map import (
    PFMapAdjoint,
    PFMapDense,
    PFMapFactored,
    PFMapImplicit,
    random_pfmap,
)
from nngeometry.object.pspace import (
    PMatBlockDiag,
    PMatDense,
    PMatDiag,
    PMatEKFAC,
    PMatImplicit,
    PMatKFAC,
    PMatLowRank,
)
from nngeometry.object.vector import FVector, PVector
from nngeometry.solve import qr

# %%


def is_interactive():
    return "ipykernel" in sys.modules


if is_interactive():
    sys.argv = [""]

parser = argparse.ArgumentParser()
add_default_arguments(parser, "mislabeled")
parser.add_argument("--noise_ratio", type=float, default=0.25)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--eval_size", type=int, default=500)
parser.add_argument("--train_size", type=int, default=None)

args = parser.parse_args()

args.dataset = "mnist-1d"
args.runs = 1
args.seed = 1337
args.epochs = 2000
args.device = "cuda"
os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)

transform, dataset, model_fn, optimizer_fn, classes = DATASETS[args.dataset]

if isinstance(optimizer_fn, tuple):
    optimizer_fn, scheduler_fn = optimizer_fn
else:
    scheduler_fn = None


if transform is not None:
    if hasattr(transform, "transforms"):
        deterministic_transform = transforms.Compose(transform.transforms[-2:])
    else:
        deterministic_transform = transform
else:
    deterministic_transform = None

train_set = dataset(root=args.data_dir, train=True, transform=deterministic_transform)
test_set = dataset(root=args.data_dir, train=False, transform=deterministic_transform)

train_size = len(train_set) if args.train_size is None else args.train_size
train_idx = torch.randperm(len(train_set))[:train_size]

train_set = subset(tensorize(train_set), train_idx)
test_set = tensorize(test_set)

DataLoader = lambda dataset, **kwargs: InMemoryDataLoader(  # noqa: E731
    *dataset.tensors, device=args.device, **kwargs
)


test = DataLoader(test_set, batch_size=1024)
train = DataLoader(train_set, batch_size=128, shuffle=True)

noisy_train_set = deepcopy(train_set)
targets = noisy_train_set.tensors[1]
noisy_probability = torch.rand(targets.shape[0])
noisy_examples = noisy_probability <= args.noise_ratio
noisy_labels = (
    targets[noisy_examples]
    + torch.randint(1, len(classes), (torch.sum(noisy_examples),))
) % len(classes)
noisy_train_set.tensors[1][noisy_examples] = noisy_labels

noisy_train = DataLoader(noisy_train_set, batch_size=128, shuffle=True)

# %%

model_fn = partial(model_fn, c=32, d=2, h=4)

# model_clean = torch.compile(deepcopy(model_fn()).to(device=args.device))
# optimizer_clean = optimizer_fn(model_clean.parameters())

model = torch.compile(deepcopy(model_fn()).to(device=args.device))
optimizer = optimizer_fn(model.parameters())
scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None

# fit(
#     model_clean,
#     optimizer_clean,
#     train,
#     epochs,
#     args.device,
# )
# gold_train_accuracy, gold_train_loss = eval(model_clean, train, args.device)
# gold_test_accuracy, gold_test_loss = eval(model_clean, test, args.device)
# print(gold_train_accuracy, gold_train_loss)
# print(gold_test_accuracy, gold_test_loss)

fit(
    model,
    optimizer,
    noisy_train,
    args.epochs,
    args.device,
)
none_ntrain_accuracy, none_ntrain_loss = eval(model, noisy_train, args.device)
none_train_accuracy, none_train_loss = eval(model, train, args.device)
none_test_accuracy, none_test_loss = eval(model, test, args.device)
print(none_ntrain_accuracy, none_ntrain_loss)
print(none_train_accuracy, none_train_loss)
print(none_test_accuracy, none_test_loss)

model = model._orig_mod

# N = train_set.tensors[0].shape[0]
# models = []
# subset_size = 1000 if not args.loo else N - 1
# n_subsets = 100 if args.subsets is None else args.subsets
# subset_indices = [torch.randperm(N)[:subset_size] for _ in range(100)]

# for subset_idx in tqdm(subset_indices):
#     model = torch.compile(deepcopy(model_fn()).to(device=args.device))
#     optimizer = optimizer_fn(model.parameters())
#     scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None
#     subset_noisy_train = DataLoader(
#         subset(noisy_train_set, subset_idx), batch_size=128, shuffle=True
#     )

#     fit(
#         model,
#         optimizer,
#         subset_noisy_train,
#         epochs,
#         args.device,
#         scheduler=scheduler,
#         verbose=False,
#     )
#     models.append(model)


# %%
def func(inputs, targets):
    return sqrt_var_classif_logits(model(inputs))


def logits(inputs, targets):
    return model(inputs)


# def func(inputs, targets):
#     logp = tF.log_softmax(model(inputs), dim=1)
#     p = torch.exp(logp).detach()
#     return torch.sqrt(p) * logp


def loss(inputs, targets):
    return tF.cross_entropy(model(inputs), targets, reduction="none")


backend = TorchHooksJacobianBackend(model, func, verbose=True)

lc = supported_kfac_layers(
    LayerCollection.from_model(model, ignore_unsupported_layers=True)
)

# %%
F_dense = PMatDense(
    generator=backend,
    examples=DataLoader(noisy_train_set, batch_size=128, shuffle=False),
    layer_collection=lc,
)

N = noisy_train_set.tensors[0].shape[0]
F_dense = N * F_dense

F_logra_diag = PMatLoGRA(
    generator=backend,
    examples=DataLoader(noisy_train_set, batch_size=128, shuffle=False),
    layer_collection=lc,
    projections="kfac",
    representation=PMatDiag,
    rank=1.0,
)
F_logra_dense = PMatLoGRA(
    generator=backend,
    examples=DataLoader(noisy_train_set, batch_size=128, shuffle=False),
    layer_collection=lc,
    projections="kfac",
    representation=PMatDense,
    rank=1.0,
)


print(lc.numel())

F_logra_dense.F = N * F_logra_dense.F
F_logra_diag.F = N * F_logra_diag.F

# %%
eval_size = N if args.eval_size is None else args.eval_size
eval_idx = torch.randperm(N)[:eval_size]
light_noisy_noaug = DataLoader(
    subset(noisy_train_set, eval_idx), shuffle=False, batch_size=4
)
# %%
# F_dense.compute_eigendecomposition()
# F_logra_dense.F.compute_eigendecomposition()

# %%
results = {}

# %% logra dense

results["logra"] = {
    key: []
    for key in [
        "si",
        "si_loo",
        "leverage",
        "leverage_loo",
        "cook",
        # "projections",
        # "projections_w",
    ]
}

for inputs, targets in tqdm(light_noisy_noaug):
    pfmap_func = Jacobian(
        model,
        (inputs, targets),
        function=func,
        layer_collection=lc,
        representation=PFMapFactored,
    )
    pfmap_grad = Jacobian(
        model,
        (inputs, targets),
        function=loss,
        layer_collection=lc,
        representation=PFMapFactored,
    )
    proj_grad = F_logra_dense.project(pfmap_grad)
    proj_func = F_logra_dense.project(pfmap_func)
    FinvG = F_logra_dense.F.solve(proj_grad, regul=args.regul).to_torch()
    self_influence = torch.einsum("onp, Onp->noO", proj_grad.to_torch(), FinvG)
    FinvJ = F_logra_dense.F.solve(proj_func, regul=args.regul).to_torch()
    leverage = torch.einsum("onp, Onp->noO", proj_func.to_torch(), FinvJ)
    cross = torch.einsum("onp, Onp->noO", proj_grad.to_torch(), FinvJ)
    schur = torch.eye(leverage.shape[-1], device=args.device) - leverage
    loo_effect = torch.linalg.solve(schur, cross.transpose(1, 2))
    results["logra"]["si"].append(self_influence)
    results["logra"]["si_loo"].append(self_influence + cross @ loo_effect)
    results["logra"]["leverage"].append(leverage)
    results["logra"]["leverage_loo"].append(
        torch.linalg.solve(schur, leverage.transpose(1, 2))
    )
    results["logra"]["cook"].append(
        self_influence + cross @ loo_effect + loo_effect.transpose(1, 2) @ loo_effect
    )
    # results["logra"]["projections"].append(
    #     (proj_grad.to_torch() @ F_logra_dense.F.evecs).square().sum(dim=0)
    # )
    # results["logra"]["projections_w"].append(
    #     results["logra"]["projections"][-1]
    #     / (F_logra_dense.F.evals[None, ...] + args.regul)
    # )

# %%

results["logra_diag"] = {
    key: []
    for key in [
        "si",
        "si_loo",
        "leverage",
        "leverage_loo",
        "cook",
        # "projections",
        # "projections_w",
    ]
}

for inputs, targets in tqdm(light_noisy_noaug):
    pfmap_func = Jacobian(
        model,
        (inputs, targets),
        function=func,
        layer_collection=lc,
        representation=PFMapFactored,
    )
    pfmap_grad = Jacobian(
        model,
        (inputs, targets),
        function=loss,
        layer_collection=lc,
        representation=PFMapFactored,
    )
    proj_grad = F_logra_diag.project(pfmap_grad).to_torch()
    proj_func = F_logra_diag.project(pfmap_func).to_torch()
    D = F_logra_diag.F.data
    D_loo = D - (proj_func**2).sum(dim=0)
    results["logra_diag"]["si"].append(
        torch.einsum(
            "onp,Onp,p->noO",
            proj_grad,
            proj_grad,
            1 / (D + args.regul),
        )
    )
    results["logra_diag"]["si_loo"].append(
        torch.einsum(
            "onp,Onp,np->noO",
            proj_grad,
            proj_grad,
            1 / (D_loo + args.regul),
        )
    )
    results["logra_diag"]["cook"].append(
        torch.einsum(
            "onp,Onp,np->noO",
            proj_grad,
            proj_grad,
            (D + args.regul) / (D_loo + args.regul) ** 2,
        )
    )
    results["logra_diag"]["leverage"].append(
        torch.einsum(
            "onp,Onp,p->noO",
            proj_func,
            proj_func,
            1 / (D + args.regul),
        )
    )
    results["logra_diag"]["leverage_loo"].append(
        torch.einsum(
            "onp,Onp,np->noO",
            proj_func,
            proj_func,
            1 / (D_loo + args.regul),
        )
    )
    ranking = torch.argsort(F_logra_diag.F.data)
    # results["logra_diag"]["projections"].append(
    #     proj_grad.square().sum(dim=0)[:, ranking]
    # )
    # results["logra_diag"]["projections_w"].append(
    #     (results["logra_diag"]["projections"][-1] / (F_logra_diag.F.data + args.regul))[
    #         :, ranking
    #     ]
    # )

# %%
results["dense"] = {
    key: []
    for key in [
        "si",
        "si_loo",
        "leverage",
        "leverage_loo",
        "cook",
        "grad",
        # "projections",
        # "projections_w",
    ]
}

for inputs, targets in tqdm(light_noisy_noaug):
    pfmap_func = Jacobian(
        model,
        (inputs, targets),
        function=func,
        layer_collection=lc,
        representation=PFMapDense,
    )
    pfmap_grad = Jacobian(
        model,
        (inputs, targets),
        function=loss,
        layer_collection=lc,
        representation=PFMapDense,
    )
    FinvG = F_dense.solve(pfmap_grad, regul=args.regul).to_torch()
    self_influence = torch.einsum(
        "onp, Onp->noO",
        pfmap_grad.to_torch(),
        FinvG,
    )
    FinvJ = F_dense.solve(pfmap_func, regul=args.regul).to_torch()
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
    results["dense"]["si"].append(self_influence)
    results["dense"]["si_loo"].append(self_influence + cross @ loo_effect)
    results["dense"]["leverage"].append(leverage)
    results["dense"]["leverage_loo"].append(
        torch.linalg.solve(schur, leverage.transpose(1, 2))
    )
    results["dense"]["cook"].append(
        self_influence + cross @ loo_effect + loo_effect.transpose(1, 2) @ loo_effect
    )
    results["dense"]["grad"].append(pfmap_grad.to_torch().sum(dim=0))
    # results["dense"]["projections"].append(
    #     (pfmap_grad.to_torch() @ F_dense.evecs).square().sum(dim=0)
    # )
    # results["dense"]["projections_w"].append(
    #     results["dense"]["projections"][-1] / (F_dense.evals[None, ...] + args.regul)
    # )

# %%
for n_iter in range(1, 4):
    results[f"cg_{n_iter}"] = {
        key: []
        for key in [
            "si",
            "si_loo",
            "leverage",
            "leverage_loo",
            "cook",
            # "projections",
            # "projections_w",
        ]
    }
F_kfac = PMatEKFAC(
    lc, backend, examples=DataLoader(noisy_train_set, batch_size=128, shuffle=False)
)
F_kfac.update_diag(DataLoader(noisy_train_set, batch_size=128, shuffle=False))
F_kfac = N * F_kfac
for inputs, targets in tqdm(light_noisy_noaug):
    pfmap_func = Jacobian(
        model,
        (inputs, targets),
        function=func,
        layer_collection=lc,
        representation=PFMapDense,
    )
    pfmap_grad = Jacobian(
        model,
        (inputs, targets),
        function=loss,
        layer_collection=lc,
        representation=PFMapDense,
    )
    V = F_kfac.solve(pfmap_func, regul=args.regul).to_torch()

    for n_iter in range(1, 4):
        Q, _ = torch.linalg.qr(
            V.cpu().permute(1, 2, 0).contiguous(),
            mode="reduced",
        )
        V = Q.to(device=args.device).permute(2, 0, 1).contiguous()

        gV = torch.einsum("onp, Onp->noO", pfmap_grad.to_torch(), V)
        JV = torch.einsum("onp, Onp->noO", pfmap_func.to_torch(), V)

        FV = F_dense.mmap(PFMapDense(lc, backend, data=V)).to_torch() + args.regul * V
        VFV = torch.einsum("onp, Onp->noO", V, FV)
        VFV_loo = VFV - JV.transpose(-1, -2) @ JV

        alpha = torch.linalg.solve(VFV, JV.transpose(-1, -2))
        R = pfmap_func.to_torch() - torch.einsum("nko, knp->onp", alpha, FV)
        MinvR = F_kfac.solve(
            PFMapDense(lc, backend, data=R), regul=args.regul
        ).to_torch()
        V = torch.cat([V, MinvR], dim=0)

        results[f"cg_{n_iter}"]["si"].append(
            gV @ torch.linalg.solve(VFV, gV.transpose(-1, -2))
        )
        results[f"cg_{n_iter}"]["si_loo"].append(
            gV @ torch.linalg.solve(VFV_loo, gV.transpose(-1, -2))
        )
        results[f"cg_{n_iter}"]["leverage"].append(
            JV @ torch.linalg.solve(VFV, JV.transpose(-1, -2))
        )
        results[f"cg_{n_iter}"]["leverage_loo"].append(
            JV @ torch.linalg.solve(VFV_loo, JV.transpose(-1, -2))
        )
        results[f"cg_{n_iter}"]["cook"].append(
            (
                torch.linalg.solve(VFV_loo, gV.transpose(-1, -2)).transpose(-1, -2)
                @ VFV
                @ torch.linalg.solve(VFV_loo, gV.transpose(-1, -2))
            )
        )
# %%
# residuals = []
# pearson_residuals = []
# for inputs, targets in tqdm(light_noisy_noaug):
#     with torch.no_grad():
#         outputs = model(inputs)
#         p = torch.softmax(outputs, dim=1)
#         Y = torch.nn.functional.one_hot(targets, num_classes=p.shape[1])
#         residuals.append(torch.linalg.vector_norm((Y - p), dim=1))
#         logp = torch.nn.functional.log_softmax(outputs, dim=1)
#         mask = Y.bool()
#         pearson_residuals.append(torch.sqrt(torch.expm1(-logp[mask])))
# residuals = torch.cat(residuals)
# pearson_residuals = torch.cat(pearson_residuals)

# %%
for method in results.keys():
    for key in results[method].keys():
        results[method][key] = torch.cat(results[method][key], dim=0)

# %%
output_dir = os.path.join("toy-results", str(args.epochs))
os.makedirs(output_dir, exist_ok=True)

# %%
for method_name in results.keys():
    if method_name == "dense":
        continue
    for metric_name in ["si", "si_loo", "cook"]:
        plt.scatter(
            results["dense"][metric_name].numpy(force=True).squeeze(),
            results[method_name][metric_name].numpy(force=True).squeeze(),
            c=noisy_examples[eval_idx],
            s=12,
            edgecolors="black",
        )
        plt.xlabel("dense")
        plt.ylabel(method_name)
        plt.title(metric_name)
        os.makedirs(os.path.join(output_dir, method_name, metric_name), exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, method_name, metric_name, "values") + ".png"
        )
        plt.clf()
    for metric_name in ["leverage", "leverage_loo"]:
        plt.scatter(
            torch.vmap(torch.trace)(results["dense"][metric_name]).numpy(force=True),
            torch.vmap(torch.trace)(results[method_name][metric_name]).numpy(
                force=True
            ),
            c=noisy_examples[eval_idx],
            s=12,
            edgecolors="black",
        )
        plt.xlabel("dense")
        plt.ylabel(method_name)
        plt.title(metric_name)
        os.makedirs(os.path.join(output_dir, method_name, metric_name), exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, method_name, metric_name, "values") + ".png"
        )
        plt.clf()


# %%
def rank(a):
    return torch.argsort(torch.argsort(a))


for method_name in results.keys():
    if method_name == "dense":
        continue
    for metric_name in ["si", "si_loo", "cook"]:
        plt.scatter(
            rank(results["dense"][metric_name].squeeze()).numpy(force=True),
            rank(results[method_name][metric_name].squeeze()).numpy(force=True),
            c=noisy_examples[eval_idx],
            s=12,
            edgecolors="black",
        )
        plt.xlabel("dense")
        plt.ylabel(method_name)
        plt.title(metric_name)
        plt.savefig(
            os.path.join(output_dir, method_name, metric_name, "ranking") + ".png"
        )
        plt.clf()
    for metric_name in ["leverage", "leverage_loo"]:
        plt.scatter(
            rank(torch.vmap(torch.trace)(results["dense"][metric_name])).numpy(
                force=True
            ),
            rank(torch.vmap(torch.trace)(results[method_name][metric_name])).numpy(
                force=True
            ),
            c=noisy_examples[eval_idx],
            s=12,
            edgecolors="black",
        )
        plt.xlabel("dense")
        plt.ylabel(method_name)
        plt.title(metric_name)
        plt.savefig(
            os.path.join(output_dir, method_name, metric_name, "ranking") + ".png"
        )
        plt.clf()


# %%

for metric_name in ["si", "si_loo", "cook"]:
    plt.plot(
        range(0, 4),
        [
            spearmanr(
                results["dense"][metric_name].squeeze().numpy(force=True),
                results[method_name][metric_name].squeeze().numpy(force=True),
            ).statistic
            for method_name in ["logra_diag"] + [f"cg_{k}" for k in range(1, 4)]
        ],
        label=metric_name,
    )
    plt.xlabel("iterations")
    plt.ylabel("spearman correlation")
for metric_name in ["leverage", "leverage_loo"]:
    plt.plot(
        range(0, 4),
        [
            spearmanr(
                torch.vmap(torch.trace)(results["dense"][metric_name]).numpy(
                    force=True
                ),
                torch.vmap(torch.trace)(results[method_name][metric_name]).numpy(
                    force=True
                ),
            ).statistic
            for method_name in ["logra_diag"] + [f"cg_{k}" for k in range(1, 4)]
        ],
        label=metric_name,
    )
    plt.xlabel("iterations")
    plt.ylabel("spearman correlation")
plt.xticks(range(0, 4), ["ekfac"] + [str(i) for i in range(1, 4)])
plt.legend()
plt.savefig(os.path.join(output_dir, "cg-improvements") + ".png")
plt.clf()

# %%

data = []
for method_name in results.keys():
    datum = {}
    datum["method"] = method_name
    for metric_name in ["si", "si_loo", "cook"]:
        score = roc_auc_score(
            noisy_examples[eval_idx],
            results[method_name][metric_name].squeeze().numpy(force=True),
        )
        datum[metric_name] = score
    for metric_name in ["leverage", "leverage_loo"]:
        score = roc_auc_score(
            noisy_examples[eval_idx],
            torch.vmap(torch.trace)(results[method_name][metric_name]).numpy(
                force=True
            ),
        )
        datum[metric_name] = score
    data.append(datum)
df = pd.DataFrame.from_records(data)
df.to_csv(os.path.join(output_dir, "auc-mislabeled-detection.csv"), index=False)

# %%
for method_name in results.keys():
    if method_name in ["dense", "logra"]:
        continue
    ratios = results["dense"]["si"].squeeze() / results[method_name]["si"].squeeze()
    plt.boxplot(
        [
            ratios[mask].numpy(force=True)
            for mask in [noisy_examples[eval_idx], ~noisy_examples[eval_idx]]
        ],
        tick_labels=["noisy", "clean"],
        showfliers=False,
    )
    plt.title(method_name)
    plt.ylabel("$gFg/gF_{\\text{approx}}g$")
    plt.text(
        1,
        10,
        f"roc AUC: {
            round(
                roc_auc_score(
                    noisy_examples[eval_idx],
                    -ratios.numpy(force=True),
                ),
                2,
            )
        }",
    )
    plt.savefig(os.path.join(output_dir, method_name, "ratios") + ".png")
    plt.clf()

# %%


results["eval_idx"] = eval_idx
results["data"] = noisy_train_set.tensors[0]
results["noisy_targets"] = noisy_train_set.tensors[1]
results["targets"] = train_set.tensors[1]


torch.save(results, os.path.join(output_dir, "dump.pt"))

# %%
