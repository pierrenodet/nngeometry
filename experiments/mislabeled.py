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
from nngeometry.object.map import random_pfmap
from nngeometry.object.pspace import PMatBlockDiag, PMatDense, PMatEKFAC, PMatKFAC
from nngeometry.object.vector import FVector

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
args.epochs = 1000
args.repr = "F_kfac"
args.competitors = True
args.seed = 1337
args.ft = False
args.output_dir = "mislabeled-output"
args.device = "cuda:0"


os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)

transform, dataset, model_fn, optimizer_fn, classes = DATASETS[args.dataset]

# model_fn = partial(model_fn, c=16)
model_fn = partial(model_fn, c=32, h=4, d=8)

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
    print(eval(model, noisy_train, args.device))
    print(eval(model, train, args.device))
    print(eval(model, test, args.device))
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
    from nngeometry.object.pspace import PMatLowRank

    args.rank = 1000
    args.repr = "F_ekfac"
    if "kfac" in args.repr:
        repr = partial(PMatKFAC, strategy="one_iter_kpsvd")
        # repr = PMatKFAC
        # repr = PMatEKFAC
        # repr = partial(PMatEKFAC, strategy="one_iter_kpsvd")
        lc = supported_kfac_layers(LayerCollection.from_model(model))
    elif "bd" in args.repr:
        repr = PMatBlockDiag
        lc = supported_kfac_layers(LayerCollection.from_model(model))
        # lc = LayerCollection.from_model(model)
    elif "lr" in args.repr:
        repr = PMatLowRank
        lc = supported_kfac_layers(LayerCollection.from_model(model))
        # lc = LayerCollection.from_model(model)
    F = FIM(
        model,
        noisy_train,
        repr,
        "classif_logits",
        device=args.device,
        layer_collection=lc,
        verbose=True,
    )

    from nngeometry.object.fspace import FMatDense

    def loss(inputs, targets):
        return tF.cross_entropy(model(inputs), targets, reduction="none")

    from nngeometry.metrics import sqrt_var_classif_logits

    def func(inputs, targets):
        return sqrt_var_classif_logits(model(inputs))

    # def func(inputs, targets):
    #     logp = tF.log_softmax(model(inputs), dim=1)
    #     p = torch.exp(logp).detach()
    #     return torch.sqrt(p) * logp

    def logits(inputs, targets):
        return model(inputs)

    # generator = TorchHooksJacobianBackend(model, logits, centering=False, verbose=True)
    # F = FMatDense(
    #     lc,
    #     generator,
    #     examples=DataLoader(noisy_train_set, batch_size=128, shuffle=False),
    # )

    # if hasattr(F, "update_diag"):
    #     F.update_diag(noisy_train)

    N = noisy_examples.shape[0]
    if not isinstance(F, FMatDense):
        if hasattr(F, "__rmul__"):
            F = N * F
        else:  # kfac
            for layer_id, layer in tqdm(lc.layers.items()):
                a, g = F.data[layer_id]
                a *= N**0.5
                g *= N**0.5

    def qr(pfmap):
        from nngeometry.object.fspace import FMatDense
        from nngeometry.object.map import PFMapDense

        sJ = pfmap.size()

        Q, R = torch.linalg.qr(pfmap.to_torch().view(-1, sJ[-1]).t(), mode="reduced")
        Q = PFMapDense(pfmap.layer_collection, pfmap.generator, data=Q.t().view(*sJ))
        R = FMatDense(
            pfmap.layer_collection,
            pfmap.generator,
            data=R.view(sJ[0], sJ[1], sJ[0], sJ[1]),
        )
        return Q, R

    from nngeometry.object.map import random_pfmap

    def rnystroem(A, k, regul=1e-8):
        lc = A.layer_collection

        # isn't there a better way ?
        layerid_to_mod = lc.get_layerid_module_map(A.generator.model)
        device = A.generator._check_same_device(layerid_to_mod.values())
        dtype = A.generator._check_same_dtype(layerid_to_mod.values())

        S = random_pfmap(lc, (k, 1), device, dtype)
        S, _ = qr(S)
        Y = A @ S
        Y = Y
        C = S @ Y.adjoint()
        B = C.solve(Y, regul=regul)
        evecs, evals, _ = torch.linalg.svd(
            B.to_torch().view(-1, B.size(-1)).t(), full_matrices=False
        )
        evals = torch.clamp(evals**2 - regul, min=0)
        return evals, evecs

    if args.rank is not None and isinstance(F, PMatLowRank):
        # _, evals, evecs = torch.svd_lowrank(
        #     F.data.view(-1, F.size(1)), q=20 + args.rank, niter=20
        # )
        # evals = evals**2
        evals, evecs = rnystroem(F, k=args.rank)
        evals = evals**2
        # F.compute_eigendecomposition(impl="gram_eigh")
        # evals, evecs = F.get_eigendecomposition()

    # %%
    if args.rank is not None and isinstance(F, PMatLowRank):
        print(evecs.shape)
        lr_evals, lr_evecs = evals[-args.rank :], evecs[:, -args.rank :]
        print(evals)
        F_old = F
        F = PMatLowRank(
            F.layer_collection,
            F.generator,
            data=(lr_evecs * (lr_evals[None, :] ** 0.5)).t(),
        )
        F.evals = lr_evals
        F.evecs = lr_evecs
    # %%
    avg_lam = F.trace() / F.layer_collection.numel()
    print(avg_lam)
    args.regul = 1e-4
    light_noisy_noaug = DataLoader(noisy_train_set, batch_size=32, shuffle=False)

    self_influence = []
    self_influence_loo = []
    leverages = []
    leverages_loo = []
    det_schurs = []
    cooks = []
    cache = {}
    if isinstance(F, PMatLowRank):
        # Q, R = torch.linalg.qr(F.data.view(-1, F.size(0)).t(), mode="reduced")
        # C = R @ R.t()
        # U, Lc = (
        #     Q.t(),
        #     torch.linalg.cholesky(
        #         C + args.regul * torch.eye(C.shape[0], device=args.device)
        #     ),
        # )
        # F.compute_eigendecomposition(impl="svd")
        # evals, evecs = F.get_eigendecomposition()
        # evals, evecs = evals[:100], evecs[:, :100]
        # F = PMatLowRank(F.layer_collection, F.generator, data=evecs)
        # F.evals = evals
        # F.evecs = evecs

        for inputs, targets in tqdm(light_noisy_noaug):
            # pfmap_func = Jacobian(
            #     model,
            #     (inputs, targets),
            #     function=func,
            #     layer_collection=lc,
            # )
            # pfmap_grad = Jacobian(
            #     model,
            #     (inputs, targets),
            #     function=loss,
            #     layer_collection=lc,
            # )
            # G = pfmap_grad.to_torch().view(-1, pfmap_grad.size(-1))
            # FinvG = (
            #     torch.cholesky_solve(U @ G.t(), Lc)
            #     .t()
            #     .reshape(pfmap_grad.size(0), pfmap_grad.size(1), -1)
            # )
            # si = torch.einsum(
            #     "onp, Onp->noO",
            #     (U @ G.t()).t().reshape(pfmap_grad.size(0), pfmap_grad.size(1), -1),
            #     FinvG,
            # )
            # J = pfmap_func.to_torch().view(-1, pfmap_func.size(-1))
            # FinvJ = (
            #     torch.cholesky_solve(U @ J.t(), Lc)
            #     .t()
            #     .reshape(pfmap_func.size(0), pfmap_func.size(1), -1)
            # )
            # leverage = torch.einsum(
            #     "onp, Onp->noO",
            #     (U @ J.t()).t().reshape(pfmap_func.size(0), pfmap_func.size(1), -1),
            #     FinvJ,
            # )
            # schur = torch.eye(leverage.shape[-1], device=args.device) - leverage
            # # print(torch.linalg.eigvalsh(schur).amin(dim=-1))
            # cross = torch.einsum(
            #     "onp, Onp->noO",
            #     (U @ G.t()).t().reshape(pfmap_grad.size(0), pfmap_grad.size(1), -1),
            #     FinvJ,
            # )
            # loo_effect = torch.linalg.solve(schur, cross.transpose(1, 2))
            # self_influence.append(si)
            # self_influence_loo.append(si + cross @ loo_effect)
            # leverages.append(leverage)
            # leverages_loo.append(torch.linalg.solve(schur, leverage.transpose(1, 2)))
            # det_schurs.append(torch.logdet(schur))
            # cooks.append(
            #     si + cross @ loo_effect + loo_effect.transpose(1, 2) @ loo_effect
            # )
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
            FinvG = F.solve(
                pfmap_grad, regul=args.regul, solve="eigendecomposition"
            ).to_torch()
            si = torch.einsum(
                "onp, Onp->noO",
                pfmap_grad.to_torch(),
                FinvG,
            )
            downdate = torch.einsum(
                "onp, Onp->noO",
                pfmap_func.to_torch(),
                F.solve(
                    pfmap_func, regul=args.regul, solve="eigendecomposition", rcond=0
                ).to_torch(),
            )
            cross = torch.einsum(
                "onp, Onp->noO",
                pfmap_func.to_torch(),
                FinvG,
            )
            self_influence.append(si)
            schur = torch.eye(downdate.shape[-1], device=args.device) - downdate
            loo_effect = torch.linalg.solve(schur, cross)
            si_loo = cross.transpose(1, 2) @ loo_effect
            self_influence_loo.append(si + si_loo)
            FinvJ = F.solve(
                pfmap_func, regul=args.regul, solve="eigendecomposition"
            ).to_torch()
            leverage = torch.einsum(
                "onp, Onp->noO",
                pfmap_func.to_torch(),
                FinvJ,
            )
            leverages.append(leverage)
            leverages_loo.append(
                leverage
                + leverage @ torch.linalg.solve(schur, leverage.transpose(1, 2))
            )
            det_schurs.append(torch.logdet(schur))
            cooks.append(si + si_loo + loo_effect.transpose(1, 2) @ loo_effect)
    elif isinstance(F, FMatDense):
        alpha = F.solve(
            FVector(
                (
                    torch.nn.functional.one_hot(
                        noisy_train_set.tensors[1],
                        num_classes=len(classes),
                    ).to(dtype=torch.float32, device=args.device)
                )
                - torch.softmax(
                    model(noisy_train_set.tensors[0].to(device=args.device)), dim=1
                )
            ),
            regul=args.regul,
        )
        Q = F.inv(regul=args.regul).to_torch()
        self_influence_loo.append(
            (
                torch.linalg.solve(
                    torch.diagonal(Q, dim1=1, dim2=3).permute(2, 0, 1),
                    alpha.to_torch().t().unsqueeze(-1),
                ).squeeze(-1)
                ** 2
            ).sum(dim=-1)
        )

    elif isinstance(F, PMatEKFAC):
        evecs, evals = F.data

        for inputs, targets in tqdm(light_noisy_noaug):
            si = []
            si_loo = []
            leverage = []
            leverage_loo = []
            cook = []
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

                si.append(torch.einsum("onij, Onij->noO", Gkfe_scaled, Gkfe))

                si_loo.append(
                    torch.einsum(
                        "onij, Onij->noO",
                        Gkfe
                        / (
                            (evals[layer_id].view(1, 1, *sG[2:]) + args.regul)
                            - (Jkfe**2).sum(dim=0)
                        ),
                        Gkfe,
                    )
                )
                leverage.append(
                    torch.einsum(
                        "onij, Onij->noO",
                        Jkfe / (evals[layer_id].view(1, 1, *sG[2:]) + args.regul),
                        Jkfe,
                    )
                )
                leverage_loo.append(
                    torch.einsum(
                        "onij, Onij->noO",
                        Jkfe
                        / (
                            evals[layer_id].view(1, 1, *sG[2:])
                            + args.regul
                            - (Jkfe**2).sum(dim=0)
                        ),
                        Jkfe,
                    )
                )
                cook.append(
                    torch.einsum(
                        "onij, Onij->noO",
                        Gkfe
                        * evals[layer_id].view(1, 1, *sG[2:])
                        / (
                            evals[layer_id].view(1, 1, *sG[2:])
                            + args.regul
                            - (Jkfe**2).sum(dim=0)
                        )
                        ** 2,
                        Gkfe,
                    )
                )

            self_influence.append(torch.stack(si, dim=0).detach().cpu())
            self_influence_loo.append(torch.stack(si_loo, dim=0).detach().cpu())
            leverages.append(torch.stack(leverage, dim=0).detach().cpu())
            leverages_loo.append(torch.stack(leverage_loo, dim=0).detach().cpu())
            cooks.append(torch.stack(cook, dim=0).detach().cpu())

    elif isinstance(F, PMatKFAC):
        cache = {}

        def solve(A, B, scale=1.0):
            evals, evecs = A
            shape = B.shape
            B = B.movedim(-2, 0).reshape(shape[-2], -1)
            B = evecs.mT @ B
            B = B.reshape(shape[-2], *shape[:-2], shape[-1]).movedim(0, -2)
            scale = torch.as_tensor(scale, dtype=B.dtype, device=B.device)
            scale = scale.reshape(*scale.shape, *((1,) * (B.ndim - scale.ndim)))
            evals = evals.reshape(*((1,) * (B.ndim - 2)), -1, 1)
            B /= scale * evals + args.regul**0.5
            B = B.movedim(-2, 0).reshape(shape[-2], -1)
            B = evecs @ B
            return B.reshape(shape[-2], *shape[:-2], shape[-1]).movedim(0, -2)

        def woodbury_downdate_solve(solve, A, B, U):
            FinvG = solve(A, B)
            FinvJ = solve(A, U)
            leverage = U.mT @ FinvJ
            FinvG_matrix = FinvG.movedim(-2, 1).flatten(2)
            cross = U.mT @ FinvG_matrix
            schur = (
                torch.eye(
                    leverage.shape[-1],
                    dtype=leverage.dtype,
                    device=leverage.device,
                )
                - leverage
            )
            effect = torch.linalg.solve(schur, cross)
            correction = (
                (FinvJ @ effect)
                .reshape(B.shape[0], B.shape[-2], *B.shape[1:-2], B.shape[-1])
                .movedim(1, -2)
            )
            return FinvG + correction

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

        for inputs, targets in tqdm(light_noisy_noaug):
            si = []
            si_loo = []
            leverage = []
            leverage_loo = []
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

                si.append(torch.einsum("onij, Onij->noO", G, solve_ga))

                trace = cache[layer_id]["trace"]
                trace_loo = torch.sqrt(trace**2 - J.square().sum(dim=(0, 2, 3)))
                a_update_root = J.permute(1, 3, 0, 2).reshape(
                    J.shape[1], J.shape[3], -1
                ) / torch.sqrt(trace_loo[:, None, None])
                g_update_root = J.permute(1, 2, 0, 3).reshape(
                    J.shape[1], J.shape[2], -1
                ) / torch.sqrt(trace_loo[:, None, None])

                solve_g_loo = woodbury_downdate_solve(
                    partial(solve, scale=trace / trace_loo),
                    cache[layer_id]["eig_g"],
                    G.permute(1, 0, 2, 3),
                    g_update_root,
                )
                solve_ga_loo = woodbury_downdate_solve(
                    partial(solve, scale=trace / trace_loo),
                    cache[layer_id]["eig_a"],
                    solve_g_loo.transpose(-1, -2),
                    a_update_root,
                ).transpose(-1, -2)
                si_loo.append(torch.einsum("onij, nOij->noO", G, solve_ga_loo))

                solve_g_j = solve(cache[layer_id]["eig_g"], J)
                solve_ga_j = solve(
                    cache[layer_id]["eig_a"], solve_g_j.transpose(-1, -2)
                ).transpose(-1, -2)
                leverage.append(torch.einsum("onij, Onij->noO", J, solve_ga_j))

                solve_g_loo_j = woodbury_downdate_solve(
                    partial(solve, scale=trace / trace_loo),
                    cache[layer_id]["eig_g"],
                    J.permute(1, 0, 2, 3),
                    g_update_root,
                )
                solve_ga_loo_j = woodbury_downdate_solve(
                    partial(solve, scale=trace / trace_loo),
                    cache[layer_id]["eig_a"],
                    solve_g_loo_j.transpose(-1, -2),
                    a_update_root,
                ).transpose(-1, -2)
                leverage_loo.append(torch.einsum("onij, nOij->noO", J, solve_ga_loo_j))

            self_influence.append(torch.stack(si, dim=0).detach().cpu())
            self_influence_loo.append(torch.stack(si_loo, dim=0).detach().cpu())
            leverages.append(torch.stack(leverage, dim=0).detach().cpu())
            leverages_loo.append(torch.stack(leverage_loo, dim=0).detach().cpu())

    elif isinstance(F, PMatBlockDiag):
        cache = {}

        for layer_id, layer in tqdm(lc.layers.items()):
            block = F.data[layer_id]

            block_reg = block + args.regul * torch.eye(
                block.shape[0], dtype=block.dtype, device=block.device
            )

            cache[layer_id] = torch.linalg.cholesky(block_reg)

            def solve(A, B):
                return torch.cholesky_solve(B, A)

        for inputs, targets in tqdm(light_noisy_noaug):
            si = []
            si_loo = []
            pfmap_func = Jacobian(
                model,
                (inputs, targets),
                function=func,
                layer_collection=lc,
            )
            leverage = []
            leverage_loo = []
            det_schur = []
            cook = []
            neff = []
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
                si.append(block_si)
                loo_effect = torch.linalg.solve(
                    block_schur, block_cross.transpose(1, 2)
                )
                si_loo.append(block_si + block_cross @ loo_effect)
                leverage.append(block_leverage)
                leverage_loo.append(
                    torch.linalg.solve(block_schur, block_leverage.transpose(1, 2))
                )
                det_schur.append(torch.logdet(block_schur))
                cook.append(
                    block_si
                    + block_cross @ loo_effect
                    + loo_effect.transpose(1, 2) @ loo_effect
                )

            leverages.append(torch.stack(leverage, dim=0).detach().cpu())
            leverages_loo.append(torch.stack(leverage_loo, dim=0).detach().cpu())
            self_influence.append(torch.stack(si, dim=0).detach().cpu())
            self_influence_loo.append(torch.stack(si_loo, dim=0).detach().cpu())
            det_schurs.append(torch.stack(det_schur, dim=0).detach().cpu())
            cooks.append(torch.stack(cook, dim=0).detach().cpu())

    elif isinstance(F, PMatDense):
        for inputs, targets in tqdm(light_noisy_noaug):
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
            self_influence.append(si)
            schur = torch.eye(leverage.shape[-1], device=args.device) - leverage
            loo_effect = torch.linalg.solve(schur, cross.transpose(1, 2))
            self_influence_loo.append(si + cross @ loo_effect)
            leverages.append(leverage)
            leverages_loo.append(torch.linalg.solve(schur, leverage.transpose(1, 2)))
            det_schurs.append(torch.logdet(schur))
            cooks.append(
                si + cross @ loo_effect + loo_effect.transpose(1, 2) @ loo_effect
            )

    # %%
    if self_influence[0].ndim == 3:
        self_influence = torch.cat(self_influence, dim=0)[None, ...]
        self_influence_loo = torch.cat(self_influence_loo, dim=0)[None, ...]
        leverages = torch.cat(leverages, dim=0)[None, ...]
        leverages_loo = torch.cat(leverages_loo, dim=0)[None, ...]
        cooks = torch.cat(cooks, dim=0)[None, ...]
        det_schurs = torch.cat(det_schurs, dim=0)[None, ...]
    else:
        self_influence = torch.cat(self_influence, dim=1)
        self_influence_loo = torch.cat(self_influence_loo, dim=1)
        leverages = torch.cat(leverages, dim=1)
        leverages_loo = torch.cat(leverages_loo, dim=1)
        cooks = torch.cat(cooks, dim=1)
        det_schurs = torch.cat(det_schurs, dim=1)

    # %%
    losses = []
    for inputs, targets in tqdm(light_noisy_noaug):
        with torch.no_grad():
            losses.append(tF.cross_entropy(model(inputs), targets, reduction="none"))
    losses = torch.cat(losses)
    margins = []
    for inputs, targets in tqdm(light_noisy_noaug):
        with torch.no_grad():
            p = torch.softmax(model(inputs), dim=1)
            y = targets
            mask = torch.zeros_like(p, dtype=bool)
            mask[torch.arange(len(y)), y] = True
            margins.append(p[mask] - p[~mask].reshape(len(y), -1).max(axis=1)[0])
    margins = -torch.cat(margins)
    # %%

    tr_self_influence = torch.diagonal(self_influence, dim1=-2, dim2=-1).sum(dim=-1)
    tr_self_influence_loo = torch.diagonal(self_influence_loo, dim1=-2, dim2=-1).sum(
        dim=-1
    )
    # tr_cooks = torch.diagonal(cooks, dim1=-2, dim2=-1).sum(dim=-1)
    tr_leverages = torch.diagonal(leverages, dim1=-2, dim2=-1).sum(dim=-1)
    tr_leverages_loo = torch.diagonal(leverages_loo, dim1=-2, dim2=-1).sum(dim=-1)

    n_blocks = self_influence.shape[0]

    for name, scores in [
        ("self-influence", tr_self_influence),
        ("LOO self-influence (or cook bar)", tr_self_influence_loo),
        # ("cook", tr_cooks),
        ("leverage", tr_leverages),
        ("LOO leverage", tr_leverages_loo),
        # ("det schur", -det_schurs),
    ]:
        plt.plot(
            [
                roc_auc_score(noisy_examples, scores[i].numpy(force=True))
                for i in range(n_blocks)
            ],
            label=name,
        )
    plt.axhline(
        roc_auc_score(noisy_examples, losses.numpy(force=True)), label="per-sample loss"
    )
    plt.ylim((0.5, 1))
    plt.legend()
    plt.show()
    # %%

    print(
        roc_auc_score(noisy_examples, losses.numpy(force=True)),
        roc_auc_score(noisy_examples, margins.numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence.sum(dim=0).numpy(force=True)),
        roc_auc_score(
            noisy_examples, tr_self_influence_loo.sum(dim=0).numpy(force=True)
        ),
        # roc_auc_score(noisy_examples, tr_cooks.sum(dim=0).numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages.sum(dim=0).numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages_loo.sum(dim=0).numpy(force=True)),
        # roc_auc_score(noisy_examples, -det_schurs.sum(dim=0).numpy(force=True)),
    )
    print(
        roc_auc_score(noisy_examples, losses.numpy(force=True)),
        roc_auc_score(noisy_examples, margins.numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence_loo[-1].numpy(force=True)),
        # roc_auc_score(noisy_examples, tr_cooks[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages_loo[-1].numpy(force=True)),
        # roc_auc_score(noisy_examples, -det_schurs[-1].numpy(force=True)),
    )
    # )
    # %%
    plt.plot(
        [
            kendalltau(
                tr_self_influence[i].numpy(force=True),
                tr_self_influence_loo[i].numpy(force=True),
            ).statistic
            for i in range(n_blocks)
        ],
        label=name,
    )
    plt.show()
    print(
        kendalltau(
            tr_self_influence.sum(dim=0).numpy(force=True),
            tr_self_influence_loo.sum(dim=0).numpy(force=True),
        ).statistic
    )
    # %%
    plt.plot(
        [
            kendalltau(
                tr_leverages[i].numpy(force=True),
                tr_leverages_loo[i].numpy(force=True),
            ).statistic
            for i in range(n_blocks)
        ],
        label=name,
    )
    print(
        kendalltau(
            tr_leverages.sum(dim=0).numpy(force=True),
            tr_leverages_loo.sum(dim=0).numpy(force=True),
        ).statistic
    )
    plt.show()


# %%
# %%
import matplotlib.pyplot as plt

plt.scatter(
    torch.argsort(torch.argsort(tr_self_influence.sum(dim=0))).numpy(force=True),
    torch.argsort(torch.argsort(tr_self_influence_loo.sum(dim=0))).numpy(force=True),
    s=1,
    # c=noisy_train_set.tensors[1]
    c=noisy_examples,
)
plt.show()

plt.scatter(
    torch.argsort(torch.argsort(tr_leverages.sum(dim=0))).numpy(force=True),
    torch.argsort(torch.argsort(tr_leverages_loo.sum(dim=0))).numpy(force=True),
    s=1,
    # c=noisy_train_set.tensors[1]
    c=noisy_examples,
)
plt.show()
# %%
plt.scatter(
    tr_self_influence.sum(dim=0).numpy(force=True),
    tr_self_influence_loo.sum(dim=0).numpy(force=True),
)
plt.axline((0, 0), slope=1)
plt.show()

plt.scatter(
    tr_leverages.sum(dim=0).numpy(force=True),
    tr_leverages_loo.sum(dim=0).numpy(force=True),
)
plt.axline((0, 0), slope=1)
plt.show()

# %%
