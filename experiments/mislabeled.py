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
model_fn = partial(model_fn, c=32, h=4, d=4)

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
    if hasattr(F, "__rmul__"):
        F = N * F
    else:  # kfac
        for layer_id, layer in tqdm(lc.layers.items()):
            a, g = F.data[layer_id]
            a *= N**0.5
            g *= N**0.5
    # %%
    avg_lam = F.trace() / F.layer_collection.numel()
    print(avg_lam)
    args.regul = 1e-4
    light_noisy_noaug = DataLoader(noisy_train_set, batch_size=15)

    self_influence = []
    self_influence_loo = []
    leverages = []
    leverages_loo = []
    det_schurs = []
    cooks = []
    if isinstance(F, PMatEKFAC):
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

                solve_g = solve(cache[layer_id]["Lg"], G)
                solve_ga = solve(
                    cache[layer_id]["La"], solve_g.transpose(-1, -2)
                ).transpose(-1, -2)

                si.append(torch.einsum("onij, Onij->noO", G, solve_ga))

                def woodbury_downdate_solve(F, G, J):
                    FinvG = solve(F, G)
                    FinvJ = solve(F, J)
                    leverage = torch.einsum("onij, Onij -> noO", J, FinvJ)
                    schur = (
                        torch.eye(leverage.shape[-1], dtype=J.dtype, device=J.device)
                        - leverage
                    )
                    check = torch.vmap(torch.linalg.det)(schur)
                    print(check.min())
                    cross = torch.einsum("onij, Onij->noO", J, FinvG)
                    return FinvG + torch.einsum(
                        "onij, noO-> Onij", FinvJ, torch.linalg.solve(schur, cross)
                    )

                tr_loo = torch.trace(
                    torch.einsum("onij, onIj->iI", J, J) / (J.shape[1] ** 0.5)
                )
                print(tr_loo)
                solve_g_loo = woodbury_downdate_solve(
                    cache[layer_id]["Lg"],
                    G,
                    J / tr_loo**0.5,
                )
                solve_ga_loo = woodbury_downdate_solve(
                    cache[layer_id]["La"],
                    solve_g_loo.transpose(-1, -2),
                    J.transpose(-1, -2) / tr_loo**0.5,
                ).transpose(-1, -2)
                si_loo.append(torch.einsum("onij, Onij->noO", G, solve_ga_loo))
                # U_g = J.permute(1, 2, 0, 3).reshape(
                #     batch_size,
                #     out_dim,
                #     n_func_outputs * in_dim,
                # )
                # # U_g = U_g / torch.sqrt(torch.trace(F.data[layer_id][1]))

                # # The loss-output axis is not part of the downdate.
                # # It consists of independent RHS columns.
                # #
                # # [loss_out, batch, out, in]
                # #     -> [batch, out, loss_out * in]
                # G_rhs = G.permute(1, 2, 0, 3).reshape(
                #     batch_size,
                #     out_dim,
                #     n_loss_outputs * in_dim,
                # )

                # # Applies (G_reg - U_g U_g^T)^(-1) to every loss-output / input RHS.
                # Ginv_G_rhs = woodbury_downdate_solve(
                #     G_rhs,
                #     cache[layer_id]["Lg"],
                #     U_g,
                # )

                # # Back to [loss_out, batch, out, in].
                # Ginv_G = Ginv_G_rhs.reshape(
                #     batch_size,
                #     out_dim,
                #     n_loss_outputs,
                #     in_dim,
                # ).permute(2, 0, 1, 3)

                # # ------------------------------------------------------------------
                # # A factor downdate
                # #
                # # delta_a[n] = sum_q J[q, n].T @ J[q, n]
                # #
                # # U_a[n, in, (q, out)] = J[q, n, out, in] / sqrt(trace(a)).
                # # ------------------------------------------------------------------
                # U_a = J.permute(1, 3, 0, 2).reshape(
                #     batch_size,
                #     in_dim,
                #     n_func_outputs * out_dim,
                # )
                # # U_a = U_a/ torch.sqrt(torch.trace(F.data[layer_id][0]))

                # # To right-solve by A^{-1}, transpose the matrix representation:
                # #
                # # G^{-1} gradient:
                # # [loss_out, batch, out, in]
                # #
                # # RHS for A:
                # # [batch, in, loss_out * out]
                # A_rhs = Ginv_G.permute(1, 3, 0, 2).reshape(
                #     batch_size,
                #     in_dim,
                #     n_loss_outputs * out_dim,
                # )

                # Ainv_A_rhs = woodbury_downdate_solve(
                #     A_rhs,
                #     cache[layer_id]["La"],
                #     U_a,
                # )

                # # Convert back to [loss_out, batch, out, in].
                # FinvG_loo = Ainv_A_rhs.reshape(
                #     batch_size,
                #     in_dim,
                #     n_loss_outputs,
                #     out_dim,
                # ).permute(2, 0, 3, 1)
                # si_loo.append(
                #     torch.einsum(
                #         "onp,Onp->noO",
                #         G.view(n_loss_outputs, batch_size, -1),
                #         FinvG_loo.view(n_loss_outputs, batch_size, -1),
                #     )
                # )
                # def woodbury_downdate_solve(G, chol, U):
                #     FinvG = torch.cholesky_solve(G, chol)
                #     FinvU = torch.cholesky_solve(U, chol)
                #     schur = torch.eye(U.shape[-1], device=args.device) - (
                #         U.transpose(-1, -2) @ FinvU
                #     )
                #     cross = U.transpose(-1, -2) @ FinvG
                #     return FinvG + FinvU @ torch.linalg.solve(schur, cross)

                # solve_g = torch.cholesky_solve(G, cache[layer_id]["Lg"])
                # solve_ga = torch.cholesky_solve(
                #     solve_g.transpose(-1, -2),
                #     cache[layer_id]["La"],
                # ).transpose(-1, -2)

                # si += torch.einsum(
                #     "onp, Onp->noO",
                #     G.view(sG[0], sG[1], -1),
                #     solve_ga.view(sG[0], sG[1], -1),
                # )

                # U_g = J.permute(1, 2, 0, 3).reshape(
                #     batch_size,
                #     out_dim,
                #     n_func_outputs * in_dim,
                # ) / torch.sqrt(torch.trace(g))

                # # The loss-output axis is not part of the downdate.
                # # It consists of independent RHS columns.
                # #
                # # [loss_out, batch, out, in]
                # #     -> [batch, out, loss_out * in]
                # G_rhs = G.permute(1, 2, 0, 3).reshape(
                #     batch_size,
                #     out_dim,
                #     1 * in_dim,
                # )

                # # Applies (G_reg - U_g U_g^T)^(-1) to every loss-output / input RHS.
                # Ginv_G_rhs = woodbury_downdate_solve(
                #     G_rhs,
                #     cache[layer_id]["Lg"],
                #     U_g,
                # )

                # # Back to [loss_out, batch, out, in].
                # Ginv_G = Ginv_G_rhs.reshape(
                #     batch_size,
                #     out_dim,
                #     1,
                #     in_dim,
                # ).permute(2, 0, 1, 3)

                # # ------------------------------------------------------------------
                # # A factor downdate
                # #
                # # delta_a[n] = sum_q J[q, n].T @ J[q, n]
                # #
                # # U_a[n, in, (q, out)] = J[q, n, out, in] / sqrt(trace(a)).
                # # ------------------------------------------------------------------
                # U_a = J.permute(1, 3, 0, 2).reshape(
                #     batch_size,
                #     in_dim,
                #     n_func_outputs * out_dim,
                # ) / torch.sqrt(torch.trace(a))

                # # To right-solve by A^{-1}, transpose the matrix representation:
                # #
                # # G^{-1} gradient:
                # # [loss_out, batch, out, in]
                # #
                # # RHS for A:
                # # [batch, in, loss_out * out]
                # A_rhs = Ginv_G.permute(1, 3, 0, 2).reshape(
                #     batch_size,
                #     in_dim,
                #     1 * out_dim,
                # )

                # Ainv_A_rhs = woodbury_downdate_solve(
                #     A_rhs,
                #     cache[layer_id]["La"],
                #     U_a,
                # )

                # # Convert back to [loss_out, batch, out, in].
                # FinvG_loo = Ainv_A_rhs.reshape(
                #     batch_size,
                #     in_dim,
                #     1,
                #     out_dim,
                # ).permute(2, 0, 3, 1)
                # si_loo += torch.einsum(
                #     "onp,Onp->noO",
                #     G.view(1, batch_size, -1),
                #     FinvG_loo.view(1, batch_size, -1),
                # )

            self_influence.append(torch.stack(si, dim=0).detach().cpu())
            self_influence_loo.append(torch.stack(si_loo, dim=0).detach().cpu())
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

            # cache[layer_id] = torch.linalg.qr(block_reg)

            # def solve(A, B):
            #     Q, R = A
            #     return torch.linalg.solve_triangular(
            #         R, Q.transpose(-2, -1) @ B, upper=True
            #     )

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
                si + block_cross @ loo_effect + loo_effect.transpose(1, 2) @ loo_effect
            )
    # %%
    self_influence = torch.cat(self_influence, dim=1)
    self_influence_loo = torch.cat(self_influence_loo, dim=1)
    # leverages = torch.cat(leverages, dim=1)
    # leverages_loo = torch.cat(leverages_loo, dim=1)
    # cooks = torch.cat(cooks, dim=1)
    # det_schurs = torch.cat(det_schurs, dim=1)

    # %%
    losses = []
    for inputs, targets in tqdm(light_noisy_noaug):
        with torch.no_grad():
            losses.append(tF.cross_entropy(model(inputs), targets, reduction="none"))
    losses = torch.cat(losses)
    # %%

    tr_self_influence = torch.vmap(torch.vmap(torch.trace))(self_influence)
    tr_self_influence_loo = torch.vmap(torch.vmap(torch.trace))(self_influence_loo)
    tr_leverages = torch.vmap(torch.vmap(torch.trace))(leverages)
    tr_leverages_loo = torch.vmap(torch.vmap(torch.trace))(leverages_loo)
    tr_cooks = torch.vmap(torch.vmap(torch.trace))(cooks)

    n_blocks = len(lc.layers)

    for name, scores in [
        ("self-influence", tr_self_influence),
        ("LOO self-influence (or cook bar)", tr_self_influence_loo),
        ("leverage", tr_leverages),
        ("LOO leverage", tr_leverages_loo),
        # ("det schur", -det_schurs),
        ("cook", tr_cooks),
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

    print(
        roc_auc_score(noisy_examples, losses.numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence.sum(dim=0).numpy(force=True)),
        roc_auc_score(
            noisy_examples, tr_self_influence_loo.sum(dim=0).numpy(force=True)
        ),
        roc_auc_score(noisy_examples, tr_leverages.sum(dim=0).numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages_loo.sum(dim=0).numpy(force=True)),
        # roc_auc_score(noisy_examples, -det_schurs.sum(dim=0).numpy(force=True)),
        roc_auc_score(noisy_examples, tr_cooks.sum(dim=0).numpy(force=True)),
        roc_auc_score(
            noisy_examples,
            (tr_self_influence_loo + tr_cooks).sum(dim=0).numpy(force=True),
        ),
    )
    print(
        roc_auc_score(noisy_examples, losses.numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_self_influence_loo[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_leverages_loo[-1].numpy(force=True)),
        # roc_auc_score(noisy_examples, -det_schurs[-1].numpy(force=True)),
        roc_auc_score(noisy_examples, tr_cooks[-1].numpy(force=True)),
        roc_auc_score(
            noisy_examples, (tr_self_influence_loo + tr_cooks)[-1].numpy(force=True)
        ),
    )
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
    torch.argsort(torch.argsort(tr_self_influence_loo.sum(dim=0))),
    torch.argsort(torch.argsort(tr_self_influence.sum(dim=0))),
    s=1,
    # c=noisy_train_set.tensors[1]
    c=noisy_examples,
)
plt.show()

# %%
plt.scatter(tr_self_influence.sum(dim=0), tr_self_influence_loo.sum(dim=0))
plt.axline((0, 0), slope=1)
plt.show()

# for i in range(self_influence.shape[0]):
#     plt.scatter(tr_self_influence[i], tr_self_influence_loo[i])
#     plt.axline((0, 0), slope=1)
#     plt.show()

# %%
