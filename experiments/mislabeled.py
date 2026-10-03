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
from nngeometry.object.fspace import FMatDense
from nngeometry.object.map import PFMapFactored, random_pfmap
from nngeometry.object.pspace import (
    PMatBlockDiag,
    PMatDense,
    PMatEKFAC,
    PMatKFAC,
    PMatLowRank,
)
from nngeometry.object.vector import FVector


def is_interactive():
    return "ipykernel" in sys.modules


if is_interactive():
    sys.argv = [""]

parser = argparse.ArgumentParser()
add_default_arguments(parser, "mislabeled")
parser.add_argument("--noise_ratio", type=float, default=0.25)

args = parser.parse_args()

args.dataset = "mnist-1d"
args.runs = 1
args.seed = 1337
args.rank = 10
args.strategy = "kfac"
args.whiten = "eigh"


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

epochs = (
    [0]
    + [1]
    + [10]
    + list(range(100, 1000, 100))
    + list(range(1000, args.epochs + 1, 2000))
)
# epochs = [0, 1000]
# %%
for i, (c, d) in enumerate(
    chain(product([256], [2, 4, 8, 16]), product([32, 64, 128, 256, 512, 1024], [4]))
):
    # model_fn = partial(model_fn, c=c, d=d, h=1)

    model_clean = deepcopy(model_fn()).to(device=args.device)
    optimizer_clean = optimizer_fn(model_clean.parameters())

    model = deepcopy(model_fn()).to(device=args.device)
    optimizer = optimizer_fn(model.parameters())

    for e in range(1, len(epochs)):
        fit(
            model_clean,
            optimizer_clean,
            train,
            epochs[e] - epochs[e - 1],
            args.device,
            augmentations=augmentations,
        )
        gold_train_accuracy, gold_train_loss = eval(model_clean, train, args.device)
        gold_test_accuracy, gold_test_loss = eval(model_clean, test, args.device)
        print(gold_train_accuracy, gold_train_loss)
        print(gold_test_accuracy, gold_test_loss)

        fit(
            model,
            optimizer,
            noisy_train,
            epochs[e] - epochs[e - 1],
            args.device,
            augmentations=augmentations,
        )
        none_ntrain_accuracy, none_ntrain_loss = eval(model, noisy_train, args.device)
        none_train_accuracy, none_train_loss = eval(model, train, args.device)
        none_test_accuracy, none_test_loss = eval(model, test, args.device)
        print(none_ntrain_accuracy, none_ntrain_loss)
        print(none_train_accuracy, none_train_loss)
        print(none_test_accuracy, none_test_loss)

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

        if "F" in args.repr:
            if "ekfac" in args.repr:
                repr = partial(PMatEKFAC, strategy=args.strategy)
                lc = supported_kfac_layers(
                    LayerCollection.from_model(model, ignore_unsupported_layers=True)
                )
            elif "kfac" in args.repr:
                repr = partial(PMatKFAC, strategy=args.strategy)
                lc = supported_kfac_layers(
                    LayerCollection.from_model(model, ignore_unsupported_layers=True)
                )
            elif "bd" in args.repr:
                repr = PMatBlockDiag
                lc = supported_kfac_layers(
                    LayerCollection.from_model(model, ignore_unsupported_layers=True)
                )
                # lc = LayerCollection.from_model(model)
            elif "lr" in args.repr:
                repr = PMatLowRank
                lc = supported_kfac_layers(
                    LayerCollection.from_model(model, ignore_unsupported_layers=True)
                )
                # lc = LayerCollection.from_model(model)

            F = FIM(
                model,
                DataLoader(noisy_train_set, batch_size=128, shuffle=False),
                repr,
                "classif_logits",
                device=args.device,
                layer_collection=lc,
                verbose=True,
            )
        elif "K" in args.repr:
            generator = TorchHooksJacobianBackend(
                model, logits, centering=False, verbose=True
            )
            if "last" in args.repr:
                lc = last_layer(LayerCollection.from_model(model))
            else:
                lc = LayerCollection.from_model(model)
            F = FMatDense(
                lc,
                generator,
                examples=DataLoader(noisy_train_set, batch_size=128, shuffle=False),
            )

        if hasattr(F, "update_diag"):
            F.update_diag(noisy_train)

        N = noisy_examples.shape[0]
        if not isinstance(F, FMatDense):
            if hasattr(F, "__rmul__"):
                F = N * F
        if isinstance(F, PMatKFAC) and args.strategy == "kfac":  # kfac
            for layer_id, layer in lc.layers.items():
                A, G = F.data[layer_id]
                A *= (1 / (len(classes) - 1)) ** 0.5
                G *= (1 * (len(classes) - 1)) ** 0.5

        # def rnystroem(A, k, regul=1e-8):
        #     lc = A.layer_collection

        #     # isn't there a better way ?
        #     layerid_to_mod = lc.get_layerid_module_map(A.generator.model)
        #     device = A.generator._check_same_device(layerid_to_mod.values())
        #     dtype = A.generator._check_same_dtype(layerid_to_mod.values())

        #     S = random_pfmap(lc, (k, 1), device, dtype)
        #     S, _ = qr(S)
        #     Y = A @ S
        #     Y = Y
        #     C = S @ Y.adjoint()
        #     B = C.solve(Y, regul=regul)
        #     evecs, evals, _ = torch.linalg.svd(
        #         B.to_torch().view(-1, B.size(-1)).t(), full_matrices=False
        #     )
        #     evals = torch.clamp(evals**2 - regul, min=0)
        #     return evals, evecs

        if args.rank is not None and isinstance(F, PMatLowRank):
            _, evals, evecs = torch.svd_lowrank(
                F.data.view(-1, F.size(1)), q=20 + args.rank, niter=20
            )
            # evals = evals**2
            # evals, evecs = rnystroem(F, k=args.rank)
            # evals = evals**2
            # F.compute_eigendecomposition(impl="gram_eigh")
            # evals, evecs = F.get_eigendecomposition()

        # %%

        if args.rank is not None and isinstance(F, PMatLowRank):
            lr_evals, lr_evecs = evals[-args.rank :], evecs[:, -args.rank :]
            F_old = F
            F = PMatLowRank(
                F.layer_collection,
                F.generator,
                data=(lr_evecs * (lr_evals[None, :] ** 0.5)).t(),
            )
            F.evals = lr_evals
            F.evecs = lr_evecs
        # %%
        avg_lam = (F.trace() / F.layer_collection.numel()).item()
        light_noisy_noaug = DataLoader(noisy_train_set, batch_size=32, shuffle=False)
        self_influences = []
        self_influences_loo = []
        leverages = []
        leverages_loo = []
        det_schurs = []
        cooks = []
        if isinstance(F, PMatLowRank):
            if args.regul_type == "trace":
                regul = args.regul * avg_lam
            else:
                regul = args.regul

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
                FinvG = F.solve(
                    pfmap_grad, regul=regul, solve="eigendecomposition"
                ).to_torch()
                self_influence = torch.einsum(
                    "onp, Onp->noO",
                    pfmap_grad.to_torch(),
                    FinvG,
                )
                downdate = torch.einsum(
                    "onp, Onp->noO",
                    pfmap_func.to_torch(),
                    F.solve(
                        pfmap_func,
                        regul=regul,
                        solve="eigendecomposition",
                        rcond=0,
                    ).to_torch(),
                )
                cross = torch.einsum(
                    "onp, Onp->noO",
                    pfmap_func.to_torch(),
                    FinvG,
                )
                self_influences.append(self_influence)
                schur = torch.eye(downdate.shape[-1], device=args.device) - downdate
                loo_effect = torch.linalg.solve(schur, cross)
                self_influence_loo = cross.transpose(1, 2) @ loo_effect
                self_influences_loo.append(self_influence + self_influence_loo)
                FinvJ = F.solve(
                    pfmap_func, regul=regul, solve="eigendecomposition"
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
                cooks.append(
                    self_influence
                    + self_influence_loo
                    + loo_effect.transpose(1, 2) @ loo_effect
                )
        elif isinstance(F, PMatEKFAC):
            evecs, evals = F.data

            for inputs, targets in tqdm(light_noisy_noaug):
                self_influence = []
                self_influence_loo = []
                leverage = []
                leverage_loo = []
                cook = []
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
                for layer_id, layer in lc.layers.items():
                    if args.regul_type == "trace":
                        regul = args.regul * evals[layer_id].mean()
                    else:
                        regul = args.regul
                    Ga, Gg = pfmap_grad.data[layer_id]
                    Ja, Jg = pfmap_func.data[layer_id]

                    Ga_kfe = Ga @ evecs[layer_id][0]
                    Gg_kfe = Gg @ evecs[layer_id][1]

                    Ja_kfe = Ga_kfe
                    Jg_kfe = Jg @ evecs[layer_id][1]

                    D = evals[layer_id].view(Jg_kfe.size(-1), Ja_kfe.size(-1))
                    D_loo = D - torch.einsum(
                        "onsg,onSg,nsa,nSa->nga", Jg_kfe, Jg_kfe, Ja_kfe, Ja_kfe
                    )

                    self_influence.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa,ga->noO",
                            Gg_kfe,
                            Gg_kfe,
                            Ga_kfe,
                            Ga_kfe,
                            1 / (D + regul),
                        )
                    )
                    self_influence_loo.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa,nga->noO",
                            Gg_kfe,
                            Gg_kfe,
                            Ga_kfe,
                            Ga_kfe,
                            1 / (D_loo + regul),
                        )
                    )
                    leverage.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa,ga->noO",
                            Jg_kfe,
                            Jg_kfe,
                            Ja_kfe,
                            Ja_kfe,
                            1 / (D + regul),
                        )
                    )
                    leverage_loo.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa,nga->noO",
                            Jg_kfe,
                            Jg_kfe,
                            Ja_kfe,
                            Ja_kfe,
                            1 / (D_loo + regul),
                        )
                    )
                    cook.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa,nga->noO",
                            Gg_kfe,
                            Gg_kfe,
                            Ga_kfe,
                            Ga_kfe,
                            (D + regul) / (D_loo + regul) ** 2,
                        )
                    )
                self_influences.append(
                    torch.stack(self_influence, dim=0).detach().cpu()
                )
                self_influences_loo.append(
                    torch.stack(self_influence_loo, dim=0).detach().cpu()
                )
                leverages.append(torch.stack(leverage, dim=0).detach().cpu())
                leverages_loo.append(torch.stack(leverage_loo, dim=0).detach().cpu())
                cooks.append(torch.stack(cook, dim=0).detach().cpu())

        elif isinstance(F, PMatKFAC) and args.strategy == "kfac":
            if args.whiten == "eigh":

                def whitener(A, damp):
                    evals, evecs = torch.linalg.eigh(A)
                    inv_sqrt = (evals.clip(min=0) + damp).rsqrt()
                    return evecs * inv_sqrt

                def whiten(L, x):
                    return x @ L

            elif args.whiten == "cholesky":

                def whitener(A, damp):
                    return torch.linalg.cholesky(
                        A + damp * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
                    )

                def whiten(L, x):
                    return torch.linalg.solve_triangular(
                        L, x.transpose(-1, -2), upper=False
                    ).transpose(-1, -2)

            cache = {}

            for layer_id, layer in tqdm(lc.layers.items()):
                A, G = F.data[layer_id]

                if args.regul_type == "trace":
                    A_reg = torch.trace(A) / A.size(0)
                    G_reg = torch.trace(G) / G.size(0)
                elif args.regul_type == "pi":
                    pi = torch.trace(A) / torch.trace(G) * A.size(0) / G.size(0)
                    A_reg = pi
                    G_reg = 1 / pi
                else:
                    A_reg = 1
                    G_reg = 1

                cache[layer_id] = {
                    "LA": whitener(A, args.regul**0.5 * A_reg**0.5),
                    "LG": whitener(G, args.regul**0.5 * G_reg**0.5),
                }

            def woodbury_solve(U, b):
                if U.size(-1) <= U.size(-2):
                    schur = -(U.transpose(-1, -2) @ U)
                    schur.diagonal(dim1=-2, dim2=-1).add_(1)
                    return torch.linalg.solve(schur, b.transpose(-1, -2)).transpose(
                        -1, -2
                    )
                else:
                    schur = -(U @ U.transpose(-1, -2))
                    schur.diagonal(dim1=-2, dim2=-1).add_(1)
                    cross = b @ U.transpose(-1, -2)
                    return b + cross @ torch.linalg.solve(schur, U)

            for inputs, targets in tqdm(light_noisy_noaug):
                self_influence = []
                self_influence_loo = []
                leverage = []
                leverage_loo = []
                cook = []
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
                for layer_id, layer in lc.layers.items():
                    Ga, Gg = pfmap_grad.data[layer_id]
                    Ja, Jg = pfmap_func.data[layer_id]

                    Ga_w = whiten(cache[layer_id]["LA"], Ga)
                    Gg_w = whiten(cache[layer_id]["LG"], Gg)

                    self_influence.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa->noO",
                            Gg_w,
                            Gg_w,
                            Ga_w,
                            Ga_w,
                        )
                    )

                    Ja_w = Ga_w
                    Jg_w = whiten(cache[layer_id]["LG"], Jg)

                    leverage.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa->noO",
                            Jg_w,
                            Jg_w,
                            Ja_w,
                            Ja_w,
                        )
                    )

                    Ga_w_loo = woodbury_solve(Ja_w * N**-0.25, Ga_w)
                    Ja_w_loo = Ga_w_loo
                    Gg_w_flat = Gg_w.permute(1, 0, 2, 3).reshape(
                        Gg_w.size(1), -1, Gg_w.size(3)
                    )
                    Jg_w_flat = Jg_w.permute(1, 0, 2, 3).reshape(
                        Jg_w.size(1), -1, Jg_w.size(3)
                    )
                    Gg_w_loo_flat, Jg_w_loo_flat = woodbury_solve(
                        Jg_w_flat * N**-0.25,
                        torch.cat((Gg_w_flat, Jg_w_flat), dim=1),
                    ).split((Gg_w_flat.size(1), Jg_w_flat.size(1)), dim=1)
                    Gg_w_loo = Gg_w_loo_flat.reshape(
                        Gg_w.size(1),
                        Gg_w.size(0),
                        Gg_w.size(2),
                        Gg_w.size(3),
                    ).permute(1, 0, 2, 3)
                    Jg_w_loo = Jg_w_loo_flat.reshape(
                        Jg_w.size(1),
                        Jg_w.size(0),
                        Jg_w.size(2),
                        Jg_w.size(3),
                    ).permute(1, 0, 2, 3)

                    self_influence_loo.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa->noO",
                            Gg_w_loo,
                            Gg_w,
                            Ga_w_loo,
                            Ga_w,
                        )
                    )
                    cook.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa->noO",
                            Gg_w_loo,
                            Gg_w_loo,
                            Ga_w_loo,
                            Ga_w_loo,
                        )
                    )
                    leverage_loo.append(
                        torch.einsum(
                            "onsg,OnSg,nsa,nSa->noO",
                            Jg_w_loo,
                            Jg_w,
                            Ja_w_loo,
                            Ja_w,
                        )
                    )

                self_influences.append(
                    torch.stack(self_influence, dim=0).detach().cpu()
                )
                self_influences_loo.append(
                    torch.stack(self_influence_loo, dim=0).detach().cpu()
                )
                leverages.append(torch.stack(leverage, dim=0).detach().cpu())
                leverages_loo.append(torch.stack(leverage_loo, dim=0).detach().cpu())
                cooks.append(torch.stack(cook, dim=0).detach().cpu())

        elif isinstance(F, PMatBlockDiag):
            cache = {}

            for layer_id, layer in tqdm(lc.layers.items()):
                block = F.data[layer_id]

                if args.regul_type == "trace":
                    regul = args.regul * torch.trace(block) / block.size(0)
                else:
                    regul = args.regul

                # cache[layer_id] = torch.linalg.cholesky(
                #     block
                #     + regul
                #     * torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
                # )

                cache[layer_id] = torch.linalg.ldl_factor(
                    block
                    + regul
                    * torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
                )

                # def solve(A, b):
                #     return torch.linalg.solve_triangular(
                #         A.mT,
                #         torch.linalg.solve_triangular(A, b, upper=False),
                #         upper=True,
                #     )

                def solve(A, b):
                    return torch.linalg.ldl_solve(*A, b)

            for inputs, targets in tqdm(light_noisy_noaug):
                self_influence = []
                self_influence_loo = []
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
                    block_FinvJ = solve(cache[layer_id], J.transpose(1, 2)).transpose(
                        1, 2
                    )
                    block_leverage = torch.einsum("nop, nOp->noO", J, block_FinvJ)
                    block_cross = torch.einsum("nop, nOp->noO", G, block_FinvJ)
                    block_schur = (
                        torch.eye(block_leverage.shape[-1], device=args.device)
                        - block_leverage
                    )
                    self_influence.append(block_si)
                    loo_effect = torch.linalg.solve(
                        block_schur, block_cross.transpose(1, 2)
                    )
                    self_influence_loo.append(block_si + block_cross @ loo_effect)
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
                self_influences.append(
                    torch.stack(self_influence, dim=0).detach().cpu()
                )
                self_influences_loo.append(
                    torch.stack(self_influence_loo, dim=0).detach().cpu()
                )
                det_schurs.append(torch.stack(det_schur, dim=0).detach().cpu())
                cooks.append(torch.stack(cook, dim=0).detach().cpu())

        elif isinstance(F, PMatDense):
            if args.regul_type == "trace":
                regul = args.regul * avg_lam
            else:
                regul = args.regul
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
                self_influence = torch.einsum(
                    "onp, Onp->noO",
                    pfmap_grad.to_torch(),
                    F.solve(pfmap_grad, regul=regul).to_torch(),
                )
                FinvJ = F.solve(pfmap_func, regul=regul).to_torch()
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
                self_influences.append(self_influence)
                schur = torch.eye(leverage.shape[-1], device=args.device) - leverage
                loo_effect = torch.linalg.solve(schur, cross.transpose(1, 2))
                self_influences_loo.append(self_influence + cross @ loo_effect)
                leverages.append(leverage)
                leverages_loo.append(
                    torch.linalg.solve(schur, leverage.transpose(1, 2))
                )
                det_schurs.append(torch.logdet(schur))
                cooks.append(
                    self_influence
                    + cross @ loo_effect
                    + loo_effect.transpose(1, 2) @ loo_effect
                )

        # %%
        if self_influences[0].ndim == 3:
            self_influences = torch.cat(self_influences, dim=0)[None, ...]
            self_influences_loo = torch.cat(self_influences_loo, dim=0)[None, ...]
            leverages = torch.cat(leverages, dim=0)[None, ...]
            leverages_loo = torch.cat(leverages_loo, dim=0)[None, ...]
            cooks = torch.cat(cooks, dim=0)[None, ...]
        else:
            self_influences = torch.cat(self_influences, dim=1)
            self_influences_loo = torch.cat(self_influences_loo, dim=1)
            leverages = torch.cat(leverages, dim=1)
            leverages_loo = torch.cat(leverages_loo, dim=1)
            cooks = torch.cat(cooks, dim=1)

        # %%
        margins = []
        confidences = []
        for inputs, targets in tqdm(light_noisy_noaug):
            with torch.no_grad():
                outputs = model(inputs)
                p = torch.softmax(outputs, dim=1)
                mask = torch.zeros_like(p, dtype=bool)
                mask[torch.arange(len(targets)), targets] = True
                margins.append(
                    p[mask] - p[~mask].reshape(len(targets), -1).max(axis=1)[0]
                )
                confidences.append(p[mask])
        margins = -torch.cat(margins)
        confidences = -torch.cat(confidences)

        for scores in [margins, confidences]:
            print(roc_auc_score(noisy_examples, scores.numpy(force=True)))

        for scores in [
            self_influences,
            self_influences_loo,
            cooks,
            leverages,
            leverages_loo,
        ]:
            print(
                roc_auc_score(
                    noisy_examples,
                    torch.diagonal(scores, dim1=-2, dim2=-1)
                    .sum(dim=-1)
                    .sum(dim=0)
                    .numpy(force=True),
                )
            )

        with h5py.File(
            os.path.join(args.output_dir, f"{result_file_name(args)}.h5"), "a"
        ) as f:
            run = f.require_group(f"runs/run_{i}")
            run.attrs.update(
                {
                    "h": 4,
                    "c": c,
                    "d": d,
                    "repr": args.repr,
                    "regul": args.regul,
                    "regul_type": args.regul_type,
                    "seed": args.seed,
                    "whiten": args.whiten if "F_kfac" == args.repr else "",
                }
            )
            epoch = run.require_group(f"epoch_{epochs[e]}")
            epoch.attrs.update(
                {
                    "gold_train_accuracy": gold_train_accuracy,
                    "gold_test_accuracy": gold_test_accuracy,
                    "gold_train_loss": gold_train_loss,
                    "gold_test_loss": gold_test_loss,
                    "none_ntrain_accuracy": none_ntrain_accuracy,
                    "none_ntrain_loss": none_ntrain_loss,
                    "none_train_accuracy": none_train_accuracy,
                    "none_test_accuracy": none_test_accuracy,
                    "none_train_loss": none_train_loss,
                    "none_test_loss": none_test_loss,
                    "avg_eval_F": avg_lam,
                }
            )

            metrics = epoch.require_group("metrics")
            metrics.create_dataset(
                "self_influence", data=self_influences.detach().cpu()
            )
            metrics.create_dataset(
                "self_influence_loo", data=self_influences_loo.detach().cpu()
            )
            metrics.create_dataset("cook", data=cooks.detach().cpu())
            metrics.create_dataset("leverage", data=leverages.detach().cpu())
            metrics.create_dataset("leverage_loo", data=leverages_loo.detach().cpu())
            metrics.create_dataset("margins", data=margins.detach().cpu())
            metrics.create_dataset("confidences", data=confidences.detach().cpu())
            if "noisy_examples" not in run:
                run.create_dataset("noisy_examples", data=noisy_examples.detach().cpu())
