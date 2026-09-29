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
from nngeometry.metrics import FIM, FIM_MonteCarlo
from nngeometry.object.map import PFMapFactored, random_pfmap
from nngeometry.object.pspace import (
    PMatBlockDiag,
    PMatDense,
    PMatEKFAC,
    PMatKFAC,
    PMatLowRank,
)
from nngeometry.object.vector import FVector

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


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
args.regul = 1e-5
args.epochs = 1000
args.repr = "F_kfac"
args.seed = 1337
args.rank = 1000
args.use_pi = True
args.strategy = "kfac"


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
noisy_probability = torch.rand(noisy_train_set.tensors[1].shape[0])
noisy_examples = noisy_probability <= args.noise_ratio
noisy_labels = torch.randint(0, len(classes), (torch.sum(noisy_examples),))
noisy_train_set.tensors[1][noisy_examples] = noisy_labels

noisy_train = DataLoader(noisy_train_set, batch_size=128, shuffle=True)

# %%
for i, (c, d) in enumerate(
    chain(product([32, 64, 128, 256, 512, 1024], [4]), product([512], [2, 4, 8, 16]))
):
    model_fn = partial(model_fn, c=c, d=d, h=4)

    model_clean = deepcopy(model_fn()).to(device=args.device)
    optimizer_clean = optimizer_fn(model_clean.parameters())

    model = deepcopy(model_fn()).to(device=args.device)
    optimizer = optimizer_fn(model.parameters())

    old_epochs = 0
    epochs = 10
    while epochs <= args.epochs:
        fit(
            model_clean,
            optimizer_clean,
            train,
            epochs - old_epochs,
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
            epochs - old_epochs,
            args.device,
            augmentations=augmentations,
        )
        none_train_accuracy, none_train_loss = eval(model, test, args.device)
        none_test_accuracy, none_test_loss = eval(model, test, args.device)
        print(none_train_accuracy, none_train_loss)
        print(none_test_accuracy, none_test_loss)

        # %%

        if "ekfac" in args.repr:
            repr = partial(PMatEKFAC, strategy=args.strategy)
            lc = supported_kfac_layers(LayerCollection.from_model(model))
        elif "kfac" in args.repr:
            repr = partial(PMatKFAC, strategy=args.strategy)
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
            DataLoader(noisy_train_set, batch_size=128, shuffle=False),
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

        if hasattr(F, "update_diag"):
            F.update_diag(noisy_train)

        N = noisy_examples.shape[0]
        if not isinstance(F, FMatDense):
            if hasattr(F, "__rmul__"):
                F = N * F
        if isinstance(F, PMatKFAC) and args.strategy == "kfac":  # kfac
            for layer_id, layer in tqdm(lc.layers.items()):
                A, G_w = F.data[layer_id]
                A /= (len(classes) - 1) ** 0.5
                G_w *= (len(classes) - 1) ** 0.5

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
            # _, evals, evecs = torch.svd_lowrank(
            #     F.data.view(-1, F.size(1)), q=20 + args.rank, niter=20
            # )
            # evals = evals**2
            # evals, evecs = rnystroem(F, k=args.rank)
            # evals = evals**2
            F.compute_eigendecomposition(impl="gram_eigh")
            evals, evecs = F.get_eigendecomposition()

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
        avg_lam = (F.trace() / F.layer_collection.numel()).item()
        light_noisy_noaug = DataLoader(noisy_train_set, batch_size=64, shuffle=False)

        self_influence = []
        self_influence_loo = []
        leverages = []
        leverages_loo = []
        det_schurs = []
        cooks = []
        cache = {}
        if isinstance(F, PMatLowRank):
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
                        pfmap_func,
                        regul=args.regul,
                        solve="eigendecomposition",
                        rcond=0,
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

                    J_w = pfmap_func.to_torch_layer(layer_id)
                    Jkfe = F._proj_to_kfe_batched(J_w, evecs[layer_id], layer)
                    G_w = pfmap_grad.to_torch_layer(layer_id)
                    Gkfe = F._proj_to_kfe_batched(G_w, evecs[layer_id], layer)

                    sG = Gkfe.size()

                    Gkfe_scaled = Gkfe / (
                        evals[layer_id].view(1, 1, *sG[2:]) + args.regul
                    )

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

        elif isinstance(F, PMatKFAC) and args.strategy == "kfac":
            cache = {}

            for layer_id, layer in tqdm(lc.layers.items()):
                A, G = F.data[layer_id]

                if args.use_pi:
                    pi = (
                        torch.trace(A) / torch.trace(G) * G.size(0) / A.size(0)
                    ) ** 0.5
                else:
                    pi = 1

                A = ((N**0.5) * A) + (
                    args.regul**0.5
                    * pi
                    * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                )

                G = ((N**0.5) * G) + (
                    args.regul**0.5
                    / pi
                    * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                )

                cache[layer_id] = {
                    "LA": torch.linalg.cholesky(A),
                    "LG": torch.linalg.cholesky(G),
                }

            def solve(A, b):
                return torch.cholesky_solve(b, A)

            def whiten(L, x):
                return torch.linalg.solve_triangular(
                    L, x.transpose(-1, -2), upper=False
                ).transpose(-1, -2)

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

                    si.append(
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

                    Ga_w_loo = woodbury_solve(Ja_w, Ga_w)
                    Ja_w_loo = Ga_w_loo
                    Gg_w_flat = Gg_w.permute(1, 0, 2, 3).reshape(
                        Gg_w.size(1), -1, Gg_w.size(3)
                    )
                    Jg_w_flat = Jg_w.permute(1, 0, 2, 3).reshape(
                        Jg_w.size(1), -1, Jg_w.size(3)
                    )
                    Gg_w_loo_flat, Jg_w_loo_flat = woodbury_solve(
                        Jg_w_flat,
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

                    si_loo.append(
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

                self_influence.append(torch.stack(si, dim=0).detach().cpu())
                self_influence_loo.append(torch.stack(si_loo, dim=0).detach().cpu())
                leverages.append(torch.stack(leverage, dim=0).detach().cpu())
                leverages_loo.append(torch.stack(leverage_loo, dim=0).detach().cpu())
                cooks.append(torch.stack(cook, dim=0).detach().cpu())

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
                    block_FinvJ = solve(cache[layer_id], J.transpose(1, 2)).transpose(
                        1, 2
                    )
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
                leverages_loo.append(
                    torch.linalg.solve(schur, leverage.transpose(1, 2))
                )
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
        else:
            self_influence = torch.cat(self_influence, dim=1)
            self_influence_loo = torch.cat(self_influence_loo, dim=1)
            leverages = torch.cat(leverages, dim=1)
            leverages_loo = torch.cat(leverages_loo, dim=1)
            cooks = torch.cat(cooks, dim=1)

        # %%
        losses = []
        margins = []
        confidences = []
        for inputs, targets in tqdm(light_noisy_noaug):
            with torch.no_grad():
                outputs = model(inputs)
                losses.append(tF.cross_entropy(outputs, targets, reduction="none"))
                p = torch.softmax(outputs, dim=1)
                mask = torch.zeros_like(p, dtype=bool)
                mask[torch.arange(len(targets)), targets] = True
                margins.append(
                    p[mask] - p[~mask].reshape(len(targets), -1).max(axis=1)[0]
                )
                confidences.append(p[mask])
        losses = torch.cat(losses)
        margins = -torch.cat(margins)
        confidences = torch.cat(confidences)

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
                    "pi": args.use_pi,
                }
            )
            epoch = run.require_group(f"epoch_{epochs}")
            epoch.attrs.update(
                {
                    "gold_train_accuracy": gold_train_accuracy,
                    "gold_test_accuracy": gold_test_accuracy,
                    "gold_train_loss": gold_train_loss,
                    "gold_test_loss": gold_test_loss,
                    "none_train_accuracy": none_train_accuracy,
                    "none_test_accuracy": none_test_accuracy,
                    "none_train_loss": none_train_loss,
                    "none_test_loss": none_test_loss,
                    "avg_eval_F": avg_lam,
                }
            )

            metrics = epoch.require_group("metrics")
            metrics.create_dataset("self-influence", data=self_influence.detach().cpu())
            metrics.create_dataset(
                "self-influence-loo", data=self_influence_loo.detach().cpu()
            )
            metrics.create_dataset("cook", data=cooks.detach().cpu())
            metrics.create_dataset("leverage", data=leverages.detach().cpu())
            metrics.create_dataset("leverage-loo", data=leverages_loo.detach().cpu())
            metrics.create_dataset("losses", data=losses.detach().cpu())
            metrics.create_dataset("margins", data=margins.detach().cpu())
            metrics.create_dataset("confidences", data=confidences.detach().cpu())
            if "noisy_examples" not in run:
                run.create_dataset("noisy_examples", data=noisy_examples.detach().cpu())

        old_epochs = epochs
        epochs *= 10
