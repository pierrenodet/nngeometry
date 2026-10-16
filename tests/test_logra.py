import math

import pytest
import torch
from tasks import (
    get_conv_task,
    get_fullyconnect_onlylast_task,
    get_fullyconnect_task,
    get_linear_conv_task,
    get_linear_fc_task,
)

from nngeometry.backend.torch_func_jacobian import TorchFuncJacobianBackend
from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.logra import PMatLoGRA
from nngeometry.object.map import PFMapDense, PFMapFactored, PFMapImplicit, random_pfmap
from nngeometry.object.pspace import PMatDense, PMatDiag, PMatEKFAC, PMatImplicit
from nngeometry.object.vector import random_fvector, random_pvector

linear_tasks = [
    get_linear_fc_task,
    get_linear_conv_task,
    get_fullyconnect_onlylast_task,
]
nonlinear_tasks = [get_fullyconnect_task, get_conv_task]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_logra_lift_adjoint_of_project():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        model.train()

        backend = TorchHooksJacobianBackend(
            model=model, function=lambda inputs, _: model(inputs)
        )
        F_ld = PMatLoGRA(
            generator=backend,
            examples=loader,
            layer_collection=lc,
            representation=PMatDense,
        )

        J = PFMapFactored(lc, backend, examples=loader)
        U = random_pfmap(F_ld.projected_layer_collection, (J.size(0), J.size(1)))

        torch.testing.assert_close(
            (F_ld.project(J).to_torch() * U.to_torch()).sum(),
            (J.to_torch() * F_ld.lift(U).to_torch()).sum(),
        )


def test_logra_ops():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        model.train()

        backend = TorchHooksJacobianBackend(
            model=model, function=lambda inputs, _: model(inputs)
        )
        F_ld = PMatLoGRA(
            generator=backend,
            examples=loader,
            layer_collection=lc,
            representation=PMatDense,
        )
        F_dense = F_ld.to_torch()

        J = PFMapFactored(lc, backend, examples=loader)
        torch.testing.assert_close(
            F_ld.mmap(J).to_torch(),
            torch.einsum("pP, onP->onp", F_dense, J.to_torch()),
        )
        torch.testing.assert_close(F_ld.get_diag(), torch.diag(F_dense))
        torch.testing.assert_close(F_ld.trace(), torch.trace(F_dense))
        torch.testing.assert_close(F_ld.norm(), torch.linalg.norm(F_dense))


def test_logra_ekfac():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        model.train()

        backend = TorchHooksJacobianBackend(
            model=model, function=lambda inputs, _: model(inputs)
        )
        F_ekfac = PMatEKFAC(
            generator=backend,
            examples=loader,
            layer_collection=lc,
        )
        F_ekfac.update_diag(loader)
        F_ld = PMatLoGRA(
            generator=backend,
            examples=loader,
            layer_collection=lc,
            representation=PMatDiag,
            projections="kfac",
            rank=1.0,
        )
        torch.testing.assert_close(F_ld.to_torch(), F_ekfac.to_torch())
        J = PFMapFactored(lc, backend, examples=loader)
        torch.testing.assert_close(F_ld.mmap(J).to_torch(), F_ekfac.mmap(J).to_torch())
