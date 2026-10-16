import math
from functools import partial

import torch

from nngeometry.backend._backend import AbstractBackend
from nngeometry.backend.dummy import DummyGenerator
from nngeometry.layercollection import LayerCollection, LinearLayer
from nngeometry.object.map import PFMapDense, PFMapFactored
from nngeometry.object.pspace import PMatAbstract, PMatDiag


class ProjectedJacobianBackend(AbstractBackend):
    def __init__(self, source_generator, source_layer_collection, project):
        self.source_generator = source_generator
        self.source_layer_collection = source_layer_collection
        self.project = project
        self.verbose = self.source_generator.verbose
        self.source_generator.verbose = False

    def get_covariance_matrix(self, examples, layer_collection):
        layerid_to_mod = self.source_layer_collection.get_layerid_module_map(
            self.source_generator.model
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = layer_collection.numel()

        G = torch.zeros((n_parameters, n_parameters), device=device, dtype=dtype)

        for batch in self._get_iter_loader(loader):
            factored = PFMapFactored(
                layer_collection=self.source_layer_collection,
                generator=self.source_generator,
                examples=batch,
            )
            projected = self.project(factored).to_torch().view(-1, n_parameters)
            G.addmm_(projected.t(), projected)

        G /= n_examples
        return G

    def get_covariance_diag(self, examples, layer_collection):
        layerid_to_mod = self.source_layer_collection.get_layerid_module_map(
            self.source_generator.model
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = layer_collection.numel()

        diag = torch.zeros(n_parameters, device=device, dtype=dtype)

        for batch in self._get_iter_loader(loader):
            factored = PFMapFactored(
                layer_collection=self.source_layer_collection,
                generator=self.source_generator,
                examples=batch,
            )
            projected = self.project(factored)
            diag += (projected.to_torch().view(-1, n_parameters) ** 2).sum(dim=0)

        diag /= n_examples

        return diag

    def get_covariance_layer_blocks(self, examples, layer_collection):
        layerid_to_mod = self.source_layer_collection.get_layerid_module_map(
            self.source_generator.model
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        blocks = {
            layer_id: torch.zeros(
                (layer.numel(), layer.numel()), device=device, dtype=dtype
            )
            for layer_id, layer in layer_collection.layers.items()
        }

        for batch in self._get_iter_loader(loader):
            factored = PFMapFactored(
                layer_collection=self.source_layer_collection,
                generator=self.source_generator,
                examples=batch,
            )
            projected = self.project(factored)
            for layer_id, layer in projected.layer_collection.layers.items():
                (p,) = projected.to_torch_layer(layer_id)
                p = p.reshape(-1, layer.numel())
                blocks[layer_id].addmm_(p.t(), p)

        for layer_id in blocks:
            blocks[layer_id] /= n_examples
        return blocks


def rademacher(i, o, generator=None, device=None, dtype=None):
    proj = torch.randint(0, 2, (i, o), generator=generator, device=device, dtype=dtype)
    proj *= 2
    proj -= 1
    return proj


def gaussian(i, o, generator=None, device=None, dtype=None):
    proj = torch.randn((i, o), generator=generator, device=device, dtype=dtype)
    return proj


PROJECTIONS = {"rademacher": rademacher, "gaussian": gaussian}


class PMatLoGRA(PMatAbstract):
    def __init__(
        self,
        layer_collection,
        generator,
        verbose=False,
        data=None,
        examples=None,
        representation=PMatDiag,
        rank=1.0,  # todo perlayer rank
        projections="rademacher",  # ease the use of projections from kfac
        seed=None,
        **kwargs,
    ):
        self.layer_collection = layer_collection
        self.generator = generator
        self.verbose = verbose
        if data is not None:
            self.data = data
        else:
            self.data = {}

            if projections in PROJECTIONS:
                layerid_to_mod = layer_collection.get_layerid_module_map(
                    generator.model
                )
                device = generator._check_same_device(layerid_to_mod.values())
                dtype = generator._check_same_dtype(layerid_to_mod.values())

                rng = None
                if seed is not None:
                    rng = torch.Generator(device=device).manual_seed(seed)
                sample = partial(
                    PROJECTIONS[projections],
                    generator=rng,
                    device=device,
                    dtype=dtype,
                )

                for layer_id, layer in layer_collection.layers.items():
                    layer_class = layer.__class__.__name__
                    if layer_class == "LinearLayer":
                        sG = layer.out_features
                        sA = layer.in_features
                    elif layer_class == "Conv2dLayer":
                        sG = layer.out_channels
                        sA = (
                            layer.in_channels
                            * layer.kernel_size[0]
                            * layer.kernel_size[1]
                        )
                    elif layer_class == "Conv1dLayer":
                        sG = layer.out_channels
                        sA = layer.in_channels * layer.kernel_size[0]
                    elif layer_class == "EmbeddingLayer":
                        sG = layer.embedding_dim
                        sA = layer.num_embeddings
                    if layer.has_bias():
                        sA += 1
                    rA = max(1, int(sA * rank)) if isinstance(rank, float) else rank
                    proj_a, _ = torch.linalg.qr(sample(sA, rA))
                    proj_a *= math.sqrt(sA / rA)
                    rG = max(1, int(sG * rank)) if isinstance(rank, float) else rank
                    proj_g, _ = torch.linalg.qr(sample(sG, rG))
                    proj_g *= math.sqrt(sG / rG)
                    self.data[layer_id] = (proj_a, proj_g)

            elif projections in ["kfac", "one_iter_kpsvd"]:
                self.data = self.generator.get_kfac_blocks(
                    examples, self.layer_collection, strategy=projections
                )
                for layer_id, layer in layer_collection.layers.items():
                    A, G = self.data[layer_id]
                    sA, sG = A.shape[0], G.shape[0]
                    _, evecs_A = torch.linalg.eigh(A)
                    _, evecs_G = torch.linalg.eigh(G)
                    rA = max(1, int(sA * rank)) if isinstance(rank, float) else rank
                    rG = max(1, int(sG * rank)) if isinstance(rank, float) else rank
                    self.data[layer_id] = (evecs_A[:, -rA:], evecs_G[:, -rG:])

        self.projected_layer_collection = LayerCollection()
        for layer_id, layer in layer_collection.layers.items():
            proj_a, proj_g = self.data[layer_id]
            rA, rG = proj_a.shape[-1], proj_g.shape[-1]
            self.projected_layer_collection.add_layer(
                layer_id, LinearLayer(rA, rG, bias=False)
            )

        projected_backend = ProjectedJacobianBackend(
            source_generator=self.generator,
            source_layer_collection=self.layer_collection,
            project=self.project,
        )
        self.F = representation(
            layer_collection=self.projected_layer_collection,
            generator=projected_backend,
            examples=examples,
            **kwargs,
        )

    def project(self, pfmap):
        if not isinstance(pfmap, PFMapFactored):
            raise NotImplementedError()

        P = []

        for layer_id, layer in self.layer_collection.layers.items():
            if layer_id not in pfmap.layer_collection.layers:
                continue
            a, g = pfmap.data[layer_id]
            proj_a, proj_g = self.data[layer_id]
            p = torch.einsum("nsr,onsR->onrR", a @ proj_a, g @ proj_g)
            P.append(p.reshape(p.shape[0], p.shape[1], -1))

        P = torch.cat(P, dim=-1)

        return PFMapDense(self.projected_layer_collection, DummyGenerator(None), data=P)

    def lift(self, pfmap: PFMapDense):
        factors = {}
        for layer_id, layer in self.layer_collection.layers.items():
            proj_a, proj_g = self.data[layer_id]
            (p,) = pfmap.to_torch_layer(layer_id)
            a = proj_a.t().unsqueeze(0).expand(p.shape[1], -1, -1)
            g = p.reshape(p.shape[0], p.shape[1], -1, proj_g.shape[-1]) @ proj_g.t()
            factors[layer_id] = (a, g)

        return PFMapFactored(self.layer_collection, DummyGenerator(None), data=factors)

    def mmap(self, pfmap):
        return self.lift(self.F.mmap(self.project(pfmap)))

    def mapTMmap(self, pfmap, reduction=None):
        return self.F.mapTMmap(self.project(pfmap), reduction=reduction)

    def solvePFMap(self, pfmap, regul, solve, **kwargs):
        return self.lift(self.F.solvePFMap(self.project(pfmap), regul, solve, **kwargs))

    def get_device(self):
        return self.F.get_device()

    def mv(self, v):
        pass

    def vTMv(self, v):
        pass

    def solvePVec(self, x, regul, solve, **kwargs):
        pass

    def _projection_gram(self):
        blocks = []

        for layer_id in self.layer_collection.layers:
            proj_a, proj_g = self.data[layer_id]
            blocks.append(torch.kron(proj_a.T @ proj_a, proj_g.T @ proj_g))

        return torch.block_diag(*blocks)

    def trace(self):
        # tr(BABt) = tr(ABBt)
        F = self.F.to_torch()
        K = self._projection_gram()
        return (F * K.T).sum()

    def norm(self, ord=None):
        if ord not in (None, "fro"):
            raise NotImplementedError()

        F = self.F.to_torch()
        K = self._projection_gram()
        FK = F @ K
        return torch.sqrt((FK * FK.t()).sum())

    def _projected_basis(self):
        return PFMapDense(
            layer_collection=self.F.layer_collection,
            data=torch.eye(self.F.size(0), device=self.get_device())[None, ...],
        )

    def to_torch(self):
        F = self.F.to_torch()
        basis = self._projected_basis()
        P = self.lift(basis).to_torch()[0]
        return P.transpose(-1, -2) @ F @ P

    def get_diag(self):
        basis = self._projected_basis()
        P = self.lift(basis).to_torch()[0]
        FP = self.lift(self.F.mmap(basis)).to_torch()[0]
        return (P * FP).sum(dim=0)
