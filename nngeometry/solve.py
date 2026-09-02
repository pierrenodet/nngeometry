import torch


def cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    tol = max(rtol * b.norm(), atol)
    lc = A.layer_collection
    if max_iter is None:
        max_iter = 10 * lc.numel()

    if x0 is None:
        r = b
        x = 0 * b
    else:
        r = b - A @ x0
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    for _ in range(max_iter):
        if r.norm() <= tol:
            return x
        Ap = A @ p
        if regul > 0:
            Ap = Ap + regul * p
        rz = r @ z
        α = rz / (p @ Ap)
        x = x + α * p
        r = r - α * Ap
        z = r if M is None else M.solve(r, regul=regul)
        β = (r @ z) / rz
        p = z + β * p
    return x


def Q(pfmap):
    from nngeometry.object.map import PFMapDense

    Q, _ = torch.linalg.qr(pfmap.to_torch().view(-1, pfmap.layer_collection.numel()).T)
    return PFMapDense(
        pfmap.layer_collection,
        generator=pfmap.generator,
        data=Q.T.view(*pfmap.size()),
    )


def block_cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    # https://arxiv.org/pdf/2502.16998 Algorithm 8 with γ->α and δ->β to match cg
    tol = max(rtol * (torch.sum(b.to_torch() ** 2, dim=-1) ** 0.5).mean(), atol)
    lc = A.layer_collection
    if max_iter is None:
        max_iter = 10 * lc.numel()

    if x0 is None:
        r = b
        x = 0 * b
    else:
        r = b - (A @ x0.adjoint()).adjoint()
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    p = Q(p)
    for i in range(max_iter):
        if torch.all(torch.sum(r.to_torch() ** 2, dim=-1) ** 0.5 <= tol):
            return x
        Ap = (A @ p.adjoint()).adjoint()
        if regul > 0:
            Ap = Ap + regul * p
        pTAp = p @ Ap.adjoint()
        α = pTAp.solve((p @ z.adjoint()), regul=0)
        x = x + p @ α
        r = r - Ap @ α
        z = r if M is None else M.solve(r, regul=regul)
        β = -1 * pTAp.solve((Ap @ z.adjoint()), regul=0)
        p = z + p @ β
        p = Q(p)
    return x
