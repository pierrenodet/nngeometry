import torch
import torch.nn.functional as F


def random_flip(p):
    def f(x):
        flip_mask = (torch.rand(x.shape[0], device=x.device) < p).view(-1, 1, 1, 1)
        return torch.where(flip_mask, x.flip(-1), x)

    return f


def random_crop(pad):
    def f(x):
        B, C, H, W = x.shape
        x = F.pad(x, (pad,) * 4, "constant", 0)

        i = torch.randint(0, 2 * pad + 1, (B,), device=x.device).view(B, 1, 1)
        j = torch.randint(0, 2 * pad + 1, (B,), device=x.device).view(B, 1, 1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing="ij",
        )

        src_y = (grid_y + i).unsqueeze(1)
        src_x = (grid_x + j).unsqueeze(1)

        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
        channel_idx = torch.arange(C, device=x.device).view(1, C, 1, 1)

        return x[batch_idx, channel_idx, src_y, src_x]

    return f


def random_erase(p, s):
    def f(x):
        B, _, H, W = x.shape

        erase_mask = (torch.rand(B, device=x.device) < p).view(B, 1, 1)

        i = torch.randint(0, H - s + 1, (B,), device=x.device).view(B, 1, 1)
        j = torch.randint(0, W - s + 1, (B,), device=x.device).view(B, 1, 1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing="ij",
        )
        in_box = (grid_y >= i) & (grid_y < i + s) & (grid_x >= j) & (grid_x < j + s)

        erase_mask = in_box & erase_mask

        return torch.where(erase_mask.unsqueeze(1), 0.0, x)

    return f
