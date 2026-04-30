import torch
import torch.nn.functional as F 
import math

COVERAGE_LIM = 0.75

def stochastic_round(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    x0 = torch.floor(x)
    p  = (x - x0).clamp(0.0, 1.0)
    r  = torch.rand(p.shape, device=p.device, dtype=p.dtype, generator=gen)
    add = (r < p).to(x0.dtype)
    return (x0 + add).long()

def _normalize_intensity(intensities, device):
    inten = torch.as_tensor(intensities, device=device, dtype=torch.float32)
    # Eğer senin pipeline 0-10 ise otomatik normalize et
    if float(inten.max().item()) > 1.5:
        inten = inten / 10.0
    return inten.clamp(0.0, 1.0)

def confusion_blocks(images, intensities, block_size=8, max_coverage_limit=COVERAGE_LIM, gen=None):
    B, C, H, W = images.shape
    N = len(intensities)

    inten = torch.tensor(intensities, device=images.device, dtype=torch.float32).clamp(0.0, 1.0)
    cov = (inten * float(max_coverage_limit)).clamp(0.0, 1.0)

    blocks_x = (W + block_size - 1) // block_size
    blocks_y = (H + block_size - 1) // block_size
    total_blocks = blocks_x * blocks_y

    k = stochastic_round(cov * total_blocks, gen=gen).clamp(0, total_blocks)  # [N]

    # nested scores per-image only
    scores = torch.rand((B, total_blocks), generator=gen, device=images.device)

    # sort once
    order = torch.argsort(scores, dim=1)  # [B, total_blocks]

    mask_blocks = torch.zeros((B, N, total_blocks), device=images.device, dtype=torch.bool)
    for n in range(N):
        kn = int(k[n].item())
        if kn > 0:
            idx = order[:, :kn]  # nested subset
            mask_blocks[:, n, :].scatter_(1, idx, True)

    mask_blocks = mask_blocks.view(B, N, blocks_y, blocks_x)
    mask_pix = mask_blocks.unsqueeze(2).expand(B, N, C, blocks_y, blocks_x)
    mask_pix = mask_pix.repeat_interleave(block_size, dim=3).repeat_interleave(block_size, dim=4)
    mask_pix = mask_pix[..., :H, :W]

    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()

    # optional nested replacement values
    noise = torch.rand((B, 1, C, H, W), generator=gen, device=images.device, dtype=images.dtype)
    noise = noise.expand(B, N, C, H, W)

    return torch.where(mask_pix, noise, imgs)

def random_stripes(images, intensities, max_coverage_limit=COVERAGE_LIM, vertical=True, gen=None):
    B, C, H, W = images.shape
    N = len(intensities)
    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()

    cov = torch.tensor(intensities, device=images.device, dtype=images.dtype).clamp(0.0, 1.0)
    cov = (cov * float(max_coverage_limit)).clamp(0,1)
    cov = cov.view(1, N, 1)

    if vertical:
        u = torch.rand((B, 1, W), generator=gen, device=images.device, dtype=images.dtype)
        mask_cols = (u < cov)  # [B,N,W] nested
        mask = mask_cols.view(B, N, 1, 1, W).expand(B, N, C, H, W)
    else:
        u = torch.rand((B, 1, H), generator=gen, device=images.device, dtype=images.dtype)
        mask_rows = (u < cov)  # [B,N,H] nested
        mask = mask_rows.view(B, N, 1, H, 1).expand(B, N, C, H, W)

    imgs[mask] = 0.0
    return imgs

def random_lines( images, intensities, max_coverage_limit=COVERAGE_LIM,
    line_length_range=(2, 6), gen=None, ref=224,):
    """
    Per-image + nested + coverage based.
    Ek mask ve pix allocate etmez.
    """
    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype
    N = len(intensities)

    inten = _normalize_intensity(intensities, device)
    coverage = (inten * float(max_coverage_limit)).clamp(0.0, 1.0)  # [N]

    scale = min(H, W) / float(ref)
    lmin = max(1, int(round(line_length_range[0] * scale)))
    lmax = max(lmin + 1, int(round(line_length_range[1] * scale)))
    avg_len = 0.5 * (lmin + lmax)

    target_points = coverage * float(H * W)          # [N]
    exp_lines = target_points / float(avg_len)       # [N]
    line_count = stochastic_round(exp_lines, gen=gen).clamp_min(0)  # [N]

    kmax = int(line_count.max().item())
    out = images.unsqueeze(1).expand(B, N, C, H, W).clone()
    if kmax == 0:
        return out

    center_x = torch.randint(0, W, (B, kmax), generator=gen, device=device)
    center_y = torch.randint(0, H, (B, kmax), generator=gen, device=device)
    lengths  = torch.randint(lmin, lmax + 1, (B, kmax), generator=gen, device=device)
    angles   = (torch.randint(0, 2, (B, kmax), generator=gen, device=device) * 90).float()
    colors   = torch.rand((B, kmax, C), generator=gen, device=device, dtype=dtype)

    max_len = int(lmax)
    offs = torch.arange(-max_len // 2, max_len // 2 + 1, device=device)  # [L]
    L = offs.numel()

    cos_a = torch.cos(torch.deg2rad(angles)).unsqueeze(-1)  # [B,kmax,1]
    sin_a = torch.sin(torch.deg2rad(angles)).unsqueeze(-1)

    x_d = (offs.view(1, 1, L) * cos_a).round().long()
    y_d = (offs.view(1, 1, L) * sin_a).round().long()

    half = (lengths // 2).clamp_min(1)
    use = offs.abs().view(1, 1, L) <= half.unsqueeze(-1)  # [B,kmax,L]

    x_pos = (center_x.unsqueeze(-1) + x_d).clamp(0, W - 1)
    y_pos = (center_y.unsqueeze(-1) + y_d).clamp(0, H - 1)

    b_idx_full = torch.arange(B, device=device).view(B, 1, 1).expand(B, kmax, L)

    for n in range(N):
        kn = int(line_count[n].item())
        if kn <= 0:
            continue

        use_n = use[:, :kn, :]
        if not use_n.any():
            continue

        b_flat = b_idx_full[:, :kn, :][use_n]
        x_flat = x_pos[:, :kn, :][use_n]
        y_flat = y_pos[:, :kn, :][use_n]

        cols = colors[:, :kn, :].unsqueeze(2).expand(B, kn, L, C)
        c_flat = cols[use_n]  # [P,C]

        out[b_flat, n, :, y_flat, x_flat] = c_flat

    return out.clamp(0, 1)

def random_crosses( images, intensities,max_coverage_limit=COVERAGE_LIM,
    gen=None):
    """
    Per-image + nested + coverage based.
    Ek mask ve pix allocate etmez.
    """
    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype
    N = len(intensities)

    inten = _normalize_intensity(intensities, device)
    coverage = (inten * float(max_coverage_limit)).clamp(0.0, 1.0)  # [N]

    target_points = coverage * float(H * W)
    exp_crosses = target_points / 5.0
    cross_count = stochastic_round(exp_crosses, gen=gen).clamp_min(0)  # [N]

    kmax = int(cross_count.max().item())
    out = images.unsqueeze(1).expand(B, N, C, H, W).clone()
    if kmax == 0:
        return out

    center_y = torch.randint(1, H - 1, (B, kmax), generator=gen, device=device)
    center_x = torch.randint(1, W - 1, (B, kmax), generator=gen, device=device)
    kind = torch.randint(0, 2, (B, kmax), generator=gen, device=device)
    colors = torch.rand((B, kmax, C), generator=gen, device=device, dtype=dtype)

    plus  = torch.tensor([(0,0), (-1,0), (1,0), (0,-1), (0,1)], device=device)
    cross = torch.tensor([(0,0), (-1,-1), (1,1), (-1,1), (1,-1)], device=device)
    pos = torch.stack([plus, cross], dim=0)  # [2,5,2]

    b_all = torch.arange(B, device=device).view(B, 1).expand(B, kmax)

    for n in range(N):
        kn = int(cross_count[n].item())
        if kn <= 0:
            continue

        cy = center_y[:, :kn]
        cx = center_x[:, :kn]
        kd = kind[:, :kn]
        col = colors[:, :kn, :]

        for shape_id in (0, 1):
            sel = (kd == shape_id)
            if not sel.any():
                continue

            b_sel = b_all[:, :kn][sel]
            cy_sel = cy[sel]
            cx_sel = cx[sel]
            col_sel = col[sel]  # [M,C]

            dy = pos[shape_id][:, 0].view(1, 5)
            dx = pos[shape_id][:, 1].view(1, 5)

            y = (cy_sel.view(-1, 1) + dy).clamp(0, H - 1).long()
            x = (cx_sel.view(-1, 1) + dx).clamp(0, W - 1).long()

            b_rep = b_sel.repeat_interleave(5)
            y_rep = y.flatten()
            x_rep = x.flatten()
            c_rep = col_sel.repeat_interleave(5, dim=0)

            out[b_rep, n, :, y_rep, x_rep] = c_rep

    return out.clamp(0, 1)


def structured_square_wave_noise(images, intensities, max_amplitude=COVERAGE_LIM,
                                        direction='horizontal', period=8, ref=224):
    B,C,H,W = images.shape
    N = len(intensities)

    scale = min(H, W) / float(ref)
    period = max(2, int(round(period * scale)))

    imgs = images.unsqueeze(1).expand(B,N,C,H,W).clone()
    inten = torch.tensor(intensities, device=images.device, dtype=images.dtype)
    amplitude_variety = inten * max_amplitude

    if direction == 'horizontal':
        bands = ((torch.arange(H, device=images.device)
                         // period) % 2).float() * 2 - 1  # [-1,1]
        pattern = bands.view(1,1,H,1).expand(N,C,H,W)
    else:
        bands = ((torch.arange(W, device=images.device)
                         // period) % 2).float() * 2 - 1
        pattern = bands.view(1,1,1,W).expand(N,C,H,W)

    pat = pattern * amplitude_variety.view(N,1,1,1)  # [N,C,H,W]
    pat = pat.unsqueeze(0).expand(B,N,C,H,W)  # [B,N,C,H,W]
    return (imgs + pat).clamp(0,1)

def colored_impulse_noise(images, intensities, gen=None, max_coverage_limit=COVERAGE_LIM):

    B,C,H,W = images.shape 
    N = len(intensities)
    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()
    inten = torch.tensor(intensities, device=imgs.device, dtype=imgs.dtype).clamp(0.0, 1.0)
    limit_inten = inten * float(max_coverage_limit)
    pix_coverage    = limit_inten.view(1,N,1,1,1)

    u = torch.rand((B,1,1,H,W), generator=gen, device=imgs.device, dtype=imgs.dtype)
    mask = (u < pix_coverage)

    noise = torch.rand((B, 1, C, H, W), device=imgs.device, dtype=imgs.dtype, generator=gen)
    noise = noise.expand(B, N, C, H, W)

    out = torch.where(mask.expand_as(imgs), noise, imgs)
    return out.clamp(0,1)

def gaussian_blur(images, sigmas, gen=None, ref=224):
    B,C,H,W = images.shape
    scale = min(H, W) / float(ref)
    outputs = []
    for sigma in sigmas:
        if sigma == 0:
            outputs.append(images)
            continue
        sigma = sigma * 10 * scale # scale sigma from the same range as other corruptions.
        # kernel size 6σ for even and large size
        k = max(3, int(2 * round(3 * sigma) + 1))
        coords = torch.arange(k, device=images.device) - k//2
        xx, yy = torch.meshgrid(coords, coords, indexing='xy')
        kern = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kern = kern / kern.sum()
        kernel = kern.view(1,1,k,k).repeat(C,1,1,1)
        blurred = F.conv2d(images, kernel, padding=k//2, groups=C)
        outputs.append(blurred)
    stacked = torch.stack(outputs, dim=1)  # [B, N, C, H, W]
    return stacked.clamp(0,1)

def gaussian_noise(images, sigmas, gen):
    B,C,H,W = images.shape; N = len(sigmas)
    imgs  = images.unsqueeze(1).expand(B,N,C,H,W)
    sigma = torch.tensor(sigmas, device=images.device, dtype=images.dtype).view(1,N,1,1,1)
    noise = torch.randn(imgs.shape, device=images.device, dtype=images.dtype, generator=gen) * sigma    
    return (imgs + noise).clamp(0,1)
def uniform_noise(images, ds, gen):
    B,C,H,W = images.shape; N = len(ds)
    imgs  = images.unsqueeze(1).expand(B,N,C,H,W)
    delta = torch.tensor(ds, device=images.device, dtype=images.dtype).view(1,N,1,1,1)
    noise = torch.rand(imgs.shape, device=images.device, dtype=images.dtype, generator=gen).mul_(2).sub_(1).mul_(delta)    
    return (imgs + noise).clamp(0,1)

def salt_and_pepper_noise(images, ps, gen=None, max_coverage_limit=COVERAGE_LIM):
    B, C, H, W = images.shape
    N = len(ps)

    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()

    inten = torch.as_tensor(ps, device=images.device, dtype=images.dtype).clamp(0.0, 1.0)
    p = (inten * float(max_coverage_limit)).clamp(0.0, 1.0).view(1, N, 1, 1, 1)

    r = torch.rand((B, 1, 1, H, W), device=images.device, dtype=images.dtype, generator=gen)

    pepper = r < (p / 2)
    salt   = r > (1.0 - p / 2)

    imgs = imgs.masked_fill(pepper.expand_as(imgs), 0.0)
    imgs = imgs.masked_fill(salt.expand_as(imgs),   1.0)

    return imgs



def speckle_noise(images, vs, gen):
    B,C,H,W = images.shape; N = len(vs)
    imgs = images.unsqueeze(1).expand(B,N,C,H,W)
    var  = torch.tensor(vs, device=images.device, dtype=images.dtype).view(1,N,1,1,1)
    noise = torch.randn(imgs.shape, device=images.device, dtype=images.dtype, generator=gen) * torch.sqrt(var)    
    return (imgs + imgs*noise).clamp(0,1)


def adjust_brightness(images, ds):
    B,C,H,W = images.shape; N = len(ds)
    imgs = images.unsqueeze(1).expand(B,N,C,H,W)
    d    = torch.tensor(ds, device=images.device, dtype=images.dtype).view(1,N,1,1,1)
    return (imgs + d).clamp(0,1)

def box_blur(images, ks, gen=None):
    # ks: list of kernel sizes
    B, C, H, W = images.shape
    outs = []
    for k in ks:
        k = int(k)
        if k <= 1:
            outs.append(images)
            continue
        kernel = torch.ones((C, 1, k, k), device=images.device, dtype=images.dtype) / float(k * k)
        y = F.conv2d(images, kernel, padding=k//2, groups=C)
        outs.append(y)
    return torch.stack(outs, dim=1).clamp(0, 1)
