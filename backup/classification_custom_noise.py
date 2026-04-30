#!/usr/bin/env python3
import os, gc, sys, json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import random, math
import numpy as np
from PIL import Image

import timm
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import inspect
from skimage.metrics import structural_similarity as ssim

# Constants 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
TOPK   = 5
EPS    = 1e-8
IQA    = False

# Reproducibity Configurations
os.environ['PYTHONHASHSEED']    = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['OMP_NUM_THREADS']   = '1'
os.environ['MKL_NUM_THREADS']   = '1'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.set_num_interop_threads(1)

# ----------------------------------------------------------
# --------------            Dataset         ----------------
# ----------------------------------------------------------

class ILSVRCImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.filenames = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        base = os.path.splitext(filename)[0]
        gt = int(base.split("_")[-1])
        img = Image.open(os.path.join(self.folder, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, gt, filename

# ----------------------------------------------------------
# ---------------- Vectorized Custom Noises ----------------
# ----------------------------------------------------------
def apply_noise(images, nt, intensities, gen):
    
    nt = nt.lower()
    if nt == "confusion":
        return confusion_blocks(images, intensities, gen=gen)
    elif nt == "randomstripesvertical":
        return random_stripes(images, intensities, vertical=True, gen=gen)
    elif nt == "randomstripeshorizontal":
        return random_stripes(images, intensities, vertical=False, gen=gen)
    elif nt == "randomlines":
        return random_lines(images, intensities, gen=gen)
    elif nt == "randomcrosses":
        return random_crosses(images, intensities, gen=gen)
    elif nt == "structuredsquarewavehorizontal":
        return structured_square_wave_noise(
            images, intensities, direction='horizontal')
    elif nt == "structuredsquarewavevertical":
        return structured_square_wave_noise(
            images, intensities, direction='vertical')
    elif nt == "coloredimpulse":
        return colored_impulse_noise(images, intensities, gen=gen)
    elif nt == "gaussianblur":
        return gaussian_blur(images, intensities, gen=gen)
    else:
        raise ValueError(f"Noise Type not found: {nt}")

def confusion_blocks(images, intensities, block_size=8, 
                            max_coverage_limit=0.5, gen=None):
    
    B,C,H,W = images.shape
    N = len(intensities)

    blocks_x, blocks_y = W//block_size, H//block_size
    total_candidate_blocks  = blocks_x * blocks_y
    block_image_coverage = (torch.tensor(intensities, device=images.device,
                                 dtype=images.dtype) / 10) * max_coverage_limit
    blocks_to_corrupt = (block_image_coverage * total_candidate_blocks ).floor().long()  
    
    block_order_randomized = torch.rand((N, total_candidate_blocks ), 
                                            generator=gen, device=images.device)
    mask_blocks = torch.zeros_like(block_order_randomized)

    for i in range(N):
        num_block = blocks_to_corrupt[i].item()
        random_scores = block_order_randomized[i]

        if num_block > 0:
            smallest_scores = random_scores.topk(num_block, largest=False).values
            cutoff_score = smallest_scores.max()
            mask_blocks[i] = (random_scores <= cutoff_score).float()
        else:
            mask_blocks[i].zero_()

    mask_blocks = mask_blocks.view(N, blocks_y, blocks_x)
    mask_pix = mask_blocks.unsqueeze(1).repeat(1, C, 1, 1)\
               .repeat_interleave(block_size, dim=2)\
               .repeat_interleave(block_size, dim=3)  # [N, C, H, W]
    mask_pix = mask_pix.unsqueeze(0).expand(B, N, C, H, W)
    
    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()
    noise = torch.rand(imgs.shape, device=imgs.device, dtype=imgs.dtype, generator=gen)
    imgs = torch.where(mask_pix.bool(), noise, imgs)
    return imgs

def random_stripes(images, intensities, vertical=True, max_coverage_limit=0.5, gen=None):
    
    B,C,H,W = images.shape
    N = len(intensities)
    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()
    stripe_image_coverage = (torch.tensor(intensities, device=images.device, 
                                dtype=images.dtype) / 10) * max_coverage_limit

    if vertical:  # mask on W
        col_rand = torch.rand((N, W), generator=gen, device=images.device)
        thresh = stripe_image_coverage.view(N,1)
        mask_cols = (col_rand < thresh)  # [N, W]
        mask = mask_cols.view(1,N,1,1,W).expand(B,N,C,H,W)
    else: # mask on H
        row_rand = torch.rand((N, H), generator=gen, device=images.device)
        thresh = stripe_image_coverage.view(N,1)
        mask_rows = (row_rand < thresh)  # [N, H]
        mask = mask_rows.view(1,N,1,H,1).expand(B,N,C,H,W)
    imgs[mask] = 0.0
    return imgs

def random_lines(images, intensities,line_count_factor=7000,
                            line_length_range=(2,6), gen=None):
    B,C,H,W = images.shape
    device = images.device
    N = len(intensities)
    line_count = (torch.tensor(intensities, device=device, dtype=torch.float32) 
                            / 10 * line_count_factor).long()  # [N]
    total_lines = int(line_count.sum())
    if total_lines == 0:
        return images.unsqueeze(1).expand(B, N, C, H, W)

    # configure line params
    center_x  = torch.randint(0, W, (total_lines,), generator=gen, device=device)
    center_y  = torch.randint(0, H, (total_lines,), generator=gen, device=device)
        # (lengths): set [2,6) pixel length, set line_length_range for different random length
    lengths = torch.randint(*line_length_range, (total_lines,), generator=gen, device=device)
        # (angles): set 0 and 90 degree direction, change values for different angles
    angles  = (torch.randint(0,2,(total_lines,), generator=gen, device=device) * 90).float() 
    colors  = torch.rand((total_lines, C), generator=gen, device=device)
    variant_index   = torch.repeat_interleave(torch.arange(N, device=device), line_count)

    mask = torch.zeros(N, H, W, device=device, dtype=torch.bool)
    pixel_value_mask = torch.zeros(N, C, H, W, device=device, dtype=images.dtype)

    # calculate pos
    max_len = line_length_range[1]
    pixel_pos = torch.arange(-max_len//2, max_len//2+1, device=device).view(1,-1)
    cos_angle = torch.cos(torch.deg2rad(angles)).view(-1,1)
    sin_angle = torch.sin(torch.deg2rad(angles)).view(-1,1)
    x_d = (pixel_pos * cos_angle).round().long()  
    y_d = (pixel_pos * sin_angle).round().long()

    x_pos = (center_x.view(-1,1) + x_d).clamp(0, W-1).flatten()
    y_pos = (center_y.view(-1,1) + y_d).clamp(0, H-1).flatten()
    variant_index_rep = torch.repeat_interleave(variant_index, x_d.size(1))

    mask[variant_index_rep, y_pos, x_pos] = True
    pixel_value_mask[variant_index_rep, :, y_pos, x_pos] = colors.repeat_interleave(x_d.size(1), dim=0)

    base_imgs = images.unsqueeze(1).expand(B, N, C, H, W)  
    return torch.where(
        mask.view(1, N, H, W).unsqueeze(2),          
        pixel_value_mask.unsqueeze(0),                
        base_imgs                                        
    )

def random_crosses(images, intensities, cross_count_factor=12000, gen=None):
    B,C,H,W = images.shape
    device = images.device

    cross_count = (torch.tensor(intensities, device=device, dtype=torch.float32)
                        / 10 * cross_count_factor).long() # [N]
    N = len(intensities)
    total_crosses = int(cross_count.sum())
    if total_crosses == 0:
        return images.unsqueeze(1).expand(B, N, C, H, W).clone()

    # configure params
    center_y = torch.randint(1, H-1, (total_crosses,), generator=gen, device=device)
    center_x = torch.randint(1, W-1, (total_crosses,), generator=gen, device=device)
        # 0=plus,1=x
    rand_plus_cross = torch.randint(0,2,(total_crosses,), generator=gen, device=device)  
    colors = torch.rand((total_crosses, C), generator=gen, device=device)
    variant_index = torch.repeat_interleave(torch.arange(N, device=device), cross_count)

    # calculate pos
    plus  = torch.tensor([(0,0),(-1,0),(1,0),(0,-1),(0,1)], device=device)
    cross = torch.tensor([(0,0),(-1,-1),(1,1),(-1,1),(1,-1)], device=device)
    pixel_pos  = torch.stack([plus, cross], dim=0)  # [2,5,2]

    mask = torch.zeros(N, H, W, device=device, dtype=torch.bool)
    pixel_value_mask = torch.zeros(N, C, H, W, device=device, dtype=images.dtype)

    # for each sampled get 5 points
    centers = torch.stack([center_y, center_x], dim=1)  # [total_crosses,2]
    for shape_id in (0,1):
        sel = rand_plus_cross==shape_id
        cross_pos = centers[sel].unsqueeze(1) + pixel_pos[shape_id].unsqueeze(0)  # [m,5,2]
        y_pos  = cross_pos[...,0].clamp(0,H-1).flatten()
        x_pos  = cross_pos[...,1].clamp(0,W-1).flatten()
        var_idx= torch.repeat_interleave(variant_index[sel], 5)
        cols= torch.repeat_interleave(colors[sel], 5, dim=0)  # [m*5, C]
        mask[var_idx, y_pos, x_pos] = True
        pixel_value_mask[var_idx, :, y_pos, x_pos] = cols

    base_imgs = images.unsqueeze(1).expand(B, N, C, H, W)
    result = torch.where(
        mask.view(1,N,H,W).unsqueeze(2),
        pixel_value_mask.unsqueeze(0),
        base_imgs
    )
    return result

def structured_square_wave_noise(images, intensities, max_amplitude=0.5,
                                        direction='horizontal', period=8):
    B,C,H,W = images.shape
    N = len(intensities)
    imgs = images.unsqueeze(1).expand(B,N,C,H,W).clone()
    amplitude_variety = (torch.tensor(intensities, device=images.device, 
                                dtype=images.dtype) / 10) * max_amplitude

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

def colored_impulse_noise(images, intensities, max_noise_limit=0.5, gen=None):

    B,C,H,W = images.shape
    N = len(intensities)
    imgs = images.unsqueeze(1).expand(B, N, C, H, W).clone()
    pix_coverage    = ((torch.tensor(intensities, device=imgs.device, 
                        dtype=imgs.dtype) / 10) * max_noise_limit).view(1,N,1,1,1)
    mask = torch.rand((B,N,1,H,W), generator=gen, device=imgs.device) < pix_coverage
     
    noise = torch.rand(imgs.shape, device=imgs.device, dtype=imgs.dtype, generator=gen)
    imgs = torch.where(mask.expand_as(imgs), noise, imgs)
    return imgs.clamp(0,1)

def gaussian_blur(images, sigmas, gen=None):
    B,C,H,W = images.shape
    outputs = []
    for sigma in sigmas:
        if sigma == 0:
            outputs.append(images)
            continue
        # kernel size 6σ for even and large size
        k = max(3, int(2 * round(3 * sigma) + 1))
        coords = torch.arange(k, device=DEVICE) - k//2
        xx, yy = torch.meshgrid(coords, coords, indexing='xy')
        kern = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kern = kern / kern.sum()
        kernel = kern.view(1,1,k,k).repeat(C,1,1,1)
        blurred = F.conv2d(images, kernel, padding=k//2, groups=C)
        outputs.append(blurred)
    stacked = torch.stack(outputs, dim=1)  # [B, N, C, H, W]
    return stacked.clamp(0,1)


# ----------------------------------------------------------
# ---------------   Image Quality Metrics   ----------------
# ----------------------------------------------------------

def compute_ssim(img1, img2):

    img1_np = img1.permute(1,2,0).cpu().numpy()
    img2_np = img2.permute(1,2,0).cpu().numpy()
    kwargs = {}
    sig = inspect.signature(ssim)
    if "channel_axis" in sig.parameters:
        kwargs["channel_axis"] = 2
    elif "multichannel" in sig.parameters:
        kwargs["multichannel"] = True
    return ssim(img1_np, img2_np, data_range=1.0, **kwargs)

def compute_psnr(orig: torch.Tensor, noisy: torch.Tensor, max_val: float = 1.0) -> float:
    diff = orig - noisy
    mse = float(torch.mean(diff * diff).item())  
    if mse == 0:
        return float('inf')  # perfect match => inf PSNR
    return 10.0 * math.log10((max_val * max_val) / mse)

# intuitive approach to compare noise distribution level difference ref. to original image
def compute_kl(orig: torch.Tensor, noisy: torch.Tensor) -> float:
    orig_f = orig.detach().cpu().flatten()
    noisy_f = noisy.detach().cpu().flatten()
    # histograms
    h0 = torch.histc(orig_f, bins=256, min=0.0, max=1.0)
    h1 = torch.histc(noisy_f, bins=256, min=0.0, max=1.0)

    prob_h0 = h0 / (h0.sum() + EPS)
    prob_h1 = h1 / (h1.sum() + EPS)

    return (prob_h0 * torch.log((prob_h0 + EPS) / (prob_h1 + EPS))).sum().item()

# ----------------------------------------------------------
# ---------------        Inference         -----------------
# ----------------------------------------------------------

# Serial gpu access
gpu_lock = Lock()

def model_inference_in_batches(model_name, model_tuple,
                                clean_imgs, image_info,
                                noise_type, noise_levels,
                                label_to_synset, index_to_synset,
                                seed_offset,
                                max_chunk=2):
    model, (mean, std, input_size) = model_tuple
    if isinstance(input_size, (list, tuple)):
        input_size = int(input_size[-1])
    
    with gpu_lock:
        model = model.to(DEVICE)
        mean, std = mean.to(DEVICE), std.to(DEVICE)
        
    noise_type_lower = noise_type.lower()

    # collect raw variants
    raw_variants, intensity_levels = [], []
    gen = torch.Generator(device=DEVICE).manual_seed(SEED + seed_offset)
    
    noisy = apply_noise(clean_imgs, noise_type_lower, noise_levels, gen)
    for i, level in enumerate(noise_levels):
        raw_variants.append(noisy[:, i])
        intensity_levels.append(level)

    B = clean_imgs.size(0)
    K = len(intensity_levels)
    results = {}
    clean_cache = {}

    # chunk through noise levels
    for start in range(0, K, max_chunk):
        end = min(K, start + max_chunk)
        chunk_vars = raw_variants[start:end]
        chunk_intensities = intensity_levels[start:end]

        # normalize & stack
        normed = []
        for var in chunk_vars:
            img = var
            if img.size(-1) != input_size:
                img = F.interpolate(img,
                           size=(input_size, input_size),
                           mode='bilinear',
                           align_corners=False
                           )

            normed.append((img - mean) / std)
        stacked = torch.stack(normed, dim=1)  # [B, chunk, C, H, W]
        b, ck, C, H, W = stacked.shape
        batch_tensor = stacked.view(b * ck, C, H, W)

        # forward under lock
        with gpu_lock, torch.no_grad():
            out = model(batch_tensor)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
        probs = probs.view(b, ck, -1).transpose(0, 1)  # [chunk, B, num_classes]

        # raw image info 
        for ck_i, intensity in enumerate(chunk_intensities):
            prob_intensity = probs[ck_i]
            for i, info in enumerate(image_info):
                filename, gt_label = info["filename"], info["gt_label"]
                results.setdefault(filename, {"gt": gt_label, "model_results": {}})
                model_res = results[filename]["model_results"].setdefault(model_name, {})
                topk_p, topk_i = torch.topk(prob_intensity[i], TOPK)

                labels = [index_to_synset.get(int(x)+1, f"Label{int(x)+1}") for x in topk_i]
                
                # Metrics for image quality assesment
                orig_img = clean_imgs[i].detach().cpu()
                noisy_img   = chunk_vars[ck_i][i].detach().cpu()
                if IQA is True:
                    ssim_v = compute_ssim(orig_img, noisy_img)
                    psnr_v = compute_psnr(orig_img, noisy_img)
                    kl = compute_kl(orig_img, noisy_img)
                else:
                    ssim_v, psnr_v, kl = 0.0, 0.0, 0.0

                # Top-K
                pred_syn = labels[0]
                gt_syn = info["gt_label"]
                rec = {
                    "top1p": topk_p[0].item(),
                    "corr":  int(pred_syn == gt_syn),
                    "KL":    kl,
                    "SSIM":  ssim_v,
                    "PSNR":  psnr_v,
                    "topkl": labels,
                    "topkp": [p.item() for p in topk_p]
                }


                model_res.setdefault(noise_type_lower, {})[intensity] = rec
    
    # Move model back to CPU to free GPU memory
    with gpu_lock:
        model.to('cpu')
        torch.cuda.empty_cache()
    
    return results

# ----------------------------------------------------------
# ---------------        Results         -----------------
# ----------------------------------------------------------
def merge_results(all_res, new_res):
    for fn, info in new_res.items():
        if fn not in all_res:
            all_res[fn] = {"gt": info["gt"], "model_results": {}}
        for m, model_res in info["model_results"].items():
            all_res[fn]["model_results"].setdefault(m, {}).update(model_res)
    return all_res

def write_top5_results_last_noise(f, noise_res, key, header, sep):
    f.write("\n" + header + "\n")
    f.write(sep + "\n")
    if key in noise_res:
        r = noise_res[key]
        for i, lbl in enumerate(r["topkl"][:TOPK], 1):
            f.write(f"{i}. {lbl:60} {r['topkp'][i-1]:.6f}\n")
    f.write("\n")

def document_results(all_results, noise_type, noise_levels, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    if not all_results:
        return {}
    example = next(iter(all_results.values()))["model_results"]
    if not example:
        return {}
    out_paths = {}
    noise_type_lower = noise_type.lower()
    RESULT_HEADERS = {
        "confusion":   ("intensity",      "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "randomstripesvertical": ("intensity", "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "randomstripeshorizontal": ("intensity", "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "randomlines": ("intensity",      "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "randomcrosses": ("intensity",    "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "structuredsquarewavehorizontal": ("intensity", "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "structuredsquarewavevertical": ("intensity", "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "coloredimpulse": ("intensity",   "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
        "gaussianblur": ("intensity",   "intensity  Value | Top-1  | Prob | Corr. | KL | SSIM | PSNR | Label |   GT", "Top-5 Results", "-"*40),
    }
    header_key, main_hdr, top5_hdr, sep = RESULT_HEADERS[noise_type_lower]
    for model_name in example:
        path = os.path.join(output_dir, f"{model_name}_{noise_type_lower}_results.txt")
        out_paths[model_name] = path
        with open(path, "w") as f:
            for fn, info in all_results.items():
                f.write(f"IMAGE: {fn}\nGT: {info['gt']}\n{'='*60}\n")
                res_all = info["model_results"].get(model_name, {}).get(noise_type_lower, {})

                f.write(f"{main_hdr}\n")
                for v in noise_levels:
                    r = res_all.get(v)
                    if not r: continue
                    corr = "1" if r["corr"] else "0"
                    if header_key == "placement":
                        label = str(v)
                    else:
                        label = f"{v*10}%"
                    f.write(f"{header_key} {label:>6} | {r['top1p']:.4f} | {corr} | {r['KL']:.4f} | {r['SSIM']:.4f} | {r['PSNR']:.2f} |  {r['topkl'][0]} | {info['gt']}\n")
                write_top5_results_last_noise(f, res_all, noise_levels[-1], top5_hdr, sep)
                f.write("\n")
    return out_paths



# ----------------------------------------------------------
# ---------------        Main         -----------------
# ----------------------------------------------------------
def worker_init_fn(worker_id):
    ws = SEED + worker_id
    random.seed(ws); np.random.seed(ws); torch.manual_seed(ws)

def main(noise_type, ds_folder, synset_words_file, synset_map_file, out_dir, models_list):
    noise_type = noise_type.lower()

    noise_levels = {
        "randomstripesvertical":          list(range(0,16)),
        "randomstripeshorizontal":        list(range(0,16)),
        "structuredsquarewavehorizontal": list(range(0,16)),
        "structuredsquarewavevertical":   list(range(0,16)),
        "confusion":                      list(range(0,16)),       
        "coloredimpulse":                 list(range(0,16)),
        "randomlines":                    list(range(0,11)),
        "randomcrosses":                  list(range(0,11)),
        "gaussianblur":                   list(range(0,11))
    }

    label_to_synset = {}
    with open(synset_map_file, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                _, ls, rd = p[0], p[1], p[2]
                try: label_to_synset[int(ls)] = f"{p[0]}-{rd}"
                except: pass
    index_to_synset = {}
    with open(synset_words_file, "r") as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) >= 2:
                syn = parts[0]
                desc = "_".join(" ".join(parts[1:]).split(",")[0].split())
                index_to_synset[i] = f"{syn}-{desc}"

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
    ])
    dataset = ILSVRCImageDataset(ds_folder, transform)
    bs = min(16, len(dataset))
    loader = DataLoader(dataset, batch_size=bs, shuffle=False,
                        num_workers=min(4, os.cpu_count() or 1),
                        pin_memory = torch.cuda.is_available(),
                        worker_init_fn=worker_init_fn)

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    models_dict = load_models_parallel(models_list)

    all_results = {}
    for batch_idx, (imgs, gts, filenames) in enumerate(tqdm(
            loader,
            desc=f"Inference [{noise_type}]",
            unit="batch",
            total=len(loader)
        )):
        clean_imgs = imgs.to(DEVICE)
        image_info = []
        for gt_tensor, filename in zip(gts, filenames):
            gt = int(gt_tensor)
            gt0 = int(gt) - 1
            image_info.append({
                "filename":     filename,
                "gt_label":     label_to_synset.get(gt, index_to_synset.get(gt, f"Label {gt}"))
            })

        # parallelize across at most 2 models to limit GPU memory
        model_items = [(name, models_dict[name]) 
                        for name in models_list if name in models_dict]
        for i in range(0, len(model_items), 2):
            group = model_items[i:i+2]
            with ThreadPoolExecutor(max_workers=len(group)) as ex:
                futures = {}
                for model_idx, (model_name, model_tuple) in enumerate(group):
                    seed_offset = batch_idx * len(model_items) + i + model_idx
                    futures[ex.submit(
                        model_inference_in_batches,
                        model_name, model_tuple,
                        clean_imgs, image_info,
                        noise_type, noise_levels[noise_type],
                        label_to_synset, index_to_synset,
                        seed_offset
                    )] = model_name

                for fut, model_name in futures.items():
                    res = fut.result()
                    all_results = merge_results(all_results, res)

        gc.collect()
        torch.cuda.empty_cache()

    document_results(all_results, noise_type, noise_levels[noise_type], out_dir)

# ----------------------------------------------------------
# ---------------       Model Loading      -----------------
# ----------------------------------------------------------

def load_model(name):
    n = name.lower()
    #### Traditional [1-2] ####
    # AlexNet
    if n == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)

    # VGG16
    elif n == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)

    #### ResNet Based [3-11] ####
    elif n == "resnet200d":
        model = timm.create_model("resnet200d", pretrained=True)
        weights = None
    elif n == "resnet101" :
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        model = models.resnet101(weights=weights)
    elif n == "resnet50" :
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    # ResNeXt, WideResNet
    elif n == "resnext101":
        weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        model   = models.resnext101_32x8d(weights=weights)
    elif n == "wideresnet101_2":
        weights = models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        model   = models.wide_resnet101_2(weights=weights)

    # Inception‐ResNet‐v2
    elif n == "inception_resnet_v2":
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        weights = None
    # Inception - GoogleNet
    elif n == "inception_v3" :
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights = weights)

    # MobileNet
    elif n == "mobilenet_v2" :
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights = weights)

    # DenseNets
    elif n == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model   = models.densenet121(weights=weights)
    elif n == "densenet161":
        weights = models.DenseNet161_Weights.IMAGENET1K_V1
        model   = models.densenet161(weights=weights)
    elif n == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        model   = models.densenet201(weights=weights)


    #### Advanced [12-22] ####
    # ConvNeXts 
    elif n == "convnext_xlarge":
        model = timm.create_model("convnext_xlarge", pretrained=True)
        weights = None
    elif n == "convnext_large":
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    elif n == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    elif n == "convnextv2_large":
        model = timm.create_model('convnextv2_large', pretrained=True)
        weights = None
    elif n == "convnextv2_base":
        model = timm.create_model('convnextv2_base', pretrained=True)
        weights = None
    # EfficientNets
    elif n == "efficientnet_v2_l":
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        model   = models.efficientnet_v2_l(weights=weights)
    elif n == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model   = models.efficientnet_v2_s(weights=weights)
    
    elif n == "efficientnet_b7":
        weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b7(weights=weights)
    elif n == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b3(weights=weights)
    elif n == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b4(weights=weights)
    elif n == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
    # NFNets
    elif n == "nfnet_f5":
        model = timm.create_model('dm_nfnet_f5', pretrained=True)
        weights = None
    elif n == "nfnet_f4":
        model = timm.create_model('dm_nfnet_f4', pretrained=True)
        weights = None
    elif n == "nfnet_f2":
        model = timm.create_model('dm_nfnet_f2', pretrained=True)
        weights = None
    # ResNeSt adds split‑attention blocks for robustness
    elif n == "resnest50d":       
        weights = None
        model   = timm.create_model('resnest50d', pretrained=True)
    elif n == "resnest101e":
        weights = None
        model   = timm.create_model('resnest101e', pretrained=True)

    #### Robustly Trained [23-26] ####
    # NoisyStudent
    elif n == "noisystudent_b0":
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_b3":
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_b4":
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_b7":
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        weights = None
    elif n == "noisystudent_v2_l":  # alias olarak böyle çağırmak istersen
        model = timm.create_model('tf_efficientnetv2_l_in21ft1k', pretrained=True)
        weights = None


    # AugMix ResNet-50
    elif n == "augmix":
        model = timm.create_model('resnet50.ram_in1k', pretrained=True)
        weights = None

    # BiT-M & BiT-L (ResNet-V2)
    elif n == "bit_m":
            # BiT-M uses a ResNet-V2-101×1 from Timm
        model   = timm.create_model('resnetv2_101x1_bitm', pretrained=True)
        weights = None

    ####  Vision Transformers [27-30] ####
    # Vanilla ViTs
    elif n == "vit_large16":
        model = timm.create_model('vit_large_patch16_224', pretrained=True)
        weights = None
    elif n == "vit_base16":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        weights = None

    # Swin Transformers
    elif n == "swin_base":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        weights = None
    elif n == "swin_large":
        model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        weights = None
    # DeiT
    elif n == "deit_base":
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
        weights = None
    elif n == "maxvit_base":
        model = timm.create_model('maxvit_base_tf_224', pretrained=True)
        weights = None

    else:
        raise ValueError()

    # torchvision pipeline
    if weights is not None:
        preprocess = weights.transforms()        
        mean = torch.tensor(preprocess.mean).view(3,1,1)
        std  = torch.tensor(preprocess.std).view(3,1,1)
        input_size = preprocess.crop_size if hasattr(preprocess, 'crop_size') else preprocess.resize[1]
       
    # timm pipeline
    else:
        config  = model.default_cfg
        mean = torch.tensor(config['mean']).view(3,1,1)
        std  = torch.tensor(config['std']).view(3,1,1)
        input_size = config['input_size'][1]

    mean = torch.tensor(mean, device=DEVICE).view(3,1,1)
    std  = torch.tensor(std,  device=DEVICE).view(3,1,1)
    model = model.to(DEVICE).eval()

    mean = mean.to(DEVICE)
    std  = std.to(DEVICE)
    return model, (mean, std, input_size)


def load_models_parallel(names):
    workers = min(os.cpu_count() or 1, len(names))
    model_dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_model, name): name for name in names}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading models"):
            name = futures[fut]
            model_dict[name] = fut.result()

    if not model_dict:
        raise RuntimeError("No models loaded")
    return model_dict


def run_all_noise_types(ds_folder, synset_words, synset_map, out_base, models_list):
    noise_types = [
        "coloredimpulse",
        "randomlines", 
        "randomcrosses",
        "confusion",
        "gaussianblur",
        "randomstripesvertical", "randomstripeshorizontal",
        "structuredsquarewavevertical", "structuredsquarewavehorizontal"
    ]
    summary = {}
    for nt in noise_types:
        summary[nt] = main(nt, ds_folder, synset_words, synset_map, os.path.join(out_base, nt), models_list)
    return summary

if __name__ == "__main__":
    import argparse
    reduced_folder     = "TESTSet_ILSVRC"
    synset_map_file    = "map_synset.txt"
    synset_words_file  = "synset_words.txt"
    output_dir_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_results")

    parser = argparse.ArgumentParser(description="Corruption robustness analysis")
    parser.add_argument("--noise_type", choices=[ 
        "confusion", "coloredimpulse", "gaussianblur",
        "randomlines", "randomcrosses",
        "randomstripesvertical", "randomstripeshorizontal",
        "structuredsquarewavehorizontal", "structuredsquarewavevertical"
    ], help="which noise to process (if omitted, --all is assumed)")
    parser.add_argument("--config",     default="classification_models.json", help="model-group JSON")
    parser.add_argument("--group",      required=True,                    help="group names: traditional, classic, advanced, robustified, transformers")
    parser.add_argument("--all", action="store_true", help="run all noise types")
    parser.add_argument("--data_dir", default=reduced_folder, help=f"path to reducedValSet (default: {reduced_folder})")
    parser.add_argument("--synset_words", default=synset_words_file, help=f"synset words txt (default: {synset_words_file})")
    parser.add_argument("--synset_map", default=synset_map_file, help=f"synset-to-label map txt (default: {synset_map_file})")
    parser.add_argument("--out_dir", default=output_dir_default, help=f"where to save results (default: {output_dir_default})")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"ERROR: cannot find config {args.config}", file=sys.stderr)
        sys.exit(1)
    with open(args.config) as f:
        groups = json.load(f)
    if args.group not in groups:
        print(f"ERROR: unknown group '{args.group}'", file=sys.stderr)
        sys.exit(1)
    models_list = groups[args.group]

    if args.all or args.noise_type is None:
        run_all_noise_types(args.data_dir, args.synset_words, args.synset_map, args.out_dir, models_list)
    else:
        main(args.noise_type, args.data_dir, args.synset_words, args.synset_map, args.out_dir, models_list)