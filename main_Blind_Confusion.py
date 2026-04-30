#!/usr/bin/env python3
import os, gc, sys, json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import random, math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import inspect
from skimage.metrics import structural_similarity as ssim

from corruptions import *
from models import load_model

# Constants 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
EPS    = 1e-8

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

CORRUPTION_TYPES = [
    "confusion",    
    "randomstripesvertical",
    "randomstripeshorizontal",
    "randomlines",
    "randomcrosses",
    "structuredsquarewavehorizontal",
    "structuredsquarewavevertical",
    "coloredimpulse",
    "gaussianblur",
    "gaussian",
    "saltnpepper",      
    "speckle",
    "uniform",
    "brightening",
    "darkening"
]

CORRUPTION_IDS = {
    "confusion": 1,
    "randomstripesvertical": 2,
    "randomstripeshorizontal": 3,
    "randomlines": 4,
    "randomcrosses": 5,
    "structuredsquarewavehorizontal": 6,
    "structuredsquarewavevertical": 7,
    "coloredimpulse": 8,
    "gaussianblur": 9,
    "gaussian": 10,
    "saltnpepper": 11,
    "speckle": 12,
    "uniform": 13,
    "brightening": 14,
    "darkening": 15
}

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
        return structured_square_wave_noise(images, intensities, direction='horizontal')
    elif nt == "structuredsquarewavevertical": 
        return structured_square_wave_noise(images, intensities, direction='vertical')
    elif nt == "coloredimpulse": return colored_impulse_noise(images, intensities, gen=gen)
    elif nt == "gaussianblur": return gaussian_blur(images, intensities, gen=gen)
    elif nt == "gaussian":    return gaussian_noise(images, intensities, gen)
    elif nt == "saltnpepper": return salt_and_pepper_noise(images, intensities, gen)
    elif nt == "speckle":     return speckle_noise(images, intensities, gen)
    elif nt == "uniform":     return uniform_noise(images, intensities, gen)
    elif nt == "brightening":   return adjust_brightness(images, intensities)
    elif nt == "darkening":     return adjust_brightness(images, intensities)
    else:
        raise ValueError(f"Noise Type not found: {nt}")

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
def model_inference_in_batches(
    model, model_name, mean, std, input_size,
    clean_imgs, image_info,
    noise_type, noise_levels,
    label_to_synset, index_to_synset,
    seed_offset, max_chunk, IQA=False,
    micro_bs=None
):
    noise_type_lower = noise_type.lower()

    K = len(noise_levels)
    results = {}

    for start in range(0, K, max_chunk):
        end = min(K, start + max_chunk)

        chunk_intensities = noise_levels[start:end]

        gen_chunk = torch.Generator(device=DEVICE).manual_seed(SEED + seed_offset + start)

        chunk_raw = apply_noise(clean_imgs, noise_type_lower, chunk_intensities, gen_chunk)
        # chunk burada [B, ck, C, H, W]
        chunk = (chunk_raw - mean) / std

        b, ck, C, H, W = chunk.shape
        flat = chunk.reshape(b * ck, C, H, W)

        if micro_bs is None:
            micro_bs = 4

        probs_all = []
        with gpu_lock, torch.inference_mode():
            for s in range(0, flat.size(0), micro_bs):
                out = model(flat[s:s+micro_bs])
                logits = out[0] if isinstance(out, tuple) else out
                probs_all.append(F.softmax(logits, dim=1))

        probs = torch.cat(probs_all, dim=0)
        probs = probs.view(b, ck, -1).transpose(0, 1)   # [ck,B,num_classes]

        for ck_i, intensity in enumerate(chunk_intensities):
            prob_intensity = probs[ck_i]
            for i, info in enumerate(image_info):
                filename = info["filename"]

                top1p, pred0 = torch.max(prob_intensity[i], dim=0)
                pred_idx = int(pred0.item()) + 1
                gt_idx = int(info["gt_idx"])

                pred_syn = index_to_synset.get(pred_idx, f"Label-{pred_idx}")
                gt_syn = label_to_synset.get(gt_idx) or index_to_synset.get(gt_idx, f"Label-{gt_idx}")

                pred_wnid = pred_syn.split("-", 1)[0]
                gt_wnid = gt_syn.split("-", 1)[0]
                corr = int(pred_wnid == gt_wnid)

                if IQA:
                    orig_img = clean_imgs[i].detach().cpu()
                    noisy_img = chunk_raw[i, ck_i].detach().cpu()
                    ssim_v = compute_ssim(orig_img, noisy_img)
                    psnr_v = compute_psnr(orig_img, noisy_img)
                    kl = compute_kl(orig_img, noisy_img)
                else:
                    ssim_v = 0.0
                    psnr_v = 0.0
                    kl = 0.0

                results.setdefault(filename, {"gt_idx": gt_idx, "model_results": {}})
                model_res = results[filename]["model_results"].setdefault(model_name, {})
                model_res.setdefault(noise_type_lower, {})[intensity] = {
                    "top1p": float(top1p.item()),
                    "pred_idx": pred_idx,
                    "corr": corr,
                    "KL": float(kl),
                    "SSIM": float(ssim_v),
                    "PSNR": float(psnr_v),
                }

        del chunk, flat, probs, probs_all, chunk_raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

# ----------------------------------------------------------
# ---------------        Results         -----------------
# ----------------------------------------------------------


def append_jsonl(jsonl_path, batch_results, model_name, noise_type, noise_id):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for filename, info in batch_results.items():
            gt_idx = int(info["gt_idx"])

            noise_res = (
                info["model_results"]
                .get(model_name, {})
                .get(noise_type, {})
            )

            for intensity, rec in noise_res.items():
                row = {
                    "filename": filename,
                    "gt_idx": gt_idx,
                    "pred_idx": int(rec["pred_idx"]),
                    "model": model_name,
                    "noise": noise_type,
                    "noise_id": noise_id,
                    "intensity": intensity,
                    "top1p": float(rec["top1p"]),
                    "corr": int(rec["corr"]),
                    "KL": float(rec["KL"]),
                    "SSIM": float(rec["SSIM"]),
                    "PSNR": float(rec["PSNR"])
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress(path, expected_model):
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("model") != expected_model:
            return 0
        return int(obj.get("next_batch", 0))
    except Exception:
        return 0


def save_progress(path, model_name, next_batch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "next_batch": int(next_batch)}, f)
    os.replace(tmp, path)



# ----------------------------------------------------------
# ---------------        Main         -----------------
# ----------------------------------------------------------

def build_transform(input_size: int):
    resize_size = int(round(input_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])

def worker_init_fn(worker_id):
    ws = SEED + worker_id
    random.seed(ws); np.random.seed(ws); torch.manual_seed(ws)

def main(noise_type, ds_folder, synset_words_file, synset_map_file, out_dir, models_list, IQA=False, batch_size=16, micro_bs = 4, max_chunk=2):

    noise_levels = {
            "brightening":  [0.0, 0.197500, 0.227033, 0.256990, 0.300000, 0.320327, 0.353633, 0.388925, 0.425477, 0.465007, 0.508593, 0.557298, 0.613276, 0.678427, 0.812500, 0.912500],
            "coloredimpulse": [0.0, 0.200000, 0.248205, 0.296410, 0.344662, 0.392874, 0.441068, 0.489282, 0.537473, 0.585615, 0.631250, 0.682103, 0.721875, 0.778615, 0.850000, 0.875000],
            "confusion": [0.0, 0.034375, 0.099524, 0.165714, 0.231905, 0.275000, 0.363054, 0.428810, 0.495000, 0.561190, 0.600000, 0.691905, 0.758095, 0.824235, 0.890148, 0.950000],
            "darkening": [0.0, -0.197500, -0.221484, -0.245698, -0.272152, -0.300000, -0.328439, -0.359029, -0.392788, -0.429257, -0.470470, -0.518673, -0.572834, -0.639369, -0.812500, -0.912500],
            "gaussian": [0.0, 0.200000, 0.215877, 0.233583, 0.253220, 0.275102, 0.299665, 0.327618, 0.350000, 0.397140, 0.442095, 0.497141, 0.566771, 0.659062, 0.812500, 0.912500],
            "gaussianblur": [0.0, 0.200000, 0.212581, 0.225725, 0.239457, 0.255071, 0.275061, 0.295945, 0.324851, 0.358450, 0.400270, 0.457669, 0.530273, 0.633498, 0.800000, 0.981250],
            "randomcrosses": [0.0, 0.200000, 0.243181, 0.287766, 0.333902, 0.381617, 0.430949, 0.482467, 0.535982, 0.591800, 0.650006, 0.711022, 0.800000, 0.842076, 0.909375, 0.981250],
            "randomlines": [0.0, 0.200000, 0.241770, 0.284979, 0.329840, 0.376474, 0.424703, 0.475391, 0.528407, 0.583955, 0.642298, 0.703639, 0.800000, 0.836930, 0.909375, 0.981250],
            "randomstripeshorizontal": [0.0, 0.157500, 0.214288, 0.268463, 0.326027, 0.380476, 0.436295, 0.490727, 0.546912, 0.606250, 0.659955, 0.718750, 0.770910, 0.825357, 0.875000, 0.935625],
            "randomstripesvertical": [0.0, 0.157500, 0.215595, 0.271339, 0.328826, 0.385743, 0.445223, 0.504683, 0.559063, 0.615889, 0.662500, 0.732946, 0.787500, 0.846786, 0.900580, 0.950000],
            "saltnpepper": [0.0, 0.200000, 0.250029, 0.300057, 0.350046, 0.425000, 0.450089, 0.500064, 0.550066, 0.600094, 0.650148, 0.690625, 0.771875, 0.800024, 0.849998, 0.900000],
            "speckle": [0.0, 0.200000, 0.219068, 0.239530, 0.262849, 0.289205, 0.319346, 0.353244, 0.393237, 0.439617, 0.493125, 0.559144, 0.625000, 0.729364, 0.909375, 0.981250],
            "structuredsquarewavehorizontal": [0.0, 0.200000, 0.300000, 0.338200, 0.393801, 0.445407, 0.494947, 0.543087, 0.590864, 0.639007, 0.688083, 0.738617, 0.803125, 0.850077, 0.893750, 0.987500],
            "structuredsquarewavevertical": [0.0, 0.200000, 0.300000, 0.338150, 0.393949, 0.445564, 0.494977, 0.543695, 0.591525, 0.639585, 0.688767, 0.739750, 0.812500, 0.849890, 0.912500, 0.987500],
            "uniform": [0.0, 0.200000, 0.218172, 0.239784, 0.263263, 0.289441, 0.318856, 0.351375, 0.387370, 0.428173, 0.474241, 0.550000, 0.587450, 0.681250, 0.743286, 0.800000]  
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


    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    noise_out_dir = os.path.join(out_dir, noise_type)
    os.makedirs(noise_out_dir, exist_ok=True)

    # parallelize across at most 2 models to limit GPU memory
    for model_name in models_list:
        if not model_name:
            continue
        model_tuple = load_model(model_name, DEVICE)
        if model_tuple is None:
            print(f"WARNING: could not load model '{model_name}'", file=sys.stderr)
            continue
        model, (mean, std, input_size) = model_tuple
        if isinstance(input_size, (list, tuple)):
            input_size = int(input_size[-1])
    
        transform = build_transform(input_size)
        dataset = ILSVRCImageDataset(ds_folder, transform)
        bs = min(batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=bs, shuffle=False,
                            num_workers=min(4, os.cpu_count() or 1),
                            pin_memory = torch.cuda.is_available(),
                            worker_init_fn=worker_init_fn)
        with gpu_lock:
            model = model.to(DEVICE).eval()
            mean, std = mean.to(DEVICE), std.to(DEVICE)
        
        model_dir = os.path.join(noise_out_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        jsonl_path = os.path.join(model_dir, "results.jsonl")
        progress_path = os.path.join(model_dir, "progress.json")
        start_batch = load_progress(progress_path, model_name)  
            
        for batch_idx, (imgs, gts, filenames) in enumerate(tqdm(
        loader,
        desc=f"[{noise_type}] | {model_name}",
        unit="batch",
        total=len(loader)
        )):
            if batch_idx < start_batch:
                continue  # skip already processed batch
            clean_imgs = imgs.to(DEVICE)
            image_info = []
            for gt_tensor, filename in zip(gts, filenames):
                gt = int(gt_tensor)
                image_info.append({
                    "filename":     filename,
                    "gt_idx":     gt
                })
            seed_offset = batch_idx * 100000 + CORRUPTION_IDS[noise_type]
            batch_results = model_inference_in_batches(
                model, model_name, mean, std, input_size,
                clean_imgs, image_info,
                noise_type, noise_levels[noise_type],
                label_to_synset, index_to_synset,
                seed_offset, IQA=IQA,
                max_chunk=max_chunk, micro_bs=micro_bs
            )
            append_jsonl(jsonl_path, batch_results, model_name, noise_type, CORRUPTION_IDS[noise_type])
            save_progress(progress_path, model_name, batch_idx + 1)

        # free GPU memory
        with gpu_lock:
            model.to("cpu")
        del model, mean, std
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        





def run_all_noise_types(ds_folder, synset_words, synset_map, out_base, models_list, IQA, batch_size, micro_bs, max_chunk):
    noise_types = [
        "coloredimpulse","randomlines", 
        "randomcrosses","confusion","gaussianblur",
        "randomstripesvertical", "randomstripeshorizontal",
        "structuredsquarewavevertical", "structuredsquarewavehorizontal",
        "gaussian","saltnpepper","speckle","uniform","brightening", "darkening"
    ]
    summary = {}
    for nt in noise_types:
        main(nt, ds_folder, synset_words, synset_map, out_base, 
                models_list, IQA, batch_size, micro_bs, max_chunk)
    return summary

if __name__ == "__main__":
    import argparse
    reduced_folder     = "/workspace/project/data/val"
    synset_map_file    = "//workspace/project/Blind_Confusion_Classifiers/map_synset.txt"
    synset_words_file  = "/workspace/project/Blind_Confusion_Classifiers/synset_words.txt"
    output_dir_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_results")

    parser = argparse.ArgumentParser(description="Corruption robustness analysis")
    parser.add_argument("--noise_type", choices=[ 
        "confusion", "coloredimpulse", "gaussianblur",
        "randomlines", "randomcrosses","randomstripesvertical", "randomstripeshorizontal",
        "structuredsquarewavehorizontal", "structuredsquarewavevertical",
        "gaussian","saltnpepper","speckle","uniform","brightening", "darkening"
    ], help="which noise to process (if omitted, --all is assumed)")
    parser.add_argument("--config",     default="classification_models.json", help="model-group JSON")
    parser.add_argument("--group",      required=True,                    help="group names: traditional, classic, advanced, robustified, transformers")
    parser.add_argument("--all", action="store_true", help="run all noise types")
    parser.add_argument("--data_dir", default=reduced_folder, help=f"path to reducedValSet (default: {reduced_folder})")
    parser.add_argument("--synset_words", default=synset_words_file, help=f"synset words txt (default: {synset_words_file})")
    parser.add_argument("--synset_map", default=synset_map_file, help=f"synset-to-label map txt (default: {synset_map_file})")
    parser.add_argument("--out_dir", default=output_dir_default, help=f"where to save results (default: {output_dir_default})")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument("--max_chunk", type=int, default=16, help="max noise levels to process in one forward pass (default: 16)")
    parser.add_argument("--micro_bs", type=int, default=8, help="micro batch size (default: 8)")
    parser.add_argument("--iqa", action="store_true", help="compute image quality metrics (SSIM, PSNR, KL) (default: False)")
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
    IQA = args.iqa
    batch_size = args.batch_size
    micro_bs   = args.micro_bs
    max_chunk  = args.max_chunk

    if args.all or args.noise_type is None:
        run_all_noise_types(args.data_dir, args.synset_words, args.synset_map, args.out_dir, models_list, IQA, batch_size, micro_bs, max_chunk)
    else:
        main(args.noise_type, args.data_dir, args.synset_words, args.synset_map, args.out_dir, models_list, IQA, batch_size, micro_bs, max_chunk)
