import os
import glob
import copy
import torch
from torch import nn
from torch.export import export as aot_export
from torch.export import save as aot_save
from torch.export import load as aot_load
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from model import Net
from time import time
import coremltools as ct
import math

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--LR_dir", type=str, default="testset/RealSR/LR")
parser.add_argument("--HR_dir", type=str, default="testset/RealSR/HR")
parser.add_argument("--SR_dir", type=str, default="result/RealSR")
args = parser.parse_args()

device = torch.device("cuda")

from diffusers import StableDiffusionPipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

vae = pipe.vae
tokenizer = pipe.tokenizer
unet = pipe.unet
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder

from diffusers.models.autoencoders.vae import Decoder
ckpt_halfdecoder = torch.load("./weight/pretrained/halfDecoder.ckpt", weights_only=False)
decoder = Decoder(in_channels=4,
            out_channels=3,
            up_block_types=["UpDecoderBlock2D" for _ in range(4)],
            block_out_channels=[64, 128, 256, 256],
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            norm_type="group",
            mid_block_add_attention=True).to(device)
decoder_ckpt = {}
for k,v in ckpt_halfdecoder["state_dict"].items():
    if "decoder" in k:
        new_k = k.replace("decoder.", "")
        decoder_ckpt[new_k] = v
decoder.load_state_dict(decoder_ckpt, strict=True)

model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))
model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (args.model_dir, args.epoch), weights_only=False))
model = torch.nn.Sequential(
    model.module,
    *decoder.up_blocks,
    decoder.conv_norm_out,
    decoder.conv_act,
    decoder.conv_out,
).to(device)
model.eval()

# model.half()

# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)


test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png")) +
                           glob.glob(os.path.join(args.LR_dir, "*.jpg")) +
                           glob.glob(os.path.join(args.LR_dir, "*.jpeg"))))
test_HR_paths = list(sorted(glob.glob(os.path.join(args.HR_dir, "*.png")) +
                           glob.glob(os.path.join(args.HR_dir, "*.jpg")) +
                           glob.glob(os.path.join(args.HR_dir, "*.jpeg"))))

os.makedirs(args.SR_dir, exist_ok=True)

print("starting export")

class SRWrapper(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # lr is expected in [-1, 1], shape [N,3,H,W]
        sr = self.core(lr)
        # match our runtime post-processing
        sr_mean = sr.mean(dim=[2, 3], keepdim=True)
        sr_std  = sr.std(dim=[2, 3], keepdim=True)
        lr_mean = lr.mean(dim=[2, 3], keepdim=True)
        lr_std  = lr.std(dim=[2, 3], keepdim=True)
        sr = (sr - sr_mean) / (sr_std + 1e-6) * lr_std + lr_mean
        return sr

wrapped = SRWrapper(model).to(device).eval()

example = torch.randn(1, 3, 256, 256, device=device)

export_dir = f"{args.model_dir}/export"
os.makedirs(export_dir, exist_ok=True)

# model_torchscript = torch.jit.trace(wrapped, example, check_trace=False)
# torch.jit.save(model_torchscript, f"{export_dir}/model_torchscript.pt")

model_pte = aot_export(wrapped, (example,))
aot_save(model_pte, f"{export_dir}/model_pte.pt")



print("starting inference")

# with torch.no_grad():
#     for i, path in enumerate(test_LR_paths):
#         LR = Image.open(path).convert("RGB")
#         LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
#         # LR = LR.half()

#         torch.cuda.synchronize()
#         start_time = time()

#         SR = model(LR)
#         SR = (SR - SR.mean(dim=[2,3],keepdim=True)) / SR.std(dim=[2,3],keepdim=True) \
#              * LR.std(dim=[2,3],keepdim=True) + LR.mean(dim=[2,3],keepdim=True)

#         torch.cuda.synchronize()
#         total_time = time() - start_time
#         print(f"time {total_time} sec")

#         SR = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())
#         SR.save(os.path.join(args.SR_dir, os.path.basename(path)))


# ---- one-line config ----
export_path = f"{export_dir}/model_pte.pt"   # where you saved torch.export
PTE_HAS_POST = True  # set True if your exported model already does the mean/std renorm

# load exported program as a callable module (no .eval())

ep = aot_load(export_path)
model_pte = ep.module().to(device)

print("starting inference (eager vs torch.export)")

with torch.no_grad():
    for i, path in enumerate(test_LR_paths):
        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
        # LR = LR.half(); model.half(); model_pte.half()

        # ----- eager -----
        torch.cuda.synchronize()
        t0 = time()
        SR_eager = model(LR)
        SR_eager = (SR_eager - SR_eager.mean(dim=[2,3], keepdim=True)) / (SR_eager.std(dim=[2,3], keepdim=True) + 1e-6) \
                   * LR.std(dim=[2,3], keepdim=True) + LR.mean(dim=[2,3], keepdim=True)
        torch.cuda.synchronize()
        t_eager = time() - t0

        # ----- torch.export -----
        torch.cuda.synchronize()
        t0 = time()
        SR_pte = model_pte(LR)
        if not PTE_HAS_POST:
            SR_pte = (SR_pte - SR_pte.mean(dim=[2,3], keepdim=True)) / (SR_pte.std(dim=[2,3], keepdim=True) + 1e-6) \
                     * LR.std(dim=[2,3], keepdim=True) + LR.mean(dim=[2,3], keepdim=True)
        torch.cuda.synchronize()
        t_pte = time() - t0

        # ----- simple diff -----
        diff = (SR_eager - SR_pte).abs()
        print(f"[{i+1}/{len(test_LR_paths)}] {os.path.basename(path)} | "
              f"eager {t_eager:.4f}s  export {t_pte:.4f}s  | "
              f"Δmax={diff.max().item():.6f}  Δmean={diff.mean().item():.6f}")

        # ----- save outputs -----
        base = os.path.splitext(os.path.basename(path))[0]
        transforms.ToPILImage()(((SR_eager[0] / 2 + 0.5).clamp(0,1)).cpu()).save(os.path.join(args.SR_dir, f"{base}_eager.png"))
        transforms.ToPILImage()(((SR_pte[0]   / 2 + 0.5).clamp(0,1)).cpu()).save(os.path.join(args.SR_dir, f"{base}_pte.png"))
