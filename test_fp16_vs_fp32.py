import torch, os, glob, copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from model import Net
from time import time

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--LR_dir", type=str, default="testset/RealSR/LR")
parser.add_argument("--HR_dir", type=str, default="testset/RealSR/HR")
parser.add_argument("--SR_dir", type=str, default="result/RealSR")
parser.add_argument("--compare_precision", action="store_true", help="Enable detailed layer-by-layer precision comparison")
args = parser.parse_args()

device = torch.device("cuda")

def initialize_model(use_fp16=False):
    """Initialize model from scratch with specified precision"""
    # Load diffusers pipeline components
    from diffusers import StableDiffusionPipeline
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    unet = pipe.unet

    # Create decoder
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

    # Create main model
    model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))
    model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (args.model_dir, args.epoch), weights_only=False))

    # Create the sequential model like in the original code
    final_model = torch.nn.Sequential(
        model.module,
        *decoder.up_blocks,
        decoder.conv_norm_out,
        decoder.conv_act,
        decoder.conv_out,
    ).to(device)

    final_model.eval()

    # Convert to half precision if requested
    if use_fp16:
        final_model.half()

    return final_model

# Initialize both models
print("Initializing FP32 model...")
model_fp32 = initialize_model(use_fp16=False)

print("Initializing FP16 model...")
model_fp16 = initialize_model(use_fp16=True)

def analyze_tensor(tensor, name, precision="fp32"):
    """Analyze tensor statistics"""
    if tensor.numel() == 0:
        print(f"  {name} ({precision}): Empty tensor")
        return

    stats = {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'zeros': (tensor == 0).sum().item(),
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype)
    }

    print(f"  {name} ({precision}): mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
          f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
          f"nan={stats['has_nan']}, inf={stats['has_inf']}, zeros={stats['zeros']}, "
          f"shape={stats['shape']}, dtype={stats['dtype']}")

    return stats

def compare_tensors(tensor1, tensor2, name):
    """Compare two tensors and report differences"""
    # Convert fp16 tensor to fp32 for comparison
    if tensor2.dtype == torch.float16:
        tensor2_fp32 = tensor2.float()
    else:
        tensor2_fp32 = tensor2

    if tensor1.shape != tensor2_fp32.shape:
        print(f"  ❌ {name}: Shape mismatch! fp32={tensor1.shape}, fp16={tensor2_fp32.shape}")
        return

    # Calculate differences
    abs_diff = torch.abs(tensor1 - tensor2_fp32)
    rel_diff = abs_diff / (torch.abs(tensor1) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # Check for significant differences
    significant_diff = max_abs_diff > 1e-2 or mean_abs_diff > 1e-3

    status = "❌" if significant_diff else "✅"
    print(f"  {status} {name}: max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}, "
          f"max_rel_diff={max_rel_diff:.6f}, mean_rel_diff={mean_rel_diff:.6f}")

    return {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'significant_diff': significant_diff
    }

class DebugHook:
    """Hook to capture intermediate layer outputs"""
    def __init__(self, name, store_dict):
        self.name = name
        self.store_dict = store_dict

    def __call__(self, module, input, output):
        # Store output
        if isinstance(output, torch.Tensor):
            self.store_dict[self.name] = output.detach().clone()
        elif isinstance(output, tuple):
            self.store_dict[self.name] = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output)

def register_hooks(model, prefix=""):
    """Register forward hooks on all modules"""
    activations = {}
    hooks = []

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            # Use consistent naming without prefix to ensure matching keys
            hook = module.register_forward_hook(DebugHook(name, activations))
            hooks.append(hook)

    return activations, hooks

def remove_hooks(hooks):
    """Remove all registered hooks"""
    for hook in hooks:
        hook.remove()

# Test paths
test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png")) +
                           glob.glob(os.path.join(args.LR_dir, "*.jpg")) +
                           glob.glob(os.path.join(args.LR_dir, "*.jpeg"))))
test_HR_paths = list(sorted(glob.glob(os.path.join(args.HR_dir, "*.png")) +
                           glob.glob(os.path.join(args.HR_dir, "*.jpg")) +
                           glob.glob(os.path.join(args.HR_dir, "*.jpeg"))))

os.makedirs(args.SR_dir, exist_ok=True)
os.makedirs(f"{args.SR_dir}_fp32", exist_ok=True)
os.makedirs(f"{args.SR_dir}_fp16", exist_ok=True)

print("Starting inference with FP32 vs FP16 comparison")

with torch.no_grad():
    for i, path in enumerate(test_LR_paths[:3]):  # Limit to first 3 images for debugging
        print(f"\n=== Processing image {i+1}: {os.path.basename(path)} ===")

        LR = Image.open(path).convert("RGB")
        LR_fp32 = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
        LR_fp16 = LR_fp32.half()

        print(f"Input shape: {LR_fp32.shape}")
        analyze_tensor(LR_fp32, "Input", "fp32")
        analyze_tensor(LR_fp16, "Input", "fp16")

        # Register hooks for detailed debugging if requested
        if args.compare_precision:
            activations_fp32, hooks_fp32 = register_hooks(model_fp32)
            activations_fp16, hooks_fp16 = register_hooks(model_fp16)

        # Forward pass
        torch.cuda.synchronize()
        start_time = time()

        SR_fp32 = model_fp32(LR_fp32)

        torch.cuda.synchronize()
        fp32_time = time() - start_time

        torch.cuda.synchronize()
        start_time = time()

        SR_fp16 = model_fp16(LR_fp16)

        torch.cuda.synchronize()
        fp16_time = time() - start_time

        print(f"Timing - FP32: {fp32_time:.4f}s, FP16: {fp16_time:.4f}s, Speedup: {fp32_time/fp16_time:.2f}x")

        # Analyze raw outputs
        print("\n--- Raw Model Outputs ---")
        analyze_tensor(SR_fp32, "Raw_Output", "fp32")
        analyze_tensor(SR_fp16, "Raw_Output", "fp16")
        compare_tensors(SR_fp32, SR_fp16, "Raw_Output")

        # Apply normalization
        SR_fp32_norm = (SR_fp32 - SR_fp32.mean(dim=[2,3],keepdim=True)) / SR_fp32.std(dim=[2,3],keepdim=True) \
                       * LR_fp32.std(dim=[2,3],keepdim=True) + LR_fp32.mean(dim=[2,3],keepdim=True)

        SR_fp16_norm = (SR_fp16 - SR_fp16.mean(dim=[2,3],keepdim=True)) / SR_fp16.std(dim=[2,3],keepdim=True) \
                       * LR_fp16.std(dim=[2,3],keepdim=True) + LR_fp16.mean(dim=[2,3],keepdim=True)

        print("\n--- After Normalization ---")
        analyze_tensor(SR_fp32_norm, "Normalized_Output", "fp32")
        analyze_tensor(SR_fp16_norm, "Normalized_Output", "fp16")
        compare_tensors(SR_fp32_norm, SR_fp16_norm, "Normalized_Output")

        # Detailed layer analysis if enabled
        if args.compare_precision:
            print("\n--- Layer-by-Layer Analysis ---")

            # Debug: Check what keys we captured
            print(f"FP32 captured {len(activations_fp32)} layers: {list(activations_fp32.keys())[:5]}...")
            print(f"FP16 captured {len(activations_fp16)} layers: {list(activations_fp16.keys())[:5]}...")

            # Compare activations
            common_keys = set(activations_fp32.keys()) & set(activations_fp16.keys())
            print(f"Common keys: {len(common_keys)}")

            if len(common_keys) == 0:
                print("🚨 No common keys found! Debugging hook registration...")
                print("FP32 keys sample:", list(activations_fp32.keys())[:10])
                print("FP16 keys sample:", list(activations_fp16.keys())[:10])

            significant_diffs = []
            nan_layers = []

            for key in sorted(common_keys):
                act_fp32 = activations_fp32[key]
                act_fp16 = activations_fp16[key]

                if isinstance(act_fp32, torch.Tensor) and isinstance(act_fp16, torch.Tensor):
                    # Check for NaN/Inf in FP16 activations
                    has_nan = torch.isnan(act_fp16).any().item()
                    has_inf = torch.isinf(act_fp16).any().item()

                    if has_nan or has_inf:
                        nan_layers.append((key, has_nan, has_inf))
                        print(f"  🚨 {key}: FP16 has NaN={has_nan}, Inf={has_inf}")

                    diff_stats = compare_tensors(act_fp32, act_fp16, key)
                    if diff_stats and diff_stats['significant_diff']:
                        significant_diffs.append((key, diff_stats))

            if nan_layers:
                print(f"\n🚨 Found {len(nan_layers)} layers with NaN/Inf in FP16:")
                for key, has_nan, has_inf in nan_layers:
                    print(f"  {key}: NaN={has_nan}, Inf={has_inf}")
                print("\n💡 First NaN layer is likely the root cause!")

            if significant_diffs:
                print(f"\n🚨 Found {len(significant_diffs)} layers with significant differences:")
                for key, stats in significant_diffs[:10]:  # Show top 10
                    print(f"  {key}: max_abs_diff={stats['max_abs_diff']:.6f}")

            if not nan_layers and not significant_diffs and len(common_keys) > 0:
                print("\n✅ No significant differences found in layer outputs")

            # Clean up hooks
            remove_hooks(hooks_fp32)
            remove_hooks(hooks_fp16)

        # Save results
        SR_fp32_final = transforms.ToPILImage()((SR_fp32_norm[0] / 2 + 0.5).clamp(0, 1).cpu())
        SR_fp16_final = transforms.ToPILImage()((SR_fp16_norm[0] / 2 + 0.5).clamp(0, 1).cpu())

        base_name = os.path.basename(path)
        SR_fp32_final.save(os.path.join(f"{args.SR_dir}_fp32", base_name))
        SR_fp16_final.save(os.path.join(f"{args.SR_dir}_fp16", base_name))

        # Check final image statistics
        sr_fp32_array = np.array(SR_fp32_final)
        sr_fp16_array = np.array(SR_fp16_final)

        print(f"\n--- Final Image Statistics ---")
        print(f"FP32 image - mean: {sr_fp32_array.mean():.2f}, std: {sr_fp32_array.std():.2f}, "
              f"min: {sr_fp32_array.min()}, max: {sr_fp32_array.max()}")
        print(f"FP16 image - mean: {sr_fp16_array.mean():.2f}, std: {sr_fp16_array.std():.2f}, "
              f"min: {sr_fp16_array.min()}, max: {sr_fp16_array.max()}")

        # Check if FP16 image is mostly black
        black_pixels_fp16 = (sr_fp16_array.sum(axis=2) < 10).sum()
        total_pixels = sr_fp16_array.shape[0] * sr_fp16_array.shape[1]
        black_ratio_fp16 = black_pixels_fp16 / total_pixels

        if black_ratio_fp16 > 0.9:
            print(f"🚨 WARNING: FP16 image is {black_ratio_fp16*100:.1f}% black pixels!")
            print(f"💡 This is likely due to NaN values in the FP16 forward pass")
        else:
            print(f"✅ FP16 image looks normal ({black_ratio_fp16*100:.1f}% black pixels)")

        print(f"Images saved to {args.SR_dir}_fp32 and {args.SR_dir}_fp16")



print("\n=== Analysis Complete ===")
print(f"Results saved in:")
print(f"  FP32: {args.SR_dir}_fp32/")
print(f"  FP16: {args.SR_dir}_fp16/")
print(f"\nUse --compare_precision flag for detailed layer-by-layer analysis")