import torch, os, glob, copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from model import Net
from time import time

def print_all_layer_precisions(model, model_name="Model"):
    """
    Print every single layer name and its precision by walking through all named modules
    """
    print(f"\n{'='*80}")
    print(f"ALL LAYERS AND THEIR PRECISIONS - {model_name}")
    print(f"{'='*80}")

    total_layers = 0
    fp16_layers = 0
    fp32_layers = 0
    no_param_layers = 0

    # Walk through ALL named modules (every single layer/sublayer)
    for name, module in model.named_modules():
        total_layers += 1

        # Get parameters for this specific module (not including children)
        params = list(module.parameters(recurse=False))

        if len(params) > 0:
            # This module has parameters, check their precision
            first_param = params[0]
            dtype = first_param.dtype

            if dtype == torch.float16:
                precision = "FP16"
                fp16_layers += 1
            elif dtype == torch.float32:
                precision = "FP32"
                fp32_layers += 1
            else:
                precision = str(dtype)

            param_count = sum(p.numel() for p in params)
            print(f"{name:<60} | {type(module).__name__:<25} | {precision:<6} | {param_count:>10,} params")
        else:
            # No parameters in this module
            no_param_layers += 1
            print(f"{name:<60} | {type(module).__name__:<25} | No params")

    print(f"{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total modules examined: {total_layers}")
    print(f"  Modules with FP16 params: {fp16_layers}")
    print(f"  Modules with FP32 params: {fp32_layers}")
    print(f"  Modules with no params: {no_param_layers}")
    print(f"{'='*80}\n")


def print_sequential_layer_precisions(model, model_name="Sequential Model"):
    """
    For Sequential models, print each indexed layer and its precision
    """
    if not isinstance(model, torch.nn.Sequential):
        print(f"Model is not Sequential, it's {type(model)}")
        return

    print(f"\n{'='*80}")
    print(f"SEQUENTIAL LAYERS AND THEIR PRECISIONS - {model_name}")
    print(f"{'='*80}")

    for i, layer in enumerate(model):
        params = list(layer.parameters())

        if len(params) > 0:
            dtype = params[0].dtype
            if dtype == torch.float16:
                precision = "FP16"
            elif dtype == torch.float32:
                precision = "FP32"
            else:
                precision = str(dtype)

            param_count = sum(p.numel() for p in params)
            print(f"[{i:2d}] {type(layer).__name__:<30} | {precision:<6} | {param_count:>10,} params")

            # Also print submodules if they exist
            for name, submodule in layer.named_modules():
                if name != "":  # Skip the layer itself
                    subparams = list(submodule.parameters(recurse=False))
                    if len(subparams) > 0:
                        subdtype = subparams[0].dtype
                        subprecision = "FP16" if subdtype == torch.float16 else "FP32" if subdtype == torch.float32 else str(subdtype)
                        subparam_count = sum(p.numel() for p in subparams)
                        print(f"    └── {name:<40} | {type(submodule).__name__:<20} | {subprecision:<6} | {subparam_count:>8,} params")
        else:
            print(f"[{i:2d}] {type(layer).__name__:<30} | No params")

    print(f"{'='*80}\n")


def print_wrapper_layer_precisions(model, model_name="Wrapped Model"):
    """
    For models that have a 'layers' attribute (like our SafeMixedPrecisionWrapper)
    """
    if not hasattr(model, 'layers'):
        print(f"Model doesn't have 'layers' attribute")
        return

    print(f"\n{'='*80}")
    print(f"WRAPPER LAYERS AND THEIR PRECISIONS - {model_name}")
    print(f"{'='*80}")

    if hasattr(model, 'fp32_indices'):
        print(f"FP32 indices in wrapper: {model.fp32_indices}")
        print()

    for i, layer in enumerate(model.layers):
        params = list(layer.parameters())

        if len(params) > 0:
            dtype = params[0].dtype
            if dtype == torch.float16:
                precision = "FP16"
            elif dtype == torch.float32:
                precision = "FP32"
            else:
                precision = str(dtype)

            param_count = sum(p.numel() for p in params)
            wrapper_note = " (FP32 in wrapper)" if hasattr(model, 'fp32_indices') and i in model.fp32_indices else ""
            print(f"[{i:2d}] {type(layer).__name__:<30} | {precision:<6} | {param_count:>10,} params{wrapper_note}")

            # Print submodules
            for name, submodule in layer.named_modules():
                if name != "":
                    subparams = list(submodule.parameters(recurse=False))
                    if len(subparams) > 0:
                        subdtype = subparams[0].dtype
                        subprecision = "FP16" if subdtype == torch.float16 else "FP32" if subdtype == torch.float32 else str(subdtype)
                        subparam_count = sum(p.numel() for p in subparams)
                        print(f"    └── {name:<40} | {type(submodule).__name__:<20} | {subprecision:<6} | {subparam_count:>8,} params")
        else:
            print(f"[{i:2d}] {type(layer).__name__:<30} | No params")

    print(f"{'='*80}\n")


def inspect_model_completely(model, model_name="Model"):
    """
    Complete model inspection - tries all methods to show layer precisions
    """
    print(f"\n🔍 COMPLETE MODEL INSPECTION: {model_name}")
    print(f"Model type: {type(model)}")

    # Method 1: Print all named modules (most comprehensive)
    print_all_layer_precisions(model, f"{model_name} - All Named Modules")

    # Method 2: If it's Sequential, print indexed layers
    if isinstance(model, torch.nn.Sequential):
        print_sequential_layer_precisions(model, f"{model_name} - Sequential View")

    # Method 3: If it has layers attribute (wrapper), print those
    if hasattr(model, 'layers'):
        print_wrapper_layer_precisions(model, f"{model_name} - Wrapper View")




# Add the mixed precision helper
class MixedPrecisionHelper:
    @staticmethod
    def print_model_precisions(model, title="Model Layer Precisions"):
        """
        Print the precision (dtype) of all layers in the model
        """
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")

        total_params = 0
        fp16_params = 0
        fp32_params = 0

        # Handle sequential model
        if isinstance(model, torch.nn.Sequential):
            for i, layer in enumerate(model):
                layer_params = sum(p.numel() for p in layer.parameters())
                total_params += layer_params

                if layer_params > 0:
                    # Get dtype of first parameter
                    first_param = next(layer.parameters())
                    dtype = first_param.dtype
                    precision = "FP16" if dtype == torch.float16 else "FP32" if dtype == torch.float32 else str(dtype)

                    if dtype == torch.float16:
                        fp16_params += layer_params
                    elif dtype == torch.float32:
                        fp32_params += layer_params

                    print(f"  [{i:2d}] {type(layer).__name__:<20} | {precision:<6} | {layer_params:>10,} params")
                else:
                    print(f"  [{i:2d}] {type(layer).__name__:<20} | No params")

        # Handle our safe wrapper
        elif hasattr(model, 'layers'):
            print("SafeMixedPrecisionWrapper detected:")
            print(f"  FP32 indices: {getattr(model, 'fp32_indices', 'Not set')}")
            print("\nUnderlying layers:")

            for i, layer in enumerate(model.layers):
                layer_params = sum(p.numel() for p in layer.parameters())
                total_params += layer_params

                if layer_params > 0:
                    first_param = next(layer.parameters())
                    dtype = first_param.dtype
                    precision = "FP16" if dtype == torch.float16 else "FP32" if dtype == torch.float32 else str(dtype)

                    if dtype == torch.float16:
                        fp16_params += layer_params
                    elif dtype == torch.float32:
                        fp32_params += layer_params

                    # Show if this index is marked as FP32
                    fp32_marker = " (FP32 idx)" if hasattr(model, 'fp32_indices') and i in model.fp32_indices else ""
                    print(f"  [{i:2d}] {type(layer).__name__:<20} | {precision:<6} | {layer_params:>10,} params{fp32_marker}")
                else:
                    print(f"  [{i:2d}] {type(layer).__name__:<20} | No params")

        # Handle regular model with named modules
        else:
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Only leaf modules
                    module_params = sum(p.numel() for p in module.parameters())
                    if module_params > 0:
                        total_params += module_params
                        first_param = next(module.parameters())
                        dtype = first_param.dtype
                        precision = "FP16" if dtype == torch.float16 else "FP32" if dtype == torch.float32 else str(dtype)

                        if dtype == torch.float16:
                            fp16_params += module_params
                        elif dtype == torch.float32:
                            fp32_params += module_params

                        print(f"  {name:<30} | {precision:<6} | {module_params:>10,} params")

        # Summary
        print(f"{'-'*60}")
        print(f"SUMMARY:")
        print(f"  Total parameters: {total_params:>15,}")
        print(f"  FP16 parameters:  {fp16_params:>15,} ({fp16_params/total_params*100:.1f}%)")
        print(f"  FP32 parameters:  {fp32_params:>15,} ({fp32_params/total_params*100:.1f}%)")
        print(f"  Memory savings:   ~{(fp16_params * 2) / (1024**3):.2f} GB")
        print(f"{'='*60}\n")

    @staticmethod
    def print_layer_details(model, show_shapes=True):
        """
        Print detailed information about each layer including shapes and precision
        """
        print(f"\n{'='*80}")
        print("DETAILED LAYER INFORMATION")
        print(f"{'='*80}")

        def print_module_info(module, name, indent=0):
            indent_str = "  " * indent

            # Get parameter info
            params = list(module.parameters())
            if params:
                dtype = params[0].dtype
                param_count = sum(p.numel() for p in params)
                precision = "FP16" if dtype == torch.float16 else "FP32" if dtype == torch.float32 else str(dtype)

                if show_shapes and params:
                    shapes = [str(list(p.shape)) for p in params]
                    shape_info = f" | Shapes: {', '.join(shapes)}"
                else:
                    shape_info = ""

                print(f"{indent_str}{name:<30} | {type(module).__name__:<20} | {precision:<6} | {param_count:>8,} params{shape_info}")
            else:
                print(f"{indent_str}{name:<30} | {type(module).__name__:<20} | No params")

            # Recursively print children (but limit depth to avoid spam)
            if indent < 2:
                for child_name, child_module in module.named_children():
                    print_module_info(child_module, f"{name}.{child_name}", indent + 1)

        # Handle different model types
        if isinstance(model, torch.nn.Sequential):
            for i, layer in enumerate(model):
                print_module_info(layer, f"[{i}]")
        elif hasattr(model, 'layers'):
            print("SafeMixedPrecisionWrapper:")
            for i, layer in enumerate(model.layers):
                print_module_info(layer, f"layers[{i}]")
        else:
            print_module_info(model, "model")

        print(f"{'='*80}\n")

    @staticmethod
    def convert_to_mixed_precision(model, keep_fp32_layers=None):
        if keep_fp32_layers is None:
            # Critical layers for your diffusion model - MORE CONSERVATIVE NOW
            keep_fp32_layers = [
                '3',              # UpDecoderBlock2D (values getting large)
                '4',              # UpDecoderBlock2D (values very large)
                '5',              # GroupNorm (conv_norm_out)
                '6',              # SiLU (conv_act)
                '7',              # Final Conv2d (conv_out)
                'norm',           # Any other norm layers
                'groupnorm',
                'layernorm',
                'mid_block',      # UNet middle block
                'attention'       # Attention layers if problematic
            ]

        print("Converting to mixed precision...")

        # Convert entire model to FP16
        model = model.half()

        # Keep critical layers in FP32
        fp32_layers = []

        # Handle sequential model indices
        for i, module in enumerate(model):
            if str(i) in keep_fp32_layers:
                module.float()
                fp32_layers.append(f"Sequential[{i}]")
                print(f"  Keeping FP32: Sequential[{i}] ({type(module).__name__})")

        # Handle named modules (for the UNet inside)
        for name, module in model.named_modules():
            should_keep_fp32 = any(pattern.lower() in name.lower()
                                 for pattern in keep_fp32_layers)
            if should_keep_fp32 and not any(f"Sequential[{i}]" in fp32_layers for i in range(len(model))):
                module.float()
                fp32_layers.append(name)
                print(f"  Keeping FP32: {name}")

        print(f"Total layers kept in FP32: {len(fp32_layers)}")
        return model, fp32_layers

    @staticmethod
    def create_safe_mixed_precision_model(model, keep_fp32_layers=None):
        """
        Create a wrapper that explicitly controls precision flow
        to prevent automatic conversions that cause overflow
        """
        if keep_fp32_layers is None:
            keep_fp32_layers = ['3', '4', '5', '6', '7']

        class SafeMixedPrecisionWrapper(torch.nn.Module):
            def __init__(self, sequential_model, fp32_indices):
                super().__init__()
                self.layers = sequential_model
                self.fp32_indices = set(int(idx) for idx in fp32_indices if idx.isdigit())

                print(f"Safe wrapper: FP32 indices = {self.fp32_indices}")

            def forward(self, x):
                # Ensure input precision based on first layer
                if 0 in self.fp32_indices:
                    x = x.float()
                else:
                    x = x.half()

                for i, layer in enumerate(self.layers):
                    # Check if this layer should be FP32
                    layer_needs_fp32 = i in self.fp32_indices

                    # Convert input to appropriate precision for this layer
                    if layer_needs_fp32:
                        x = x.float()
                    else:
                        x = x.half()

                    # Run the layer
                    x = layer(x)

                    # For debugging: check for overflow after each layer
                    if torch.isnan(x).any():
                        print(f"🚨 NaN detected after layer {i}!")
                        return x

                    if torch.isinf(x).any():
                        print(f"🚨 Inf detected after layer {i}!")
                        return x

                return x

        # Convert model to mixed precision first
        model, fp32_layer_names = MixedPrecisionHelper.convert_to_mixed_precision(model, keep_fp32_layers)

        # Extract just the numeric indices for the wrapper
        fp32_indices = [layer.split('[')[1].split(']')[0]
                       for layer in fp32_layer_names
                       if 'Sequential[' in layer]

        # Create safe wrapper
        safe_model = SafeMixedPrecisionWrapper(model, fp32_indices)

        return safe_model, fp32_layer_names

    @staticmethod
    def debug_nan_propagation(model, input_tensor):
        """
        Debug function to track where NaNs first appear in the model
        """
        print("🔍 Debugging NaN propagation through model layers...")

        nan_detected = False
        hooks = []

        def make_debug_hook(layer_name):
            def hook(module, input, output):
                nonlocal nan_detected

                if isinstance(output, tuple):
                    # Handle tuple outputs
                    for i, out in enumerate(output):
                        # Check for overflow values that could become NaN
                        max_val = out.max().item()
                        min_val = out.min().item()

                        if abs(max_val) > 60000 or abs(min_val) > 60000:
                            print(f"⚠️  {layer_name}[{i}]: Values approaching FP16 limit!")
                            print(f"   Range: [{min_val:.1f}, {max_val:.1f}] (FP16 limit: ±65504)")

                        if torch.isnan(out).any():
                            print(f"🚨 {layer_name}[{i}]: FIRST NaN detection!")
                            print(f"   Input has NaN: {torch.isnan(input[0]).any().item() if isinstance(input, tuple) else torch.isnan(input).any().item()}")
                            print(f"   Output NaN count: {torch.isnan(out).sum().item()}")
                            print(f"   Output shape: {out.shape}")
                            nan_detected = True
                            return
                else:
                    # Check for overflow values that could become NaN
                    max_val = output.max().item()
                    min_val = output.min().item()

                    if abs(max_val) > 60000 or abs(min_val) > 60000:
                        print(f"⚠️  {layer_name}: Values approaching FP16 limit!")
                        print(f"   Range: [{min_val:.1f}, {max_val:.1f}] (FP16 limit: ±65504)")
                        print(f"   This layer should be kept in FP32!")

                    if torch.isnan(output).any():
                        if not nan_detected:
                            print(f"🚨 {layer_name}: FIRST NaN detection!")
                            print(f"   Input has NaN: {torch.isnan(input[0]).any().item() if isinstance(input, tuple) else torch.isnan(input).any().item()}")
                            print(f"   Output NaN count: {torch.isnan(output).sum().item()}")
                            print(f"   Output shape: {output.shape}")
                            print(f"   Layer type: {type(module).__name__}")
                            print(f"   Layer precision: {next(module.parameters()).dtype if list(module.parameters()) else 'No params'}")
                            nan_detected = True

                if not nan_detected and abs(max_val) <= 60000:
                    # Only print clean status for layers that aren't overflowing
                    out_to_check = output[0] if isinstance(output, tuple) else output
                    print(f"✅ {layer_name}: Clean (range: [{out_to_check.min().item():.6f}, {out_to_check.max().item():.6f}])")

            return hook

        # Register hooks for each layer in the sequential model
        if hasattr(model, 'layers'):
            # Handle our safe wrapper
            for i, layer in enumerate(model.layers):
                hook = layer.register_forward_hook(make_debug_hook(f"Sequential[{i}]-{type(layer).__name__}"))
                hooks.append(hook)
        else:
            # Handle regular sequential model
            for i, layer in enumerate(model):
                hook = layer.register_forward_hook(make_debug_hook(f"Sequential[{i}]-{type(layer).__name__}"))
                hooks.append(hook)

        # Run model and check for precision conversion issues
        try:
            with torch.no_grad():
                print("\n🔍 Checking precision conversions...")

                output = model(input_tensor)

        except Exception as e:
            print(f"🚨 Exception during forward pass: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup hooks
        for hook in hooks:
            hook.remove()

        if not nan_detected:
            print("✅ No NaNs detected in layer hooks, but check precision conversions above!")

        return nan_detected

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--LR_dir", type=str, default="testset/RealSR/LR")
parser.add_argument("--HR_dir", type=str, default="testset/RealSR/HR")
parser.add_argument("--SR_dir", type=str, default="result/RealSR")

# Mixed precision options
parser.add_argument("--use_mixed_precision", action="store_true",
                   help="Enable mixed precision (recommended)")
parser.add_argument("--precision_mode", type=str, default="safe",
                   choices=["manual", "autocast", "safe"],
                   help="Mixed precision mode: manual, autocast, or safe")
parser.add_argument("--keep_fp32_layers", type=str, nargs='+',
                   default=["3", "4", "5", "6", "7", "norm", "attention"],
                   help="Layer patterns to keep in FP32")
parser.add_argument("--debug_nan", action="store_true",
                   help="Enable detailed NaN debugging (slower)")
parser.add_argument("--stop_on_nan", action="store_true",
                   help="Stop processing when NaN is detected for debugging")

# NEW: Precision printing options
parser.add_argument("--print_precisions", action="store_true",
                   help="Print all layer precisions before inference")
parser.add_argument("--print_detailed", action="store_true",
                   help="Print detailed layer information including shapes")

args = parser.parse_args()

device = torch.device("cuda")

# Load diffusion pipeline components
from diffusers import StableDiffusionPipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

vae = pipe.vae
tokenizer = pipe.tokenizer
unet = pipe.unet
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder

# Load custom decoder
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

# Create model
model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))
model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (args.model_dir, args.epoch), weights_only=False))
model = torch.nn.Sequential(
    model.module,
    *decoder.up_blocks,           # Indices 1-4: UpDecoderBlock2D
    decoder.conv_norm_out,        # Index 5: GroupNorm
    decoder.conv_act,             # Index 6: SiLU
    decoder.conv_out,             # Index 7: Conv2d
).to(device)
model.eval()

# Print precisions BEFORE mixed precision conversion
if args.print_precisions or args.print_detailed:
    MixedPrecisionHelper.print_model_precisions(model, "BEFORE Mixed Precision Conversion")

if args.print_detailed:
    MixedPrecisionHelper.print_layer_details(model)

# Apply mixed precision
if args.use_mixed_precision:
    print(f"\n{'='*50}")
    print("APPLYING MIXED PRECISION")
    print(f"{'='*50}")

    if args.precision_mode == "manual":
        print("Using manual mixed precision...")
        model, fp32_layers = MixedPrecisionHelper.convert_to_mixed_precision(
            model, args.keep_fp32_layers)

        print("\nMixed precision summary:")
        print(f"  FP32 layers: {fp32_layers}")
        print("  All other layers: FP16")

    elif args.precision_mode == "safe":
        print("Using safe mixed precision wrapper...")
        model, fp32_layers = MixedPrecisionHelper.create_safe_mixed_precision_model(
            model, args.keep_fp32_layers)

        print("\nSafe mixed precision summary:")
        print(f"  FP32 layers: {fp32_layers}")
        print("  Explicit precision control: Enabled")

    elif args.precision_mode == "autocast":
        print("Using autocast mixed precision...")
        # Keep model in FP32, autocast will handle precision during forward pass
        model = model.float()
        print("  Model kept in FP32, autocast will handle precision automatically")

    print(f"{'='*50}\n")
else:
    print("Using standard FP32 precision")

# Print precisions AFTER mixed precision conversion
if args.print_precisions or args.print_detailed:
    MixedPrecisionHelper.print_model_precisions(model, "AFTER Mixed Precision Conversion")

if args.print_detailed:
    MixedPrecisionHelper.print_layer_details(model)

# Get test images
test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png")) +
                           glob.glob(os.path.join(args.LR_dir, "*.jpg")) +
                           glob.glob(os.path.join(args.LR_dir, "*.jpeg"))))

os.makedirs(args.SR_dir, exist_ok=True)
print("Starting inference")

inspect_model_completely(model)

with torch.no_grad():
    for i, path in enumerate(test_LR_paths):
        print(f"\nProcessing image {i+1}/{len(test_LR_paths)}: {os.path.basename(path)}")

        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1

        torch.cuda.synchronize()
        start_time = time()

        # Optional NaN debugging
        if args.debug_nan:
            print("\n🔍 Running NaN propagation analysis...")
            nan_found = MixedPrecisionHelper.debug_nan_propagation(model, LR)
            if nan_found and args.stop_on_nan:
                print("Stopping due to NaN detection (--stop_on_nan enabled)")
                break

        # Run inference with appropriate precision handling
        if args.use_mixed_precision and args.precision_mode == "autocast":
            # Automatic mixed precision with autocast
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                SR = model(LR)
        elif args.use_mixed_precision and args.precision_mode == "safe":
            # Safe mixed precision with explicit control
            SR = model(LR.float())  # Safe wrapper handles precision internally
        else:
            # Manual mixed precision or standard FP32
            # Convert input to appropriate precision
            if args.use_mixed_precision and args.precision_mode == "manual":
                LR = LR.half()  # Convert input to FP16 for manual mixed precision

            SR = model(LR)

        # Check for NaN/Inf immediately after model output
        if torch.isnan(SR).any():
            print(f"🚨 FOUND NaN VALUES after model forward pass!")
            print(f"   NaN count: {torch.isnan(SR).sum().item()} / {SR.numel()} total values")
            print(f"   NaN percentage: {torch.isnan(SR).sum().item() / SR.numel() * 100:.2f}%")

            # Replace NaNs with zeros as emergency fix
            print("   Replacing NaNs with zeros...")
            SR = torch.where(torch.isnan(SR), torch.zeros_like(SR), SR)

        if torch.isinf(SR).any():
            print(f"🚨 FOUND Inf VALUES after model forward pass!")
            print(f"   Inf count: {torch.isinf(SR).sum().item()} / {SR.numel()} total values")

            # Replace Infs with reasonable values
            print("   Clamping Inf values...")
            SR = torch.where(torch.isinf(SR), torch.sign(SR) * 10.0, SR)

        # Apply your normalization (always in FP32 for stability)
        SR = SR.float()  # Ensure FP32 for normalization
        LR = LR.float()  # Ensure FP32 for normalization

        # Check for NaN in normalization inputs
        if torch.isnan(SR).any() or torch.isnan(LR).any():
            print("🚨 NaN detected before normalization - skipping normalization!")
            # Skip normalization if we have NaNs
        else:
            # Safe normalization with additional checks
            sr_mean = SR.mean(dim=[2,3], keepdim=True)
            sr_std = SR.std(dim=[2,3], keepdim=True)
            lr_mean = LR.mean(dim=[2,3], keepdim=True)
            lr_std = LR.std(dim=[2,3], keepdim=True)

            # Check if std is too small (would cause division issues)
            if sr_std.min() < 1e-6:
                print("🚨 SR std too small for normalization, skipping...")
            else:
                SR = (SR - sr_mean) / sr_std * lr_std + lr_mean

                # Final NaN check after normalization
                if torch.isnan(SR).any():
                    print("🚨 NaN appeared during normalization!")
                    SR = torch.where(torch.isnan(SR), torch.zeros_like(SR), SR)

        torch.cuda.synchronize()
        total_time = time() - start_time
        print(f"Processing time: {total_time:.4f} seconds")

        # Better black image detection
        def detect_black_image(tensor, threshold_std=0.01, threshold_range=0.05):
            """
            Detect black/corrupted images using multiple criteria
            """
            # Convert to proper range [0, 1] for analysis
            img_normalized = (tensor / 2 + 0.5).clamp(0, 1)

            mean_val = img_normalized.mean().item()
            std_val = img_normalized.std().item()
            min_val = img_normalized.min().item()
            max_val = img_normalized.max().item()
            value_range = max_val - min_val

            # Check for various black image indicators
            is_all_zero = (max_val < 0.001)
            is_all_same = (std_val < threshold_std)
            is_very_dark = (mean_val < 0.05)
            is_no_range = (value_range < threshold_range)
            is_has_nan = torch.isnan(tensor).any().item()
            is_has_inf = torch.isinf(tensor).any().item()

            problems = []
            if is_all_zero: problems.append("all zeros")
            if is_all_same: problems.append("no variation")
            if is_very_dark: problems.append("very dark")
            if is_no_range: problems.append("no dynamic range")
            if is_has_nan: problems.append("contains NaN")
            if is_has_inf: problems.append("contains Inf")

            is_black = any([is_all_zero, is_all_same, is_no_range, is_has_nan, is_has_inf])

            return {
                'is_black': is_black,
                'problems': problems,
                'stats': {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'range': value_range
                }
            }

        # Analyze output
        detection = detect_black_image(SR)
        stats = detection['stats']

        print(f"Output analysis:")
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}] (span: {stats['range']:.6f})")

        if detection['is_black']:
            print(f"🚨 BLACK IMAGE DETECTED! Issues: {', '.join(detection['problems'])}")

            # Additional debugging info
            if torch.isnan(SR).any():
                print(f"   NaN count: {torch.isnan(SR).sum().item()}")
            if torch.isinf(SR).any():
                print(f"   Inf count: {torch.isinf(SR).sum().item()}")

            # Save debug info
            debug_info = {
                'filename': os.path.basename(path),
                'problems': detection['problems'],
                'stats': stats,
                'model_precision': 'mixed' if args.use_mixed_precision else 'fp32'
            }

            # You could save this to a log file
            print(f"   Debug info: {debug_info}")
        else:
            print("✅ Output looks normal")

        # Save result
        SR_pil = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())
        SR_pil.save(os.path.join(args.SR_dir, os.path.basename(path)))

print("Inference completed!")

# Print usage examples
print(f"\n{'='*60}")
print("USAGE EXAMPLES:")
print(f"{'='*60}")
print("# Standard FP32 (safe but slow):")
print("python your_script.py")
print()
print("# Safe mixed precision (recommended for overflow issues):")
print("python your_script.py --use_mixed_precision --precision_mode safe")
print()
print("# Print layer precisions:")
print("python your_script.py --print_precisions")
print()
print("# Print detailed layer info with shapes:")
print("python your_script.py --print_detailed")
print()
print("# Mixed precision with precision debugging:")
print("python your_script.py --use_mixed_precision --print_precisions")
print()
print("# Manual mixed precision:")
print("python your_script.py --use_mixed_precision --precision_mode manual")
print()
print("# Autocast mixed precision (automatic):")
print("python your_script.py --use_mixed_precision --precision_mode autocast")
print()
print("# Custom FP32 layers:")
print("python your_script.py --use_mixed_precision --keep_fp32_layers 5 6 7 norm")
print()
