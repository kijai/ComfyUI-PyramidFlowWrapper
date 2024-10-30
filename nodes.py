import os
import torch
import folder_paths
import json
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from contextlib import nullcontext
from einops import rearrange
from .pyramid_dit import PyramidDiTForVideoGeneration
import torchvision.transforms as transforms
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "pyramidflow" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("pyramidflow", os.path.join(folder_paths.models_dir, "pyramidflow"))

from .pyramid_dit.mmdit_modules import PyramidDiffusionMMDiT
from .pyramid_dit.flux_modules import PyramidFluxTransformer
from .video_vae.modeling_causal_vae import CausalVideoVAE

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False

class PyramidFlowTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "compile_whole_model": ("BOOLEAN", {"default": False, "tooltip": "Compile the whole model, overrides other block settings"}),
                "single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single_blocks"}),
                "double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile transformer blocks"}),
                "embedders": ("BOOLEAN", {"default": True, "tooltip": "Compile embedders"}),
                "compile_rest": ("BOOLEAN", {"default": True, "tooltip": "Compile the rest of the model (proj and norm out)"}),
            },
        }
    RETURN_TYPES = ("MOCHICOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "MochiWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, compile_whole_model, single_blocks, double_blocks, embedders, compile_rest):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "compile_whole_model": compile_whole_model,
            "single_blocks": single_blocks,
            "double_blocks": double_blocks,
            "embedders": embedders,
            "compile_rest": compile_rest,
        }

        return (compile_args, )

class PyramidFlowVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": (folder_paths.get_filename_list("vae"), {"tooltip": "The name of the checkpoint (model) to load.",}),
                    "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "compile_args": ("MOCHICOMPILEARGS", {"tooltip": "Optional torch.compile arguments",}),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWVAE", )
    RETURN_NAMES = ("pyramidflow_vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "PyramidFlowWrapper"

    def loadmodel(self, vae, precision, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae_path = folder_paths.get_full_path_or_raise("vae", vae)

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        config_path = os.path.join(script_directory, 'configs', 'causal_video_vae_config.json')
        with open(config_path) as f:
            config = json.load(f)

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            vae = CausalVideoVAE.from_config(config, torch_dtype=dtype, interpolate=False)
        vae_sd = load_torch_file(vae_path)
        if is_accelerate_available:
            for name, param in vae.named_parameters():
                set_module_tensor_to_device(vae, name, dtype=dtype, device=device, value=vae_sd[name])
        else:
            vae.load_state_dict(vae_sd)
        del vae_sd
        # Freeze vae
        for parameter in vae.parameters():
            parameter.requires_grad = False
        
        vae.eval().to(device)
        #torch.compile
        if compile_args is not None:
           vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=False, backend=compile_args["backend"])

        return (vae,)
    
class PyramidFlowModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load.",}),
                "precision": (["fp8_e4m3fn","fp8_e4m3fn_fast","fp16", "fp32", "bf16"], {"default": "bf16"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enable sequential cpu offload, saves VRAM but is MUCH slower, do not use unless you have to"}),
            },
            "optional": {
                "compile_args": ("MOCHICOMPILEARGS", {"tooltip": "Optional torch.compile arguments",}),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWMODEL", )
    RETURN_NAMES = ("pyramidflow_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "PyramidFlowWrapper"

    def loadmodel(self, model, precision,enable_sequential_cpu_offload, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        transformer_sd = load_torch_file(model_path)

        for key in transformer_sd:
            if key.startswith("pos_embed."):
                model_name = "pyramid_mmdit"
                continue
        else:
            model_name = "pyramid_flux"
        
        if model_name == "pyramid_flux":
            config_path = os.path.join(script_directory, 'configs', 'miniflux_transformer_config.json')
            with open(config_path) as f:
                config = json.load(f)

            with (init_empty_weights() if is_accelerate_available else nullcontext()):
                transformer = PyramidFluxTransformer.from_config(config)

            if is_accelerate_available:
                logging.info("Using accelerate to load and assign model weights to device...")
                for name, param in transformer.named_parameters():
                    set_module_tensor_to_device(transformer, name, dtype=dtype, device=device, value=transformer_sd[name])
            else:
                transformer.load_state_dict(transformer_sd)
                transformer = transformer.to(dtype)

        elif model_name == "pyramid_mmdit":
            config_path = os.path.join(script_directory, 'configs', 'mmdit_transformer_config.json')
            with open(config_path) as f:
                config = json.load(f)
            transformer = PyramidDiffusionMMDiT.from_config(config)
            params_to_keep = {"pos_embedding"}
            if is_accelerate_available:
                logging.info("Using accelerate to load and assign model weights to device...")
                for name, param in transformer.named_parameters():
                    if not any(keyword in name for keyword in params_to_keep):
                        set_module_tensor_to_device(transformer, name, dtype=dtype, device=device, value=transformer_sd[name])
                    else:
                        set_module_tensor_to_device(transformer, name, dtype=torch.bfloat16, device=device, value=transformer_sd[name])
            else:
                transformer.load_state_dict(transformer_sd)
                if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    for name, param in transformer.named_parameters():
                        if not any(keyword in name for keyword in params_to_keep):
                            param.data = param.data.to(dtype)

        if precision == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, torch.bfloat16)
        
        transformer.to(device)
        #torch.compile
        if compile_args is not None:
            torch._dynamo.config.force_parameter_static_shapes = False
            dynamic = True # because of the stages the compiliation should be dynamic
            if compile_args["compile_whole_model"]:
                transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
            else:
                if compile_args["single_blocks"]:
                    for i, block in enumerate(transformer.single_transformer_blocks):
                        transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
                if compile_args["double_blocks"]:
                    for i, block in enumerate(transformer.transformer_blocks):
                        transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
                if compile_args["embedders"]:
                    transformer.context_embedder = torch.compile(transformer.context_embedder, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
                    transformer.time_text_embed = torch.compile(transformer.time_text_embed, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
                    transformer.x_embedder = torch.compile(transformer.x_embedder, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
                if compile_args["compile_rest"]:    
                    transformer.norm_out.linear = torch.compile(transformer.norm_out.linear, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
                    transformer.proj_out = torch.compile(transformer.proj_out, fullgraph=compile_args["fullgraph"], dynamic=dynamic, backend=compile_args["backend"])
 

        pyramid_model = PyramidDiTForVideoGeneration(transformer, dtype, model_name, device)

        if enable_sequential_cpu_offload:
            pyramid_model.enable_sequential_cpu_offload()

        return (pyramid_model,)

class PyramidFlowSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMIDFLOWMODEL",),
                "prompt_embeds": ("PYRAMIDFLOWPROMPT",),
                "width": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 8}),
                "first_frame_steps": ("STRING", {"default": "10, 10, 10", "tooltip": "Number of steps for each of the 3 stages, for the first frame, no effect when using input_latent"}),
                "video_steps": ("STRING", {"default": "10, 10, 10", "tooltip": "Number of steps for each of the 3 stages, for the video latents"}),
                "temp": ("INT", {"default": 8, "min": 1, "tooltip": "temp=16: 5s, temp=31: 10s"}),
                "guidance_scale": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "The guidance for the first frame"}),
                "video_guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "The guidance for the video latents"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
               
            },
            "optional": {
                "input_latent": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, first_frame_steps, prompt_embeds, seed, height, width, video_steps, temp, guidance_scale, video_guidance_scale, 
               keep_model_loaded, input_latent=None):
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if isinstance(model, dict):
            pyramid_model = model["model"]
            #dtype = model["dtype"]
        else:
            pyramid_model = model

        dtype = pyramid_model.dit.dtype

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        first_frame_steps = [int(num) for num in first_frame_steps.replace(" ", "").split(",")]
        video_steps = [int(num) for num in video_steps.replace(" ", "").split(",")]

        autocast_dtype = dtype if dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16
        autocastcondition = not dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device), dtype=autocast_dtype) if autocastcondition else nullcontext()

        if input_latent is None:
            with autocast_context:
                latents = pyramid_model.generate(
                    prompt_embeds_dict = prompt_embeds,
                    device=device,
                    num_inference_steps=first_frame_steps,
                    video_num_inference_steps=video_steps,
                    height=height,
                    width=width,
                    temp=temp,
                    guidance_scale=guidance_scale,         # The guidance for the first frame
                    video_guidance_scale=video_guidance_scale,   # The guidance for the other video latent
                    output_type="latent",
                )
        else:
            with autocast_context:
                latents = pyramid_model.generate_i2v(
                    prompt_embeds_dict = prompt_embeds,
                    input_image_latent=input_latent,
                    device=device,
                    num_inference_steps=video_steps,
                    height=height,
                    width=width,
                    temp=temp,
                    video_guidance_scale=video_guidance_scale,   # The guidance for the other video latent
                    output_type="latent",
                )

        if not keep_model_loaded and not pyramid_model.sequential_offload_enabled:
            pyramid_model.dit.to(offload_device)      

        return ({"samples": latents},)

#todo: sd3 version
class PyramidFlowTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "positive_prompt": ("STRING", {"default": "hyper quality, Ultra HD, 8K", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror", "multiline": True} ),
            "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWPROMPT",)
    RETURN_NAMES = ("prompt_embeds",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, positive_prompt, negative_prompt, force_offload=True):
        max_lenght = 128
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        clip.cond_stage_model.reset_clip_options()
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.truncation = True
        clip.tokenizer.t5xxl.max_length = max_lenght
        clip.tokenizer.t5xxl.min_length = 1
        clip.tokenizer.clip_l.max_length = 77
        clip.cond_stage_model.t5xxl.return_attention_masks = True
        clip.cond_stage_model.t5xxl.enable_attention_masks = True
        clip.cond_stage_model.t5_attention_mask = True

        clip.cond_stage_model.to(device)#.to(torch.bfloat16)

        #positive
        tokens = clip.tokenizer.t5xxl.tokenize_with_weights(positive_prompt, return_word_ids=False)
        prompt_embeds, _, prompt_attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(tokens)
        tokens = clip.tokenizer.clip_l.tokenize_with_weights(positive_prompt, return_word_ids=False)
        _, pooled_prompt_embeds, = clip.cond_stage_model.clip_l.encode_token_weights(tokens)
        #negative
        tokens = clip.tokenizer.t5xxl.tokenize_with_weights(negative_prompt, return_word_ids=False)
        negative_prompt_embeds, _, negative_prompt_attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(tokens)
        tokens = clip.tokenizer.clip_l.tokenize_with_weights(negative_prompt, return_word_ids=False)
        _, pooled_negative_prompt_embeds, = clip.cond_stage_model.clip_l.encode_token_weights(tokens)

        if force_offload:
            clip.cond_stage_model.to(offload_device)
        
        embeds = {
            "prompt_embeds": prompt_embeds.to(device),
            "attention_mask": prompt_attention_mask["attention_mask"].to(device),
            "pooled_embeds": pooled_prompt_embeds.to(device),
            "negative_prompt_embeds": negative_prompt_embeds.to(device),
            "negative_attention_mask": negative_prompt_attention_mask["attention_mask"].to(device),
            "negative_pooled_embeds": pooled_negative_prompt_embeds.to(device),
        }

        return (embeds, )
class PyramidFlowVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("PYRAMIDFLOWVAE",),
                "image": ("IMAGE",), 
                "enable_tiling": ("BOOLEAN", {"default": False}),           
            },
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, vae, image, enable_tiling):
        mm.soft_empty_cache()

        dtype = vae.dtype
        if enable_tiling:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # For the image latent
        vae_shift_factor = 0.1490
        vae_scale_factor = 1 / 1.8415

        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        input_image_tensor = rearrange(image, 'b h w c -> b c h w')
        input_image_tensor = normalize(input_image_tensor)
        input_image_tensor = input_image_tensor.unsqueeze(2)  # Add temporal dimension t=1
        input_image_tensor = input_image_tensor.to(dtype=dtype, device=device)

        vae.to(device)
        input_image_latent = (vae.encode(input_image_tensor).latent_dist.sample() - vae_shift_factor) * vae_scale_factor  # [b c 1 h w]
        vae.to(offload_device)

        return (input_image_latent,)
    
class PyramidFlowVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("PYRAMIDFLOWVAE",),
                "samples": ("LATENT",),
                "tile_sample_min_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 8}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "enable_tiling": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, vae, samples, tile_sample_min_size, window_size, enable_tiling):
        mm.soft_empty_cache()

        latents = samples["samples"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        if enable_tiling:
            vae.enable_tiling()
        else:
            vae.disable_tiling()

        # For the image latent
        vae_shift_factor = 0.1490
        vae_scale_factor = 1 / 1.8415

        # For the video latent
        vae_video_shift_factor = -0.2343
        vae_video_scale_factor = 1 / 3.0986

        vae.to(device)
        latents = latents.to(vae.dtype)
        if latents.shape[2] == 1:
            latents = (latents / vae_scale_factor) + vae_shift_factor
        else:
            latents[:, :, :1] = (latents[:, :, :1] / vae_scale_factor) + vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / vae_video_scale_factor) + vae_video_shift_factor

        image = vae.decode(latents, temporal_chunk=True, window_size=window_size, tile_sample_min_size=tile_sample_min_size).sample

        vae.to(offload_device)

        image = image.float()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, "B C T H W -> (B T) H W C")
        image = image.cpu().float()

        
        return (image,)
    

NODE_CLASS_MAPPINGS = {
    "PyramidFlowSampler": PyramidFlowSampler,
    "PyramidFlowVAEDecode": PyramidFlowVAEDecode,
    "PyramidFlowTextEncode": PyramidFlowTextEncode,
    "PyramidFlowVAEEncode": PyramidFlowVAEEncode,
    "PyramidFlowTorchCompileSettings": PyramidFlowTorchCompileSettings,
    "PyramidFlowTransformerLoader": PyramidFlowModelLoader,
    "PyramidFlowVAELoader": PyramidFlowVAELoader
   
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadPyramidFlowModel": "(Down)load PyramidFlow Model",
    "PyramidFlowSampler": "PyramidFlow Sampler",
    "PyramidFlowVAEDecode" : "PyramidFlow VAE Decode",
    "PyramidFlowTextEncode": "PyramidFlow Text Encode",
    "PyramidFlowVAEEncode": "PyramidFlow VAE Encode",
    "PyramidFlowTorchCompileSettings": "PyramidFlow Torch Compile Settings",
    "PyramidFlowTransformerLoader": "PyramidFlow Model Loader",
    "PyramidFlowVAELoader": "PyramidFlow VAE Loader"
    }
