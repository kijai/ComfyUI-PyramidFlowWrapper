import copy
import math
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
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (
                ["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "compile_whole_model": (
                "BOOLEAN", {"default": False, "tooltip": "Compile the whole model, overrides other block settings"}),
                "single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single_blocks"}),
                "double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile transformer blocks"}),
                "embedders": ("BOOLEAN", {"default": True, "tooltip": "Compile embedders"}),
                "compile_rest": (
                "BOOLEAN", {"default": True, "tooltip": "Compile the rest of the model (proj and norm out)"}),
            },
        }

    RETURN_TYPES = ("MOCHICOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "MochiWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, compile_whole_model, single_blocks, double_blocks, embedders,
                  compile_rest):
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

        return (compile_args,)


class PyramidFlowVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": (
                folder_paths.get_filename_list("vae"), {"tooltip": "The name of the checkpoint (model) to load.", }),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "compile_args": ("MOCHICOMPILEARGS", {"tooltip": "Optional torch.compile arguments", }),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWVAE",)
    RETURN_NAMES = ("pyramidflow_vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "PyramidFlowWrapper"

    def loadmodel(self, vae, precision, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae_path = folder_paths.get_full_path_or_raise("vae", vae)

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16,
                 "fp16": torch.float16, "fp32": torch.float32}[precision]

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
        # torch.compile
        if compile_args is not None:
            vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=False,
                                backend=compile_args["backend"])

        return (vae,)


class PyramidFlowModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"),
                          {"tooltip": "The name of the checkpoint (model) to load.", }),
                "precision": (["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp16", "fp32", "bf16"], {"default": "bf16"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False,
                                                              "tooltip": "Enable sequential cpu offload, saves VRAM but is MUCH slower, do not use unless you have to"}),
            },
            "optional": {
                "compile_args": ("MOCHICOMPILEARGS", {"tooltip": "Optional torch.compile arguments", }),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWMODEL",)
    RETURN_NAMES = ("pyramidflow_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "PyramidFlowWrapper"

    def loadmodel(self, model, precision, enable_sequential_cpu_offload, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16,
                 "fp16": torch.float16, "fp32": torch.float32}[precision]

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
                    set_module_tensor_to_device(transformer, name, dtype=dtype, device=device,
                                                value=transformer_sd[name])
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
                        set_module_tensor_to_device(transformer, name, dtype=dtype, device=device,
                                                    value=transformer_sd[name])
                    else:
                        set_module_tensor_to_device(transformer, name, dtype=torch.bfloat16, device=device,
                                                    value=transformer_sd[name])
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
        # torch.compile
        if compile_args is not None:
            torch._dynamo.config.force_parameter_static_shapes = False
            dynamic = True  # because of the stages the compiliation should be dynamic
            if compile_args["compile_whole_model"]:
                transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=dynamic,
                                            backend=compile_args["backend"])
            else:
                if compile_args["single_blocks"]:
                    for i, block in enumerate(transformer.single_transformer_blocks):
                        transformer.single_transformer_blocks[i] = torch.compile(block,
                                                                                 fullgraph=compile_args["fullgraph"],
                                                                                 dynamic=dynamic,
                                                                                 backend=compile_args["backend"])
                if compile_args["double_blocks"]:
                    for i, block in enumerate(transformer.transformer_blocks):
                        transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"],
                                                                          dynamic=dynamic,
                                                                          backend=compile_args["backend"])
                if compile_args["embedders"]:
                    transformer.context_embedder = torch.compile(transformer.context_embedder,
                                                                 fullgraph=compile_args["fullgraph"], dynamic=dynamic,
                                                                 backend=compile_args["backend"])
                    transformer.time_text_embed = torch.compile(transformer.time_text_embed,
                                                                fullgraph=compile_args["fullgraph"], dynamic=dynamic,
                                                                backend=compile_args["backend"])
                    transformer.x_embedder = torch.compile(transformer.x_embedder, fullgraph=compile_args["fullgraph"],
                                                           dynamic=dynamic, backend=compile_args["backend"])
                if compile_args["compile_rest"]:
                    transformer.norm_out.linear = torch.compile(transformer.norm_out.linear,
                                                                fullgraph=compile_args["fullgraph"], dynamic=dynamic,
                                                                backend=compile_args["backend"])
                    transformer.proj_out = torch.compile(transformer.proj_out, fullgraph=compile_args["fullgraph"],
                                                         dynamic=dynamic, backend=compile_args["backend"])

        pyramid_model = PyramidDiTForVideoGeneration(transformer, dtype, model_name, device)

        if enable_sequential_cpu_offload:
            pyramid_model.enable_sequential_cpu_offload()

        return (pyramid_model,)


class PyramidFlowSlidingContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_temp_per_batch": ("INT", {"default": 16, "min": 1, "max": 31, "step": 1,
                                               "tooltip": "Maximum temp (number of frames) per batch"}),
                "use_same_seed": ("BOOLEAN", {"default": False, "tooltip": "Use the same seed for each batch"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Seed for random number generator"}),
                "skip_last_x_intermediate": ("INT", {"default": 0, "min": 0, "max": 31, "step": 1,
                                                     "tooltip": "Number of last latents to skip in each batch except the last"}),
            },
        }

    RETURN_TYPES = ("SLIDINGCONTEXTOPTIONS",)
    RETURN_NAMES = ("sliding_context_options",)
    FUNCTION = "get_options"
    CATEGORY = "PyramidFlowWrapper"
    DESCRIPTION = "Provides options for sliding context batching in the PyramidFlowSampler node."

    def get_options(self, max_temp_per_batch, use_same_seed, seed, skip_last_x_intermediate):
        options = {
            "max_temp_per_batch": max_temp_per_batch,
            "use_same_seed": use_same_seed,
            "seed": seed,
            "skip_last_x_intermediate": skip_last_x_intermediate,
        }
        return (options,)


class PyramidFlowSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMIDFLOWMODEL",),
                "prompt_embeds": ("PYRAMIDFLOWPROMPT",),
                "width": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 8}),
                "first_frame_steps": ("STRING", {"default": "10, 10, 10",
                                                 "tooltip": "Number of steps for each of the 3 stages, for the first frame, no effect when using input_latent"}),
                "video_steps": ("STRING", {"default": "10, 10, 10",
                                           "tooltip": "Number of steps for each of the 3 stages, for the video latents"}),
                "temp": ("INT", {"default": 16, "min": 1,
                                 "tooltip": "Total number of frames (temp) to generate. temp=16: 5s, temp=31: 10s"}),
                "guidance_scale": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 30.0, "step": 0.01,
                                             "tooltip": "The guidance for the first frame"}),
                "video_guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01,
                                                   "tooltip": "The guidance for the video latents"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_latent": ("LATENT",),
                "sliding_context_options": ("SLIDINGCONTEXTOPTIONS",),
                "vae": ("PYRAMIDFLOWVAE",),  # Optional VAE input
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, first_frame_steps, prompt_embeds, seed, height, width, video_steps, temp, guidance_scale,
               video_guidance_scale,
               keep_model_loaded, input_latent=None, sliding_context_options=None, vae=None):
        images = None
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if isinstance(model, dict):
            pyramid_model = model["model"]
        else:
            pyramid_model = model

        dtype = pyramid_model.dit.dtype

        from .latent_preview import prepare_callback
        callback = prepare_callback(model, temp)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        first_frame_steps = [int(num) for num in first_frame_steps.replace(" ", "").split(",")]
        video_steps = [int(num) for num in video_steps.replace(" ", "").split(",")]

        autocast_dtype = dtype if dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16
        autocastcondition = not dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device),
                                          dtype=autocast_dtype) if autocastcondition else nullcontext()

        if sliding_context_options is not None:
            # Sliding context batching enabled
            max_temp_per_batch = sliding_context_options.get("max_temp_per_batch", temp)
            use_same_seed = sliding_context_options.get("use_same_seed", False)
            batch_seed = sliding_context_options.get("seed", seed)
            skip_last_x_intermediate = sliding_context_options.get("skip_last_x_intermediate", 0)
            temp_per_batch = max_temp_per_batch
            num_batches = math.ceil(temp / temp_per_batch)

            total_latents = []
            images = []
            if input_latent is not None:
                current_input_latent = {"samples":input_latent.get("samples")}
            else:
                current_input_latent = None

            for batch_idx in range(num_batches):
                current_temp = min(temp_per_batch, temp - batch_idx * temp_per_batch)
                if use_same_seed:
                    torch.manual_seed(batch_seed)
                    torch.cuda.manual_seed(batch_seed)
                else:
                    # Increment seed for each batch to ensure different randomness
                    batch_seed = seed + batch_idx
                    torch.manual_seed(batch_seed)
                    torch.cuda.manual_seed(batch_seed)

                if batch_idx == 0 and current_input_latent is None:
                    # First batch without input latent
                    with autocast_context:
                        batch_latents = pyramid_model.generate(
                            prompt_embeds_dict=copy.deepcopy(prompt_embeds),
                            device=device,
                            num_inference_steps=first_frame_steps,
                            video_num_inference_steps=video_steps,
                            height=height,
                            width=width,
                            temp=current_temp,
                            guidance_scale=guidance_scale,
                            video_guidance_scale=video_guidance_scale,
                            callback=callback,
                        )
                else:
                    # Subsequent batches or first batch with input latent
                    with autocast_context:
                        batch_latents = pyramid_model.generate_i2v(
                            prompt_embeds_dict=copy.deepcopy(prompt_embeds),
                            input_image_latent=current_input_latent["samples"],
                            device=device,
                            num_inference_steps=video_steps,
                            height=height,
                            width=width,
                            temp=current_temp,
                            video_guidance_scale=video_guidance_scale,
                            callback=callback,
                        )

                # Determine whether to skip last X frames
                is_last_batch = (batch_idx == num_batches - 1)
                skip_frames = skip_last_x_intermediate if not is_last_batch else 0

                # Decode, optionally skip, and re-encode if skip_frames > 0 and vae is provided
                if skip_frames > 0 and not is_last_batch:
                    current_input_latent, decoded = self.dec_enc(vae, batch_latents, skip=skip_frames,
                                                                 is_last=is_last_batch)
                else:
                    current_input_latent, decoded = self.dec_enc(vae, batch_latents, skip=0, is_last=is_last_batch)
                #     # Use the last latent frame as the next input
                total_latents.append(batch_latents.detach())
                images.append(decoded)

            latents = torch.cat(total_latents, dim=2)
            images = torch.cat(images, dim=0)

        else:
            # No sliding context batching
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            if input_latent is None:
                with autocast_context:
                    latents = pyramid_model.generate(
                        prompt_embeds_dict=prompt_embeds,
                        device=device,
                        num_inference_steps=first_frame_steps,
                        video_num_inference_steps=video_steps,
                        height=height,
                        width=width,
                        temp=temp,
                        guidance_scale=guidance_scale,
                        video_guidance_scale=video_guidance_scale,
                        callback=callback,
                    )
            else:
                with autocast_context:
                    latents = pyramid_model.generate_i2v(
                        prompt_embeds_dict=prompt_embeds,
                        input_image_latent=input_latent["samples"],
                        device=device,
                        num_inference_steps=video_steps,
                        height=height,
                        width=width,
                        temp=temp,
                        video_guidance_scale=video_guidance_scale,
                        callback=callback,
                    )

        if not keep_model_loaded and not pyramid_model.sequential_offload_enabled:
            pyramid_model.dit.to(offload_device)

        return ({"samples": latents}, images,)

    def dec_enc(self, vae, samples, skip=0, is_last=False):

        decoder = PyramidFlowVAEDecode()
        decoded = decoder.sample(vae, {"samples": samples}, 256, 2, True)[0]

        if skip > 0 and not is_last:
            decoded = decoded[:-skip:, :, :]
        encoder = PyramidFlowVAEEncode()
        encoded_last = encoder.sample(vae, decoded[-1].unsqueeze(0), enable_tiling=True)[0]
        return encoded_last, decoded


# todo: sd3 version
class PyramidFlowTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "positive_prompt": ("STRING", {"default": "hyper quality, Ultra HD, 8K", "multiline": True}),
            "negative_prompt": ("STRING", {
                "default": "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
                "multiline": True}),
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

        clip.cond_stage_model.to(device)  # .to(torch.bfloat16)

        # positive
        tokens = clip.tokenizer.t5xxl.tokenize_with_weights(positive_prompt, return_word_ids=False)
        prompt_embeds, _, prompt_attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(tokens)
        tokens = clip.tokenizer.clip_l.tokenize_with_weights(positive_prompt, return_word_ids=False)
        _, pooled_prompt_embeds, = clip.cond_stage_model.clip_l.encode_token_weights(tokens)
        # negative
        tokens = clip.tokenizer.t5xxl.tokenize_with_weights(negative_prompt, return_word_ids=False)
        negative_prompt_embeds, _, negative_prompt_attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(
            tokens)
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

        return (embeds,)


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

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, vae, image, enable_tiling):
        B, H, W, C = image.shape
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
        input_image_tensor = normalize(input_image_tensor).unsqueeze(0)
        input_image_tensor = rearrange(input_image_tensor, 'b t c h w -> b c t h w', t=B)
        # input_image_tensor = input_image_tensor.unsqueeze(2)  # Add temporal dimension t=1
        input_image_tensor = input_image_tensor.to(dtype=dtype, device=device)

        vae.to(device)
        input_image_latent = (vae.encode(
            input_image_tensor).latent_dist.sample() - vae_shift_factor) * vae_scale_factor  # [b c 1 h w]
        vae.to(offload_device)

        return ({"samples": input_image_latent},)


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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
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

        image = vae.decode(latents, temporal_chunk=True, window_size=window_size,
                           tile_sample_min_size=tile_sample_min_size).sample

        vae.to(offload_device)

        image = image.float()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, "B C T H W -> (B T) H W C")
        image = image.cpu().float()

        return (image,)


class PyramidFlowLatentPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # "min_val": ("FLOAT", {"default": -0.15, "min": -1.0, "max": 0.0, "step": 0.001}),
                # "max_val": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "latent_rgb_factors",)
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, samples):  # , seed, min_val, max_val):
        mm.soft_empty_cache()

        latents = samples["samples"].clone()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # For the image latent
        vae_shift_factor = 0.1490
        vae_scale_factor = 1 / 1.8415

        # For the video latent
        vae_video_shift_factor = -0.2343
        vae_video_scale_factor = 1 / 3.0986

        if latents.shape[2] == 1:
            latents = (latents / vae_scale_factor) + vae_shift_factor
        else:
            latents[:, :, :1] = (latents[:, :, :1] / vae_scale_factor) + vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / vae_video_scale_factor) + vae_video_shift_factor

        latent_rgb_factors = [[0.05389399697934166, 0.025018778505575393, -0.009193515248318657],
                              [0.02318250640590553, -0.026987363837713156, 0.040172639061236956],
                              [0.046035451343323666, -0.02039565868920197, 0.01275569344290342],
                              [-0.015559161155025095, 0.051403973219861246, 0.03179031307996347],
                              [-0.02766167769640129, 0.03749545161530447, 0.003335141009473408],
                              [0.05824598730479011, 0.021744367381243884, -0.01578925627951616],
                              [0.05260929401500947, 0.0560165014956886, -0.027477296572565126],
                              [0.018513891242931686, 0.041961785217662514, 0.004490763489747966],
                              [0.024063060899760215, 0.065082853069653, 0.044343437673514896],
                              [0.05250992323006226, 0.04361117432588933, 0.01030076055524387],
                              [0.0038921710021782366, -0.025299228133723792, 0.019370764014574535],
                              [-0.00011950534333568519, 0.06549370069727675, -0.03436712163379723],
                              [-0.026020578032683626, -0.013341758571090847, -0.009119046570271953],
                              [0.024412451175602937, 0.030135064560817174, -0.008355486384198006],
                              [0.04002209845752687, -0.017341304390739463, 0.02818338690302971],
                              [-0.032575108695213684, -0.009588338926775117, -0.03077312160940468]]

        # import random
        # random.seed(seed)
        # latent_rgb_factors = [[random.uniform(min_val, max_val) for _ in range(3)] for _ in range(16)]
        out_factors = latent_rgb_factors
        print(latent_rgb_factors)

        latent_rgb_factors_bias = [0, 0, 0]

        latent_rgb_factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0,
                                                                                                                    1)
        latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        print("latent_rgb_factors", latent_rgb_factors.shape)

        latent_images = []
        for t in range(latents.shape[2]):
            latent = latents[:, :, t, :, :]
            latent = latent[0].permute(1, 2, 0)
            latent_image = torch.nn.functional.linear(
                latent,
                latent_rgb_factors,
                bias=latent_rgb_factors_bias
            )
            latent_images.append(latent_image)
        latent_images = torch.stack(latent_images, dim=0)
        print("latent_images", latent_images.shape)
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)

        return (latent_images.float().cpu(), out_factors)


NODE_CLASS_MAPPINGS = {
    "PyramidFlowSampler": PyramidFlowSampler,
    "PyramidFlowVAEDecode": PyramidFlowVAEDecode,
    "PyramidFlowTextEncode": PyramidFlowTextEncode,
    "PyramidFlowVAEEncode": PyramidFlowVAEEncode,
    "PyramidFlowTorchCompileSettings": PyramidFlowTorchCompileSettings,
    "PyramidFlowTransformerLoader": PyramidFlowModelLoader,
    "PyramidFlowVAELoader": PyramidFlowVAELoader,
    "PyramidFlowSlidingContextOptions": PyramidFlowSlidingContextOptions,  # Added new node
    "PyramidFlowLatentPreview": PyramidFlowLatentPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyramidFlowSampler": "PyramidFlow Sampler",
    "PyramidFlowVAEDecode": "PyramidFlow VAE Decode",
    "PyramidFlowTextEncode": "PyramidFlow Text Encode",
    "PyramidFlowVAEEncode": "PyramidFlow VAE Encode",
    "PyramidFlowTorchCompileSettings": "PyramidFlow Torch Compile Settings",
    "PyramidFlowTransformerLoader": "PyramidFlow Model Loader",
    "PyramidFlowVAELoader": "PyramidFlow VAE Loader",
    "PyramidFlowSlidingContextOptions": "PyramidFlow Sliding Context Options",  # Added new node
    "PyramidFlowLatentPreview": "PyramidFlow Latent Preview"
}