import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from contextlib import nullcontext
from einops import rearrange
from .pyramid_dit import PyramidDiTForVideoGeneration

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

if not "pyramidflow" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("pyramidflow", os.path.join(folder_paths.models_dir, "pyramidflow"))
    
class DownloadAndLoadPyramidFlowModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "rain1011/pyramid-flow-sd3",
                    ],
                ),
                "variant": (
                    ["diffusion_transformer_384p", "diffusion_transformer_768p"],
                ),

            },
            "optional": {
                "model_dtype": (["fp16", "fp32", "bf16"],{"default": "bf16", }),
                "text_encoder_dtype": (["fp16", "fp32", "bf16"],{"default": "bf16", }),
                "vae_dtype": (["fp16", "fp32", "bf16"],{"default": "bf16", }),
                "use_flash_attn": ("BOOLEAN", {"default": False}),
                #"fp8_transformer": (['disabled', 'enabled', 'fastmode'], {"default": 'disabled', "tooltip": "enabled casts the transformer to torch.float8_e4m3fn, fastmode is only for latest nvidia GPUs"}),
                #"compile": (["disabled","onediff","torch"], {"tooltip": "compile the model for faster inference, these are advanced options only available on Linux, see readme for more info"}),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWMODEL", )
    RETURN_NAMES = ("pyramidflow_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "PyramidFlowWrapper"

    def loadmodel(self, model, variant, model_dtype, text_encoder_dtype, vae_dtype, use_flash_attn=False):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_dtype]
        text_encoder_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[text_encoder_dtype]
        vae_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[vae_dtype]

        base_path = folder_paths.get_folder_paths("pyramidflow")[0]
        
        model_path = os.path.join(base_path, model.split("/")[-1])
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model,
                #ignore_patterns=["*text_encoder*", "*tokenizer*"],
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        model = PyramidDiTForVideoGeneration(
            model_path,
            model_dtype,
            text_encoder_dtype,
            vae_dtype,
            model_variant=variant,
        )
       
        # #fp8
        # if fp8_transformer == "enabled" or fp8_transformer == "fastmode":
        #     if "2b" in model:
        #         for name, param in transformer.named_parameters():
        #             if name != "pos_embedding":
        #                 param.data = param.data.to(torch.float8_e4m3fn)
        #     elif "I2V" in model:
        #         for name, param in transformer.named_parameters():
        #             if "patch_embed" not in name:
        #                 param.data = param.data.to(torch.float8_e4m3fn)
        #     else:
        #         transformer.to(torch.float8_e4m3fn)
        
        #     if fp8_transformer == "fastmode":
        #         from .fp8_optimization import convert_fp8_linear
        #         convert_fp8_linear(transformer, dtype)

    

        # # compilation
        # if compile == "torch":
        #     torch._dynamo.config.suppress_errors = True
        #     pipe.transformer.to(memory_format=torch.channels_last)
        #     pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        # elif compile == "onediff":
        #     from onediffx import compile_pipe
        #     os.environ['NEXFORT_FX_FORCE_TRITON_SDPA'] = '1'
            
        #     pipe = compile_pipe(
        #     pipe,
        #     backend="nexfort",
        #     options= {"mode": "max-optimize:max-autotune:max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": False, "triton.fuse_attention_allow_fp16_reduction": False}},
        #     ignores=["vae"],
        #     fuse_qkv_projections=True if pab_config is None else False,
        #     )

        pyramid_pipe = {
            "model": model,
            "dtype": model_dtype,
            "text_encoder_dtype": text_encoder_dtype,
            "vae_dtype": vae_dtype,
        }
        return (pyramid_pipe,)

    
class CogVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, clip, prompt, strength=1.0, force_offload=True):
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = 226
        clip.cond_stage_model.to(load_device)
        tokens = clip.tokenize(prompt, return_word_ids=True)

        embeds = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=False)
        embeds *= strength
        if force_offload:
            clip.cond_stage_model.to(offload_device)

        return (embeds, )

class PyramidFlowSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMIDFLOWMODEL",),
                "prompt_embeds": ("PYRAMIDFLOWPROMPT",),
                "width": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 8}),
                "first_frame_steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1, "tooltip": "Number of steps for the first frame, no effect when using input_latent"}),
                "video_steps": ("INT", {"default": 10, "min": 1, "max": 2048, "step": 1, "tooltip": "Number of steps for the video latents"}),
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

    RETURN_TYPES = ("PYRAMIDFLOWMODEL", "LATENT", )
    RETURN_NAMES = ("model","samples", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, first_frame_steps, prompt_embeds, seed, height, width, video_steps, temp, guidance_scale, video_guidance_scale, 
               keep_model_loaded, input_latent=None):
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        autocastcondition = not model["model"].dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device), dtype=model["model"].dtype) if autocastcondition else nullcontext()

        if input_latent is None:
            with autocast_context:
                latents = model["model"].generate(
                    prompt_embeds_dict = prompt_embeds,
                    device=device,
                    num_inference_steps=[first_frame_steps, first_frame_steps, first_frame_steps],
                    video_num_inference_steps=[video_steps, video_steps, video_steps], 
                    height=height,
                    width=width,
                    temp=temp,
                    guidance_scale=guidance_scale,         # The guidance for the first frame
                    video_guidance_scale=video_guidance_scale,   # The guidance for the other video latent
                    output_type="latent",
                )
        else:
            with autocast_context:
                latents = model["model"].generate_i2v(
                    prompt_embeds_dict = prompt_embeds,
                    input_image_latent=input_latent,
                    device=device,
                    num_inference_steps=[video_steps, video_steps, video_steps], #why's this a list
                    height=height,
                    width=width,
                    temp=temp,
                    video_guidance_scale=video_guidance_scale,   # The guidance for the other video latent
                    output_type="latent",
                )


        if not keep_model_loaded:
            model["model"].dit.to(offload_device)      

        return (model, {"samples": latents},)
    
class PyramidFlowTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMIDFLOWMODEL",),
                "positive_prompt": ("STRING", {"default": "hyper quality, Ultra HD, 8K", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
               
            },
            # "optional": {
            #     "samples": ("LATENT", ),
            # }
        }

    RETURN_TYPES = ("PYRAMIDFLOWPROMPT", )
    RETURN_NAMES = ("prompt_embeds", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, positive_prompt, negative_prompt, keep_model_loaded):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        text_encoder = model["model"].text_encoder

        autocastcondition = not model["text_encoder_dtype"] == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()

        text_encoder.to(device)
        with autocast_context:
            prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = text_encoder(positive_prompt, device)
            negative_prompt_embeds, negative_prompt_attention_mask, pooled_negative_prompt_embeds = text_encoder(negative_prompt, device)
        if not keep_model_loaded:
            text_encoder.to(offload_device)

        embeds = {
            "prompt_embeds": prompt_embeds,
            "attention_mask": prompt_attention_mask,
            "pooled_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_attention_mask": negative_prompt_attention_mask,
            "negative_pooled_embeds": pooled_negative_prompt_embeds
        }

        return (embeds,)
    
class PyramidFlowVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMIDFLOWMODEL",),
                "image": ("IMAGE",),               
            },
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, image):
        mm.soft_empty_cache()

        self.vae = model["model"].vae
        dtype = model["vae_dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        self.vae.enable_tiling()

        # For the image latent
        self.vae_shift_factor = 0.1490
        self.vae_scale_factor = 1 / 1.8415

        # For the video latent
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986
        input_image_tensor = image * 2 - 1
        input_image_tensor = rearrange(input_image_tensor, 'b h w c -> b c h w')
        input_image_tensor = input_image_tensor.unsqueeze(2)  # Add temporal dimension t=1
        input_image_tensor = input_image_tensor.to(dtype=dtype, device=device)

        self.vae.to(device)
        input_image_latent = (self.vae.encode(input_image_tensor).latent_dist.sample() - self.vae_shift_factor) * self.vae_scale_factor  # [b c 1 h w]
        self.vae.to(offload_device)

        
        return (input_image_latent,)
    
class PyramidFlowVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMIDFLOWMODEL",),
                "samples": ("LATENT",),
                "tile_sample_min_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 8}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
               
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, samples, tile_sample_min_size, window_size):
        mm.soft_empty_cache()

        latents = samples["samples"]
        self.vae = model["model"].vae

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        self.vae.enable_tiling()

        # For the image latent
        self.vae_shift_factor = 0.1490
        self.vae_scale_factor = 1 / 1.8415

        # For the video latent
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

        self.vae.to(device)
        latents = latents.to(self.vae.dtype)
        if latents.shape[2] == 1:
            latents = (latents / self.vae_scale_factor) + self.vae_shift_factor
        else:
            latents[:, :, :1] = (latents[:, :, :1] / self.vae_scale_factor) + self.vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / self.vae_video_scale_factor) + self.vae_video_shift_factor

        image = self.vae.decode(latents, temporal_chunk=True, window_size=window_size, tile_sample_min_size=tile_sample_min_size).sample

        self.vae.to(offload_device)

        image = image.float()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, "B C T H W -> (B T) H W C")
        image = image.cpu().float()

        
        return (image,)
    

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadPyramidFlowModel": DownloadAndLoadPyramidFlowModel,
    "PyramidFlowSampler": PyramidFlowSampler,
    "PyramidFlowVAEDecode": PyramidFlowVAEDecode,
    "PyramidFlowTextEncode": PyramidFlowTextEncode,
    "PyramidFlowVAEEncode": PyramidFlowVAEEncode,
   
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadPyramidFlowModel": "(Down)load PyramidFlow Model",
    "PyramidFlowSampler": "PyramidFlow Sampler",
    "PyramidFlowVAEDecode" : "PyramidFlow VAE Decode",
    "PyramidFlowTextEncode": "PyramidFlow Text Encode",
    "PyramidFlowVAEEncode": "PyramidFlow VAE Encode",
    }
