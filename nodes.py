import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from contextlib import nullcontext

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
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16", "tooltip": "official recommendation is that 2b model should be fp16, 5b model should be bf16"}
                ),
                #"fp8_transformer": (['disabled', 'enabled', 'fastmode'], {"default": 'disabled', "tooltip": "enabled casts the transformer to torch.float8_e4m3fn, fastmode is only for latest nvidia GPUs"}),
                #"compile": (["disabled","onediff","torch"], {"tooltip": "compile the model for faster inference, these are advanced options only available on Linux, see readme for more info"}),
            }
        }

    RETURN_TYPES = ("PYRAMIDFLOWMODEL", )
    RETURN_NAMES = ("pyramidflow_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "PyramidFlowWrapper"

    def loadmodel(self, model, variant, precision):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
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
            dtype,
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


        return (model,)

    
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
                "height": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 656, "min": 128, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "video_steps": ("INT", {"default": 10, "min": 5, "max": 2048, "step": 4}),
                "temp": ("INT", {"default": 8, "min": 1}),
                "guidance_scale": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "The guidance for the first frame"}),
                "video_guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "The guidance for the other video latent"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
               
            },
            # "optional": {
            #     "samples": ("LATENT", ),
            # }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, model, steps, prompt, seed, height, width, video_steps, temp, guidance_scale, video_guidance_scale, keep_model_loaded):
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model.vae.enable_tiling()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        autocastcondition = not model.dtype == torch.float32
        autocast_context = torch.autocast(mm.get_autocast_device(device)) if autocastcondition else nullcontext()

        model.dit.to(device)
        #model.vae.to(device)
        #model.text_encoder.to(device)
        with autocast_context:
            frames = model.generate(
                prompt=prompt,
                num_inference_steps=[steps, steps, steps],
                video_num_inference_steps=[video_steps, video_steps, video_steps],
                height=height,
                width=width,
                temp=temp,
                guidance_scale=guidance_scale,         # The guidance for the first frame
                video_guidance_scale=video_guidance_scale,   # The guidance for the other video latent
                output_type="pt",
            )
        print(frames.shape)

        if not keep_model_loaded:
            model.to(offload_device)      

        return (frames,)
    

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadPyramidFlowModel": DownloadAndLoadPyramidFlowModel,
    "PyramidFlowSampler": PyramidFlowSampler,
   
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadPyramidFlowModel": "(Down)load PyramidFlow Model",
    "PyramidFlowSampler": "PyramidFlow Sampler",
    }
