import io

import torch
from PIL import Image
import struct
import numpy as np
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import comfy.model_management
import folder_paths
import comfy.utils
import logging

MAX_PREVIEW_RESOLUTION = args.preview_size

def preview_to_image(latent_image):
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            ).to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("GIF", preview_image, MAX_PREVIEW_RESOLUTION)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        latent_rgb_factors =  [[0.05389399697934166, 0.025018778505575393, -0.009193515248318657], [0.02318250640590553, -0.026987363837713156, 0.040172639061236956], [0.046035451343323666, -0.02039565868920197, 0.01275569344290342], [-0.015559161155025095, 0.051403973219861246, 0.03179031307996347], [-0.02766167769640129, 0.03749545161530447, 0.003335141009473408], [0.05824598730479011, 0.021744367381243884, -0.01578925627951616], [0.05260929401500947, 0.0560165014956886, -0.027477296572565126], [0.018513891242931686, 0.041961785217662514, 0.004490763489747966], [0.024063060899760215, 0.065082853069653, 0.044343437673514896], [0.05250992323006226, 0.04361117432588933, 0.01030076055524387], [0.0038921710021782366, -0.025299228133723792, 0.019370764014574535], [-0.00011950534333568519, 0.06549370069727675, -0.03436712163379723], [-0.026020578032683626, -0.013341758571090847, -0.009119046570271953], [0.024412451175602937, 0.030135064560817174, -0.008355486384198006], [0.04002209845752687, -0.017341304390739463, 0.02818338690302971], [-0.032575108695213684, -0.009588338926775117, -0.03077312160940468]]
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        # if latent_rgb_factors_bias is not None:
        #     self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        latent_image = torch.nn.functional.linear(x0[0].permute(1, 2, 0), self.latent_rgb_factors,
                                                    bias=self.latent_rgb_factors_bias)
        return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in folder_paths.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias)
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

