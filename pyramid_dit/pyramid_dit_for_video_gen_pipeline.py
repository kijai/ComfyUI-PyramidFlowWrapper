from types import SimpleNamespace

import torch
import torch.nn.functional as F

from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor

import math
from tqdm import tqdm

from typing import List, Optional, Union, Callable
from ..diffusion_schedulers import PyramidFlowMatchEulerDiscreteScheduler
from accelerate import cpu_offload
from comfy.utils import ProgressBar

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

class PyramidDiTForVideoGeneration:
    """
        The pyramid dit for both image and video generation, The running class wrapper
        This class is mainly for fixed unit implementation: 1 + n + n + n
    """
    def __init__(
            self,
            transformer, 
            model_dtype, 
            model_name, 
            main_device,
            return_log=True, 
            timestep_shift=1.0, 
            stage_range=[0, 1/3, 2/3, 1],
            sample_ratios=[1, 1, 1], 
            scheduler_gamma=1/3, 
            max_temporal_length=31, 
            frame_per_unit=1, 
            use_temporal_causal=True, 
            corrupt_ratio=1/3, 
            interp_condition_pos=True, 
            stages=[1, 2, 4], 
            **kwargs,
        ):
        super().__init__()

        self.dit = transformer
        self.device = main_device
        self.sequential_offload_enabled = False

        from comfy import latent_formats
        self.model = SimpleNamespace(latent_format=latent_formats.Flux())
        self.load_device = main_device

        if model_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            self.dtype = torch.bfloat16
        else:
            self.dtype = model_dtype

        self.stages = stages
        self.sample_ratios = sample_ratios
        self.corrupt_ratio = corrupt_ratio
        self.model_name = model_name
        
        # For the image latent
        self.vae_shift_factor = 0.1490
        self.vae_scale_factor = 1 / 1.8415

        # For the video latent
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

        self.downsample = 8

        # Configure the video training hyper-parameters
        # The video sequence: one frame + N * unit
        self.frame_per_unit = frame_per_unit
        self.max_temporal_length = max_temporal_length
        assert (max_temporal_length - 1) % frame_per_unit == 0, "The frame number should be divided by the frame number per unit"
        self.num_units_per_video = 1 + ((max_temporal_length - 1) // frame_per_unit) + int(sum(sample_ratios))

        self.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            shift=timestep_shift, stages=len(self.stages), 
            stage_range=stage_range, gamma=scheduler_gamma,
        )
        #round the gamma as 1/3 seems to have issues on some systems
        self.gamma = round(self.scheduler.config.gamma, 5)
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 + self.gamma) - torch.ones(4, 4) * self.gamma)        
        
        print(f"The start sigmas and end sigmas of each stage is Start: {self.scheduler.start_sigmas}, End: {self.scheduler.end_sigmas}, Ori_start: {self.scheduler.ori_start_sigmas}")
        
        self.cfg_rate = 0.1
        self.return_log = return_log
        self.use_flash_attn = False
    
    @torch.no_grad()
    def get_pyramid_latent(self, x, stage_num):
        # x is the origin vae latent
        vae_latent_list = []
        vae_latent_list.append(x)

        temp, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
        for _ in range(stage_num):
            height //= 2
            width //= 2
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.interpolate(x, size=(height, width), mode='bilinear')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=temp)
            vae_latent_list.append(x)

        vae_latent_list = list(reversed(vae_latent_list))
        return vae_latent_list

    def _enable_sequential_cpu_offload(self, model, device):
        self.sequential_offload_enabled = True
        offload_buffers = len(model._parameters) > 0
        cpu_offload(model, device, offload_buffers=offload_buffers)
    
    def enable_sequential_cpu_offload(self):
        self._enable_sequential_cpu_offload(self.dit, device=self.device)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        temp,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(temp),
            int(height) // self.downsample,
            int(width) // self.downsample,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents
    
    def sample_block_noise(self, bs, ch, temp, height, width):
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        noise = torch.stack([self.dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',b=bs,c=ch,t=temp,h=height//2,w=width//2,p=2,q=2)
        return noise



    @torch.no_grad()
    def generate_one_unit(
        self,
        latents,
        past_conditions, # List of past conditions, contains the conditions of each stage
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        num_inference_steps,
        height,
        width,
        temp,
        device,
        dtype,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        is_first_frame: bool = False,
        callback = None,
    ):
        stages = self.stages
        intermed_latents = []
        #print(f"Start generating one unit, the latents shape is {latents.shape}")

        for i_s in range(len(stages)):
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device)
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                height *= 2; width *= 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)
                # Fix the stage
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]   # the original coeff of signal
                alpha = 1 / (math.sqrt(1 + (1 / self.gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(self.gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(bs, ch, temp, height, width)
                noise = noise.to(device=device, dtype=dtype)
                latents = alpha * latents + beta * noise    # To fix the block artifact

            for idx, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                
                latent_model_input = past_conditions[i_s] + [latent_model_input]
                noise_pred = self.dit(
                    sample=[latent_model_input],
                    timestep_ratio=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                )

                noise_pred = noise_pred[0]
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if is_first_frame:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + self.video_guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)
           

        return intermed_latents

    @torch.no_grad()
    def generate_i2v(
        self,
        prompt_embeds_dict: dict,
        device: torch.device,
        input_image_latent: torch.Tensor,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 4.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable] = None,
    ):
        dtype = self.dtype

        assert temp % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"
        batch_size = prompt_embeds_dict['prompt_embeds'].shape[0]

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)
        elif isinstance(num_inference_steps, list) and len(num_inference_steps) < len(self.stages):
            num_inference_steps = (num_inference_steps * len(self.stages))[:len(self.stages)]
        
        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp+1)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        positive_prompt_embeds = prompt_embeds_dict['prompt_embeds']
        positive_pooled_prompt_embeds = prompt_embeds_dict['pooled_embeds']
        positive_prompt_attention_mask = prompt_embeds_dict['attention_mask']

        negative_prompt_embeds = prompt_embeds_dict['negative_prompt_embeds']
        negative_pooled_prompt_embeds = prompt_embeds_dict['negative_pooled_embeds']
        negative_prompt_attention_mask = prompt_embeds_dict['negative_attention_mask']

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, positive_pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, positive_prompt_attention_mask], dim=0)

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by defalut, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = temp // self.frame_per_unit
        stages = self.stages
        
        input_image_latent = input_image_latent.to(dtype).to(device)
        generated_latents_list = [input_image_latent]    # The generated results

        if not self.sequential_offload_enabled:
            self.dit.to(device)
        comfy_pbar = ProgressBar(num_units)

        for unit_index in tqdm(range(1, num_units + 1)):
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            # prepare the condition latents
            past_condition_latents = []
            clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
            
            for i_s in range(len(stages)):
                last_cond_latent = clean_latents_list[i_s][:,:,-self.frame_per_unit:]

                stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
        
                # pad the past clean latents
                cur_unit_num = unit_index
                cur_stage = i_s
                cur_unit_ptx = 1

                while cur_unit_ptx < cur_unit_num:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                    cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                    stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
            
                stage_input = list(reversed(stage_input))
                past_condition_latents.append(stage_input)

            intermed_latents = self.generate_one_unit(
                latents[:,:,(unit_index - 1) * self.frame_per_unit:unit_index * self.frame_per_unit],
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                num_inference_steps,
                height,
                width,
                self.frame_per_unit,
                device,
                dtype,
                generator,
                is_first_frame=False,
                callback=callback
            )
            
            
            if callback is not None:
                callback(unit_index, intermed_latents[-1].detach()[0].permute(1,0,2,3), None, temp)
            else:
                comfy_pbar.update(1)
            generated_latents_list.append(intermed_latents[-1])

        generated_latents = torch.cat(generated_latents_list, dim=2)

        return generated_latents

    @torch.no_grad()
    def generate(
        self,
        prompt_embeds_dict: dict,
        height: Optional[int] = None,
        width: Optional[int] = None,
        temp: int = 1,
        num_inference_steps: Optional[Union[int, List[int]]] = 28,
        video_num_inference_steps: Optional[Union[int, List[int]]] = 28,
        guidance_scale: float = 7.0,
        video_guidance_scale: float = 7.0,
        min_guidance_scale: float = 2.0,
        use_linear_guidance: bool = False,
        alpha: float = 0.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device: Optional[torch.device] = None,
        callback = None
    ):
        assert (temp - 1) % self.frame_per_unit == 0, "The frames should be divided by frame_per unit"

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.stages)
        elif isinstance(num_inference_steps, list) and len(num_inference_steps) < len(self.stages):
            num_inference_steps = (num_inference_steps * len(self.stages))[:len(self.stages)]

        if isinstance(video_num_inference_steps, int):
            video_num_inference_steps = [video_num_inference_steps] * len(self.stages)
        elif isinstance(video_num_inference_steps, list) and len(video_num_inference_steps) < len(self.stages):
            video_num_inference_steps = (video_num_inference_steps * len(self.stages))[:len(self.stages)]


        batch_size = prompt_embeds_dict['prompt_embeds'].shape[0]

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp)]
            print(guidance_scale_list)

        self._guidance_scale = guidance_scale
        self._video_guidance_scale = video_guidance_scale

        positive_prompt_embeds = prompt_embeds_dict['prompt_embeds']
        positive_pooled_prompt_embeds = prompt_embeds_dict['pooled_embeds']
        positive_prompt_attention_mask = prompt_embeds_dict['attention_mask']

        negative_prompt_embeds = prompt_embeds_dict['negative_prompt_embeds']
        negative_pooled_prompt_embeds = prompt_embeds_dict['negative_pooled_embeds']
        negative_prompt_attention_mask = prompt_embeds_dict['negative_attention_mask']
       

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, positive_pooled_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, positive_prompt_attention_mask], dim=0)

        prompt_embeds = prompt_embeds.to(self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.dtype)
        prompt_attention_mask = prompt_attention_mask.to(self.dtype)

        # Create the initial random noise
        num_channels_latents = (self.dit.config.in_channels // 4) if self.model_name == "pyramid_flux" else  self.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            temp,
            height,
            width,
            self.dtype,
            device,
            generator,
        )

        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        # by defalut, we needs to start from the block noise
        for _ in range(len(self.stages)-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2
        
        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = 1 + (temp - 1) // self.frame_per_unit
        stages = self.stages

        generated_latents_list = []    # The generated results

        if not self.sequential_offload_enabled:
            self.dit.to(device)
        comfy_pbar = ProgressBar(num_units)

        for unit_index in tqdm(range(num_units)):
            if use_linear_guidance:
                self._guidance_scale = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            if unit_index == 0:
                past_condition_latents = [[] for _ in range(len(stages))]
                intermed_latents = self.generate_one_unit(
                    latents[:,:,:1],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    num_inference_steps,
                    height,
                    width,
                    1,
                    device,
                    self.dtype,
                    generator,
                    is_first_frame=True,
                    callback=callback
                )
            else:
                # prepare the condition latents
                past_condition_latents = []
                clean_latents_list = self.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                
                for i_s in range(len(stages)):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(self.frame_per_unit):]

                    stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
            
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                intermed_latents = self.generate_one_unit(
                    latents[:,:, 1 + (unit_index - 1) * self.frame_per_unit:1 + unit_index * self.frame_per_unit],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    video_num_inference_steps,
                    height,
                    width,
                    self.frame_per_unit,
                    device,
                    self.dtype,
                    generator,
                    is_first_frame=False,
                    callback=callback
                )

            comfy_pbar.update(1)
            generated_latents_list.append(intermed_latents[-1])
            if callback is not None:
                callback(unit_index, intermed_latents[-1].detach()[0].permute(1,0,2,3), None, temp)
            else:
                comfy_pbar.update(1)

        generated_latents = torch.cat(generated_latents_list, dim=2)

        return generated_latents
    
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def video_guidance_scale(self):
        return self._video_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0
