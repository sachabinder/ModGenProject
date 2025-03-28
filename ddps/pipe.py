import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteSchedulerOutput,
)
import numpy as np


class EulerAncestralDSG(EulerAncestralDiscreteScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            prediction_type=prediction_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict=False,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        noise = randn_tensor(
            model_output.shape,
            dtype=model_output.dtype,
            device=device,
            generator=generator,
        )

        prev_sample_mean = prev_sample
        prev_sample_mean = prev_sample_mean.to(model_output.dtype)

        prev_sample = prev_sample + noise * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return prev_sample, pred_original_sample, prev_sample_mean, sigma_up


class StableDiffusionInverse(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    def run_dps(
        self,
        f,
        y,
        scale,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        timesteps,
        timestep_cond,
        num_warmup_steps,
        num_inference_steps,
        generator,
        callback,
        callback_steps,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        added_cond_kwargs,
        extra_step_kwargs,
        **kwargs,
    ):
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                with torch.enable_grad():
                    latents.requires_grad = True
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    scheduler_out = self.scheduler.step(
                        noise_pred, t, latents, return_dict=True, **extra_step_kwargs
                    )
                    latents_next, pred_z0 = (
                        scheduler_out.prev_sample.to(torch.float16),
                        scheduler_out.pred_original_sample.to(torch.float16),
                    )

                    pred_x0 = self.vae.decode(
                        pred_z0 / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    norm = torch.linalg.norm(f(pred_x0) - y)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                    progress_bar.set_postfix({"distance": norm.item()}, refresh=False)

                latents_next = latents_next - norm_grad * scale
                latents = latents_next.detach()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        return latents

    def run_fdm(
        self,
        f,
        y,
        scale,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        timesteps,
        timestep_cond,
        num_warmup_steps,
        num_inference_steps,
        generator,
        callback,
        callback_steps,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        added_cond_kwargs,
        extra_step_kwargs,
        **kwargs,
    ):
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if i >= kwargs["fdm_c1"] and i <= kwargs["fdm_c2"]:
                    K = kwargs["fdm_k"]
                else:
                    K = 1
                for j in range(K):
                    with torch.enable_grad():
                        latents.requires_grad = True
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (
                            torch.cat([latents] * 2)
                            if self.do_classifier_free_guidance
                            else latents
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        if (
                            self.do_classifier_free_guidance
                            and self.guidance_rescale > 0.0
                        ):
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(
                                noise_pred,
                                noise_pred_text,
                                guidance_rescale=self.guidance_rescale,
                            )

                        # compute the previous noisy sample x_t -> x_t-1
                        scheduler_out = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            return_dict=True,
                            **extra_step_kwargs,
                        )
                        latents_next, pred_z0 = (
                            scheduler_out.prev_sample,
                            scheduler_out.pred_original_sample,
                        )

                        pred_x0 = self.vae.decode(
                            pred_z0 / self.vae.config.scaling_factor,
                            return_dict=False,
                            generator=generator,
                        )[0]
                        norm = torch.linalg.norm(f(pred_x0) - y)
                        norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                        progress_bar.set_postfix(
                            {"distance": norm.item()}, refresh=False
                        )

                    latents_next = latents_next - norm_grad * scale
                    latents = latents_next.detach()

                    if j != (K - 1) and i != len(timesteps) - 1:
                        eps = torch.randn_like(latents)
                        sigma_t = self.scheduler.sigmas[i]
                        sigma_prevt = self.scheduler.sigmas[i + 1]
                        sigma_delta = torch.sqrt(sigma_t**2 - sigma_prevt**2)
                        latents = latents + sigma_delta * eps
                        self.scheduler._step_index = self.scheduler._step_index - 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        return latents

    def run_dsg(
        self,
        f,
        y,
        scale,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        timesteps,
        timestep_cond,
        num_warmup_steps,
        num_inference_steps,
        generator,
        callback,
        callback_steps,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        added_cond_kwargs,
        extra_step_kwargs,
        **kwargs,
    ):
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                with torch.enable_grad():
                    latents.requires_grad = True
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    (
                        latents_next,
                        pred_z0,
                        latents_next_mean,
                        sigma_up,
                    ) = self.scheduler.step(
                        noise_pred, t, latents, return_dict=True, **extra_step_kwargs
                    )

                    pred_x0 = self.vae.decode(
                        pred_z0 / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    norm = torch.linalg.norm(f(pred_x0) - y)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]
                    progress_bar.set_postfix({"distance": norm.item()}, refresh=False)

                grad_norm = torch.linalg.norm(norm_grad)

                _, c, h, w = latents_next.shape
                r = torch.sqrt(torch.tensor(c * h * w)) * sigma_up
                eps = 1e-8

                d_star = -r * norm_grad / (grad_norm + eps)
                d_sample = latents_next - latents_next_mean
                mix_direction = d_sample + scale * (d_star - d_sample)
                mix_direction_norm = torch.linalg.norm(mix_direction)
                mix_step = mix_direction / (mix_direction_norm + eps) * r

                latents = latents_next_mean.detach() + mix_step

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        return latents

    def run_psld(
        self,
        f,
        y,
        scale,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        timesteps,
        timestep_cond,
        num_warmup_steps,
        num_inference_steps,
        generator,
        callback,
        callback_steps,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        added_cond_kwargs,
        extra_step_kwargs,
        **kwargs,
    ):
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                with torch.enable_grad():
                    latents.requires_grad = True
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    scheduler_out = self.scheduler.step(
                        noise_pred, t, latents, return_dict=True, **extra_step_kwargs
                    )
                    latents_next, pred_z0 = (
                        scheduler_out.prev_sample.to(torch.float16),
                        scheduler_out.pred_original_sample.to(torch.float16),
                    )
                    pred_x0 = self.vae.decode(
                        pred_z0 / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    # TODO : Check here for the gluing term
                    ortho_project = pred_x0 - f.transpose(f(pred_x0))
                    parallel_project = f.transpose(y)
                    inpainted_image = parallel_project + ortho_project

                    encoded_z_0 = (
                        self.vae.encode(inpainted_image).latent_dist.sample()
                        * self.vae.config.scaling_factor
                    )
                    inpaint_error = torch.linalg.norm(encoded_z_0 - pred_z0)

                    dist = torch.linalg.norm(f(pred_x0) - y)
                    norm = dist + kwargs["psld_gamma"] * inpaint_error
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]

                    progress_bar.set_postfix({"distance": dist.item()}, refresh=False)

                latents_next = latents_next - norm_grad * scale
                latents = latents_next.detach()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        return latents

    def run_gml_dps(
        self,
        f,
        y,
        scale,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        timesteps,
        timestep_cond,
        num_warmup_steps,
        num_inference_steps,
        generator,
        callback,
        callback_steps,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        added_cond_kwargs,
        extra_step_kwargs,
        **kwargs,
    ):
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                with torch.enable_grad():
                    latents.requires_grad = True
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    scheduler_out = self.scheduler.step(
                        noise_pred, t, latents, return_dict=True, **extra_step_kwargs
                    )
                    latents_next, pred_z0 = (
                        scheduler_out.prev_sample.to(torch.float16),
                        scheduler_out.pred_original_sample.to(torch.float16),
                    )
                    pred_x0 = self.vae.decode(
                        pred_z0 / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    encoded_z_0 = (
                        self.vae.encode(pred_x0).latent_dist.sample()
                        * self.vae.config.scaling_factor
                    )
                    inpaint_error = torch.linalg.norm(encoded_z_0 - pred_z0)

                    dist = torch.linalg.norm(f(pred_x0) - y)
                    norm = dist + kwargs["psld_gamma"] * inpaint_error
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=latents)[0]

                    progress_bar.set_postfix({"distance": dist.item()}, refresh=False)

                latents_next = latents_next - norm_grad * scale
                latents = latents_next.detach()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        f,
        y,
        algo,
        scale,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        latents_start=None,
        timesteps_start=0,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        num_channels_latents = self.unet.config.in_channels

        # 5. Prepare latent variables
        if latents_start == None:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        else:
            # add noise to latent
            latents = latents_start
            latent_noise = randn_tensor(
                latents.shape, generator, device, prompt_embeds.dtype
            )
            latents = self.scheduler.add_noise(
                latents, latent_noise, timesteps[timesteps_start : timesteps_start + 1]
            )
            # clip the timestep
            timesteps = timesteps[timesteps_start:]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if algo == "dps":
            latent_generation_function = self.run_dps
        elif algo == "fdm":
            latent_generation_function = self.run_fdm
        elif algo == "dsg":
            latent_generation_function = self.run_dsg
        elif algo == "psld":
            latent_generation_function = self.run_psld
        elif algo == "gml_dps":
            latent_generation_function = self.run_gml_dps
        else:
            raise NotImplementedError

        latents = latent_generation_function(
            f,
            y,
            scale,
            latents,
            prompt_embeds,
            negative_prompt_embeds,
            timesteps,
            timestep_cond,
            num_warmup_steps,
            num_inference_steps,
            generator,
            callback,
            callback_steps,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            added_cond_kwargs,
            extra_step_kwargs,
            **kwargs,
        )

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
