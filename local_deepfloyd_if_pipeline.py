import torch
from typing import Any, Callable, Dict, List, Optional, Union, Literal

from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils.torch_utils import randn_tensor
from flipd_utils import compute_trace_of_jacobian


class LocalStableDiffusionUpscalePipeline(StableDiffusionUpscalePipeline):
    def get_text_cond_grad(
        self,
        prompt: Union[str, List[str]] = None,
        image = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        target_steps=[0],
        method: Literal["cfg_norm", "flipd", "score_norm"] = "cfg_norm",
        start_timestep: Optional[int] = None,
    ):
        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            noise_level,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)
        image = image.to(dtype=prompt_embeds.dtype, device=device)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        if start_timestep is not None:
            timesteps = timesteps[start_timestep:]

        # 5. Add noise to image
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = torch.cat([image] * batch_multiplier * num_images_per_prompt)
        noise_level = torch.cat([noise_level] * image.shape[0])

        # 6. Prepare latent variables
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
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

        # 7. Check that sizes of image and latents match
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents + num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        all_token_grads = []
        all_losses = [] 
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image], dim=1)

                if i in target_steps:
                    single_prompt_embeds = prompt_embeds[[0], :, :].clone().detach()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[-1], :, :].clone()

                    if do_classifier_free_guidance:
                        input_prompt_embeds = torch.cat(
                            [
                                dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                                single_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                            ]
                        )
                    else:
                        input_prompt_embeds = single_prompt_embeds.repeat(num_images_per_prompt, 1, 1)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=input_prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=noise_level,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                    if method == "flipd":
                        alpha_bar = self.scheduler.alphas_cumprod[t]
                        def score_fn(x):
                            x_in = torch.cat([x] * 2) if do_classifier_free_guidance else x
                            x_in = self.scheduler.scale_model_input(x_in, t)
                            x_in = torch.cat([x_in, image], dim=1)
                            in_prompt = torch.cat(
                                [dummy_prompt_embeds.repeat(x.shape[0], 1, 1), single_prompt_embeds.repeat(x.shape[0], 1, 1)]
                            ) if do_classifier_free_guidance else single_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                            n_pred = self.unet(
                                x_in,
                                t,
                                encoder_hidden_states=in_prompt,
                                cross_attention_kwargs=None,
                                class_labels=noise_level,
                                return_dict=False,
                            )[0]
                            if do_classifier_free_guidance:
                                noise_uncond, noise_pred_text = n_pred.chunk(2)
                                return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_uncond)
                            else:
                                return n_pred
                        flipd_trace_term = compute_trace_of_jacobian(
                            score_fn,
                            x=latents,
                            method="hutchinson_gaussian",
                            hutchinson_sample_count=1,
                            chunk_size=1,
                            seed=42,
                            verbose=False,
                        )
                        flipd_score_norm_term = torch.norm(
                            noise_pred_uncond + guidance_scale * noise_pred_text, p=2
                        ) if do_classifier_free_guidance else torch.norm(noise_pred, p=2)
                        flipd = - torch.sqrt(1 - alpha_bar) * flipd_trace_term + flipd_score_norm_term # (+ D) but doesn't matter
                        loss = -flipd.mean()
                        if do_classifier_free_guidance:
                            (token_grads,) = torch.autograd.grad(loss, [prompt_embeds])
                            signs = (token_grads * prompt_embeds).sum(dim=-1).mean(dim=0) > 0
                            dot_tokens_grads = (token_grads * prompt_embeds).sum(dim=-1)
                            dot_tokens_tokens = (prompt_embeds * prompt_embeds).sum(dim=-1) + 0.00001
                            grad_projections = prompt_embeds * (dot_tokens_grads / dot_tokens_tokens).unsqueeze(dim=-1)
                            token_grads = grad_projections.norm(p=2, dim=-1).mean(dim=0).detach()
                            token_grads *= 2 * signs.float() - 1
                        else:
                            token_grads = torch.zeros_like(prompt_embeds)
                        all_token_grads.append(token_grads)
                        all_losses.append(-loss.detach())
                    else:
                        raise ValueError(f"method {method} not supported")
                    
                    all_token_grads.append(token_grads)
                    all_losses.append(-loss.detach())

                    if i == max(target_steps):
                        return torch.mean(torch.stack(all_token_grads), dim=0), torch.mean(torch.stack(all_losses), dim=0)
                else:
                    # do normal forward pass
                    with torch.no_grad():
                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            class_labels=noise_level,
                            return_dict=False,
                        )[0]

                with torch.no_grad():
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)
                if i in target_steps:
                    if do_classifier_free_guidance:
                        del loss, token_grads, noise_pred, noise_pred_uncond, noise_pred_text
                    else:
                        del loss, token_grads, noise_pred
                progress_bar.update()

            return torch.mean(torch.stack(all_token_grads), dim=0), torch.mean(torch.stack(all_losses), dim=0)
