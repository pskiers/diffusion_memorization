from typing import Literal, Union, List, Dict, Any, Callable, Optional

import torch

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import retrieve_timesteps
from flipd_utils import compute_trace_of_jacobian


class LocalStableDiffusionPipeline(StableDiffusionPipeline):
    def get_text_cond_grad(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        target_steps=[0],
        method: Literal["cfg_norm", "flipd", "score_norm"] = "cfg_norm",
    ):
        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            None,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            None,
            None,
            None,
        )

        self._guidance_scale = guidance_scale
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
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
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

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
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

        D = height * width * num_channels_latents
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (None)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        all_token_grads = []
        all_losses = [] 
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # if we should do the LID calculation
                if i in target_steps:
                    single_prompt_embeds = prompt_embeds[[0], :, :].clone().detach()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[-1], :, :].clone()

                    input_prompt_embeds = torch.cat(
                        [
                            dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                            single_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                        ]
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=input_prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_text = noise_pred_text - noise_pred_uncond

                    if method == "score_norm":
                        alpha_bar = self.scheduler.alphas_cumprod[t]
                        loss = torch.norm(noise_pred_uncond + guidance_scale * noise_pred_text, p=2).mean()
                        (token_grads,) = torch.autograd.grad(loss, [prompt_embeds])
                    elif method == "cfg_norm":
                        loss = torch.norm(noise_pred_text, p=2).mean()
                        (token_grads,) = torch.autograd.grad(loss, [prompt_embeds])
                    elif method == "flipd":
                        alpha_bar = self.scheduler.alphas_cumprod[t]
                        def score_fn(x):
                            noise_uncond, noise_pred_text = self.unet(
                                torch.cat([x, x]),
                                t,
                                encoder_hidden_states=torch.cat([dummy_prompt_embeds.repeat(x.shape[0], 1, 1),single_prompt_embeds.repeat(x.shape[0], 1, 1)]),
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=None,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0].chunk(2)
                            return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_uncond)
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
                        )
                        flipd = - torch.sqrt(1 - alpha_bar) * flipd_trace_term + flipd_score_norm_term # (+ D) but doesn't matter
                        loss = -flipd.mean()
                        (token_grads,) = torch.autograd.grad(loss, [prompt_embeds])
                    else:
                        raise ValueError(f"method {method} not supported")
                    
                    token_grads = token_grads.norm(p=2, dim=-1).mean(dim=0).detach()
                    all_token_grads.append(token_grads)
                    all_losses.append(-loss.detach())

                    if i == max(target_steps):
                        return torch.mean(torch.stack(all_token_grads), dim=0), torch.mean(torch.stack(all_losses), dim=0)
                    # delete unused variables
                    del loss, token_grads, noise_pred, noise_pred_uncond, noise_pred_text
                else:
                    # do the normal forward pass
                    with torch.no_grad():
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

                with torch.no_grad():
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
