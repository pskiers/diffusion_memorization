import torch

from diffusers import StableDiffusionPipeline


class LocalStableDiffusionPipeline(StableDiffusionPipeline):

    def __call__(
        self,
        prompt=None,
        token_intervention=None,
        token_intervention_pos=None,
        intervention_strenght=None,
        **kwargs,
    ):
        if token_intervention is not None:
            device = self._execution_device
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            negative_prompt = kwargs.pop("negative_prompt", None)
            num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
            clip_skip = kwargs.get("clip_skip", None)
            cross_attention_kwargs = kwargs.get("cross_attention_kwargs", None)
            lora_scale = (
                cross_attention_kwargs.get("scale", None)
                if isinstance(cross_attention_kwargs, dict)
                else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                negative_prompt=negative_prompt,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )

            intervention = token_intervention.to(prompt_embeds.device)
            positions = torch.tensor(token_intervention_pos, device=prompt_embeds.device)
            strenghts = (
                intervention_strenght.to(prompt_embeds.device)
                if torch.is_tensor(intervention_strenght)
                else torch.tensor(intervention_strenght, device=prompt_embeds.device)
            )
            batch_idx = torch.arange(intervention.shape[0], device=prompt_embeds.device)
            prompt_embeds[batch_idx, positions] += intervention * strenghts.unsqueeze(dim=1)

            return super().__call__(
                prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                **kwargs,
            )

        return super().__call__(prompt=prompt, **kwargs)
