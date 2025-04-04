#!/bin/bash

python experimentals/inference_stable_diffusion_enouvo.py \
	--prompt "peaceful beach, horizon, wide view, master works, movie-grade light effect, cinematic, golden hour, sunlight, grainy sand, waves, Fujifilm X-T4, HD, HQ, 4k" \
	--negative_prompt "blurred, paintings, sketches, lowres, normal quality, unnatural" \
	--num_images_per_prompt 1 \
	--batch_size 1 \
	--num_inference_steps 30 \
	--height 1024 \
	--width 1024 \
	--scheduler "ddim" \
	--model_name_or_path "models/stable-diffusion-xl-base-1.0" \
	--bf16 \
	--use_habana \
	--use_hpu_graphs \
	--gaudi_config_name "Habana/stable-diffusion"
