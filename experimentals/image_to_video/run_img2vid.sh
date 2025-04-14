#!/bin/bash

python3 experimentals/image_to_video/image_to_video_generation.py \
	--model_path "models/ali-vilab/i2vgen-xl" \
	--use_habana \
	--use_hpu_graphs \
	--gaudi_config_name Habana/stable-diffusion \
	--sdp_on_bf16 \
	--bf16 \
	--prompts "Papers were floating in the air on a table in the library" \
	--negative_prompts "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms" \
	--num_videos_per_prompt 1 \
	--batch_size 1 \
	--num_frames 25 \
	--num_inference_steps 25 \
	--guidance_scale 7.5 \
	--seed 42 \
	--image_path "data/image_to_video" \
	--fps 7 \
	--output_path "outputs"
