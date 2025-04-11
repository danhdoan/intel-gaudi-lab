#!/bin/bash

python3 experimentals/text_to_video/text_to_video_generation.py \
    --model_path "./models/ali-vilab-text-to-video-1.7b" \
	--prompts "An astronaut riding a horse." \
	--negative_prompts "low resolution, motionless, Ugly faces, incomplete arms" \
    --num_videos_per_prompt 1 \
    --output_path output/$(date +'%y%m%d_%H%M') \
    --use_habana \
    --use_hpu_graphs \
    --bf16 \
	--batch_size 1 \
	--num_frames 25 \
	--num_inference_steps 25 \
	--guidance_scale 7.5 \
	--seed 42 \
