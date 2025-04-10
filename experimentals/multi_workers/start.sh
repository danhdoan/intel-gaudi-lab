#!/bin/bash
torchrun --nproc_per_node=8 --master_port=12345 multi-hpu-app/worker.py
