#!/bin/bash

python main.py \
  --data ./example_imgs \
  --out ./example_outputs/dps/srx8 \
  --scale 4.8 \
  --algo dps \
  --operator srx8 \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 
