#!/bin/bash

# Random impainting
python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/gray/dps \
  --scale 4.8 \
  --algo dps \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color gray

python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/gray/psld \
  --scale 3.5 \
  --algo psld \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color gray


python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/blue/dps \
  --scale 4.8 \
  --algo dps \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color blue

python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/blue/psld \
  --scale 3.5 \
  --algo psld \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color blue

  python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/red/dps \
  --scale 4.8 \
  --algo dps \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color red

python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/red/psld \
  --scale 3.5 \
  --algo psld \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color red

    python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/green/dps \
  --scale 4.8 \
  --algo dps \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color green

python main.py \
  --data ./test_image \
  --out ./paper_experiment/bip_colors/green/psld \
  --scale 3.5 \
  --algo psld \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color green