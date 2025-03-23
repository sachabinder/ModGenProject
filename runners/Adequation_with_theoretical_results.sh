#!/bin/bash

# gaussian blur
python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/gdb/dps \
  --scale 4.8 \
  --algo dps \
  --operator gdb \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/gdb/psld \
  --scale 3.5 \
  --algo psld \
  --operator gdb \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/gdb/gml_dps \
  --scale 3.5 \
  --algo gml_dps \
  --operator gdb \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 


# Super resolution
python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/srx8/dps \
  --scale 4.8 \
  --algo dps \
  --operator srx8 \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/srx8/psld \
  --scale 3.5 \
  --algo psld \
  --operator srx8 \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/srx8/gml_dps \
  --scale 3.5 \
  --algo gml_dps \
  --operator srx8 \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base 

# Mask impainting
python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/bip/dps \
  --scale 4.8 \
  --algo dps \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --impating_mask_color gray

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/bip/psld \
  --scale 3.5 \
  --algo psld \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --impating_mask_color gray

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/bip/gml_dps \
  --scale 3.5 \
  --algo gml_dps \
  --operator bip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --impating_mask_color gray


# Random impainting
python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/rip/dps \
  --scale 4.8 \
  --algo dps \
  --operator rip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color gray

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/rip/psld \
  --scale 3.5 \
  --algo psld \
  --operator rip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color gray

python main.py \
  --data ./test_image \
  --out ./paper_experiment/pract_theo/rip/gml_dps \
  --scale 3.5 \
  --algo gml_dps \
  --operator rip \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \
  --rand_impating_proportion 0.25 \
  --impating_mask_color gray