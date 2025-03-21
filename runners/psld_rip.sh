#!/bin/bash
python main.py\
    --data ./example_imgs\
    --out ./example_outputs/psld/rip\
    --scale 4.8\
    --psld_gamma 0.2\
    --algo psld\
    --operator rip\
    --nstep 500\
    --model stabilityai/stable-diffusion-2-base\
    --rand_impating_proportion 0.25
