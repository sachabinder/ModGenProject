#!/bin/bash

python main.py \
  --data ./test_image \
  --out ./paper_experiment/non_linear/mdb/dps \
  --scale 4.8 \
  --algo dps \
  --operator mdb \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \

  python main.py \
  --data ./test_image \
  --out ./paper_experiment/non_linear/mdb/gml_psld \
  --scale 4.8 \
  --algo gml_psld \
  --operator mdb \
  --nstep 500 \
  --model stabilityai/stable-diffusion-2-base \