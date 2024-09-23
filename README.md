## diffusers-Diffusion-Posterior-Sampling
* Huggingface diffusers flavor Diffusion Posterior Sampling algorithms
    * An alternative to Stability-AI/stablediffusion flavor of Diffusion Posterior Sampling
    * Everything in a clean Pipeline, avoid ugly hack of Stability-AI/stablediffusion code
    * Faster inference with diffusers optimized pipeline and multiple GPU support
    * Seamless use all Stable Diffusion models and solvers available in Huggingface


## Supported DPS Algorithms
* DPS
    * Diffusion Posterior Sampling for General Noisy Inverse Problems, https://arxiv.org/abs/2209.14687
    * Code: https://github.com/DPS2022/diffusion-posterior-sampling
* FreeDOM
    * Training-Free Energy-Guided Conditional Diffusion Model, https://arxiv.org/abs/2303.09833
    * Code: https://github.com/vvictoryuki/FreeDoM
* DSG
    * Guidance with Spherical Gaussian Constraint for Conditional Diffusion, https://arxiv.org/abs/2402.03201
    * Code: https://github.com/LingxiaoYang2023/DSG2024
* PSLD:
    * Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models, https://arxiv.org/abs/2307.00619
    * Code: https://github.com/LituRout/PSLD


## Supported Operators
* Downsampling
* Gaussian Blurring
* Motion Blurring


## Usage and Examples
* Run DPS for operator srx8, 500 steps
    ```bash
    python main.py --data ./example_imgs --out ./example_outputs/dps/srx8 --scale 4.8 --algo dps --operator srx8 --nstep 500 --model stabilityai/stable-diffusion-2-base
    ```
* Run DSG for operator gaussian deblur, 500 steps
    ```bash
    python main.py --data ./example_imgs --out ./example_outputs/dsg/gdb --scale 0.02 --algo dsg --operator gdb --nstep 500 --model stabilityai/stable-diffusion-2-base
    ```
* Run FreeDOM for operator motion deblur, 500 steps
    ```bash
    python main.py --data ./example_imgs --out ./example_outputs/fdm/mdb --scale 1.2 --algo fdm --operator mdb --nstep 500 --model stabilityai/stable-diffusion-2-base
    ```
* In case you are in China, use an extra environment variable for Huggingface mirror, for example
    ```bash
    HF_ENDPOINT=https://hf-mirror.com python main.py --data ./example_imgs --out ./example_outputs/dps --scale 4.8 --algo dps --operator srx8 --nstep 500 --model stabilityai/stable-diffusion-2-base
    ```
* Multi GPU
    * pretty foolish way but works. No need to worry about randomness as we fix seed.
    ```bash
    python main.py --ngpu=2 --rank=0 ...
    python main.py --ngpu=2 --rank=1 ...
    ```

## Run a full benchmark
* Download the first 1000 image of ImageNet validation dataset:
    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/xutongda/ImageNet_val1k_512
    ```
* Current Results for stabilityai/stable-diffusion-2-base and 500 steps

    |   |-|SRx8|-|-|Motion Deblur|-|
    |---|:-:|:-:|:-:|:-:|:-:|:-:|
    |   |PSNR|LPIPS|FID|PSNR|LPIPS|FID|
    |DPS| 22.33 | 0.4137 | 58.48 |23.05|0.4267 | 59.31 |
    |PSLD| 22.28 | 0.4163 | 59.08 | 23.06 | 0.4305 | 60.73 |
    |FreeDOM| 22.64 | 0.3961 | 52.94 | 23.33 | 0.4104 | 54.26 |
    |DSG| 23.15 | 0.3912 | 51.15 | 23.70 | 0.3977 | 52.01 |

## Dependency
* (have tested with) pytorch==2.1.0, diffusers==0.30.0, transformers==4.37.2
* Usually ok to install manually, but if you are lazy, use:
    ```bash
    conda env create --name ddps --file=environments.yml
    ```

## Contributions
* We welcome new algorithms / operators / pipelines / test results
* We use ruff format and ruff check for code quality, make sure you pass the check before PR.