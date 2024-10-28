# main.py

import os
import torch
import random
import numpy as np
from ddps.pipe import StableDiffusionInverse, EulerAncestralDSG
from ddps.dataset import ImageDataset
from ddps.op import SuperResolutionOperator, GaussialBlurOperator, MotionBlurOperator
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from torchvision import transforms
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def fix_seed(seed):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)

def generate_images(model, data_path, out_path, scale, algo, operator, nstep, fdm_c1, fdm_c2, fdm_k, psld_gamma, prompt=""):
    DTYPE = torch.float32

    out_dirs = ["source", "low_res", "recon", "recon_low_res"]
    out_dirs = [os.path.join(out_path, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=data_path, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if operator == "srx8":
        f = SuperResolutionOperator([1, 3, 512, 512], 8)
    elif operator == "gdb":
        f = GaussialBlurOperator()
    elif operator == "mdb":
        f = MotionBlurOperator()
    else:
        raise NotImplementedError

    f = f.to(dtype=DTYPE, device="cuda")

    model_id = model
    if algo == "dsg":
        scheduler = EulerAncestralDSG.from_pretrained(model_id, subfolder="scheduler")
    else:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
    pipe = StableDiffusionInverse.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=DTYPE
    )
    pipe = pipe.to("cuda")

    for i, (x, x_path) in enumerate(dataloader):
        fix_seed(i)
        x_name = os.path.basename(x_path[0]).replace(".jpg", ".png").replace(".jpeg", ".png")
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x, reset=True)
        image, _ = pipe(
            f=f,
            y=y,
            algo=algo,
            scale=scale,
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=nstep,
            guidance_scale=0.0,
            output_type="pt",
            return_dict=False,
            fdm_c1=fdm_c1,
            fdm_c2=fdm_c2,
            fdm_k=fdm_k,
            psld_gamma=psld_gamma,
        )
        x_hat = image * 2.0 - 1.0
        y_hat = f(x_hat)

        out_tensors = [x, y, x_hat, y_hat]

        for j, tensor in enumerate(out_tensors):
            save_image(tensor, os.path.join(out_dirs[j], x_name), normalize=True, value_range=(-1, 1))

    print("Image generation complete!")