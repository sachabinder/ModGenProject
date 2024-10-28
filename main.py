# main.py
import os
import torch
from ddps.pipe import StableDiffusionInverse, EulerAncestralDSG
from ddps.dataset import ImageDataset
from ddps.op import SuperResolutionOperator, GaussialBlurOperator, MotionBlurOperator
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from torchvision import transforms
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_pipeline(input_image_path, output_folder, model="stabilityai/stable-diffusion-2-base", scale=4.8, algo="dps", 
                 operator="srx8", nstep=500, fdm_c1=100, fdm_c2=250, fdm_k=2, psld_gamma=0.1):
    
    DTYPE = torch.float32

    # Create output subdirectories
    out_dirs = ["source", "low_res", "recon", "recon_low_res"]
    out_dirs = [os.path.join(output_folder, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = ImageDataset(root=input_image_path, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Select operator
    if operator == "srx8":
        f = SuperResolutionOperator([1, 3, 512, 512], 8)
    elif operator == "gdb":
        f = GaussialBlurOperator()
    elif operator == "mdb":
        f = MotionBlurOperator()
    else:
        raise NotImplementedError

    f = f.to(dtype=DTYPE, device="cuda")

    # Initialize model and scheduler
    if algo == "dsg":
        scheduler = EulerAncestralDSG.from_pretrained(model, subfolder="scheduler")
    else:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    
    pipe = StableDiffusionInverse.from_pretrained(model, scheduler=scheduler, torch_dtype=DTYPE)
    pipe = pipe.to("cuda")

    # Process each image in the dataloader
    for i, (x, x_path) in enumerate(dataloader):
        x_name = x_path[0].split("/")[-1]
        x_name = x_name[:-4] + ".png"
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x, reset=True)
        image, _ = pipe(
            f=f,
            y=y,
            algo=algo,
            scale=scale,
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

        # Save output images
        out_tensors = [x, y, x_hat, y_hat]
        for j, out_tensor in enumerate(out_tensors):
            save_image(out_tensor, os.path.join(out_dirs[j], x_name), normalize=True, value_range=(-1, 1))

    return out_dirs