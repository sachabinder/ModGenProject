import os
import torch
import random
from ddps.pipe import StableDiffusionInverse, EulerAncestralDSG
from ddps.dataset import ImageDataset
from ddps.op import SuperResolutionOperator, GaussialBlurOperator, MotionBlurOperator
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from torchvision import transforms
import numpy as np
import argparse
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def fix_seed(seed):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusers-DPS")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2-base",
        help="base diffusion model",
    )
    parser.add_argument("--data", type=str, help="path to image folder")
    parser.add_argument("--out", type=str, help="path to output folder")
    parser.add_argument("--scale", type=float, default=4.8, help="scale of DPS")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    parser.add_argument("--algo", type=str, default="dps", help="algorithm to use")
    parser.add_argument("--operator", type=str, default="srx8", help="operator to use")
    parser.add_argument("--nstep", type=int, default=500, help="num of steps")
    parser.add_argument("--ngpu", type=int, default=1, help="num of gpu")
    parser.add_argument("--rank", type=int, default=0, help="local rank")

    # FreeDOM specific parameters
    # repeat for K steps, in time interval [c1, c2]
    parser.add_argument("--fdm_c1", type=int, default=100, help="c1 of FreeDOM")
    parser.add_argument("--fdm_c2", type=int, default=250, help="c2 of FreeDOM")
    parser.add_argument("--fdm_k", type=int, default=2, help="k of FreeDOM")

    # PSLD specific parameters
    parser.add_argument("--psld_gamma", type=float, default=0.1, help="gamma of PSLD")

    args = parser.parse_args()

    DTYPE = torch.float16

    out_dirs = ["source", "low_res", "recon", "recon_low_res"]
    out_dirs = [os.path.join(args.out, o) for o in out_dirs]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageDataset(root=args.data, transform=test_transforms, return_path=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if args.operator == "srx8":
        f = SuperResolutionOperator([1, 3, 512, 512], 8)
    elif args.operator == "gdb":
        f = GaussialBlurOperator()
    elif args.operator == "mdb":
        f = MotionBlurOperator()
    else:
        raise NotImplementedError

    f = f.to(dtype=DTYPE, device="cuda")

    model_id = args.model
    if args.algo == "dsg":
        scheduler = EulerAncestralDSG.from_pretrained(model_id, subfolder="scheduler", torch_dtype=DTYPE)
    else:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", torch_dtype=DTYPE
        )
    pipe = StableDiffusionInverse.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=DTYPE,
    )
    pipe = pipe.to("cuda")

    for i, (x, x_path) in enumerate(dataloader):
        # skip for multi gpu
        if i % args.ngpu != args.rank:
            continue
        fix_seed(i)
        x_name = x_path[0].split("/")[-1]
        x_name = x_name[:-4] + ".png"
        x = x.to(dtype=DTYPE, device="cuda")
        y = f(x, reset=True)
        image, _ = pipe(
            f=f,
            y=y,
            algo=args.algo,
            scale=args.scale,
            prompt=args.prompt,
            height=512,
            width=512,
            num_inference_steps=args.nstep,
            guidance_scale=0.0,
            output_type="pt",
            return_dict=False,
            fdm_c1=args.fdm_c1,
            fdm_c2=args.fdm_c2,
            fdm_k=args.fdm_k,
            psld_gamma=args.psld_gamma,
        )
        x_hat = image * 2.0 - 1.0
        y_hat = f(x_hat)

        out_tensors = [x, y, x_hat, y_hat]

        for i in range(4):
            save_image(
                out_tensors[i],
                os.path.join(out_dirs[i], x_name),
                normalize=True,
                value_range=(-1, 1),
            )