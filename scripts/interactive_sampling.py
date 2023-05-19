import argparse
import os
from contextlib import nullcontext
from itertools import islice

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    print(f'Model loaded: {model.__class__.__name__}')
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action="store_true",
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = scale * eps(x, cond) + (1 - scale) * eps(x, empty)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--store_intermediates",
        action="store_true",
        help="store intermediate samples",
    )
    return parser.parse_args()


def main(opt):
    seed_everything(opt.seed)
    start_codes = {}

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda")
    model: LatentDiffusion = load_model_from_config(config, f"{opt.ckpt}", device)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    precision_scope = (
        autocast if opt.precision == "autocast" else nullcontext
    )

    while True:
        prompt = input('Enter prompt: ')
        if prompt == 'exit':
            break
        while (steps := int(input('# of inference steps: '))) not in range(1, 51):
            print('# of steps must be in [1, 50]')

        if prompt in start_codes:
            start_code = start_codes[prompt]
        else:
            start_code = torch.randn((1, opt.C, opt.H // opt.f, opt.W // opt.f), device=device)
            start_codes[prompt] = start_code

        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            # Invoke model.cond_stage_model.<forward|encode|constructor>([prompt])
            # This is `ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder` with the default config.
            c = model.get_learned_conditioning([prompt])
            uc = model.get_learned_conditioning([""])
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples, intermediates = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=1,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                log_every_t=1,
                eta=opt.ddim_eta,
                x_T=start_code,
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = x_samples[0]
            x_sample = 255.0 * rearrange(
                x_sample.cpu().numpy(), "c h w -> h w c"
            )
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))

            if opt.store_intermediates:
                for i, int_t in enumerate(intermediates['x_inter']):
                    int_samples = model.decode_first_stage(int_t)
                    int_samples = torch.clamp((int_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    int_sample = int_samples[0]
                    int_sample = 255.0 * rearrange(
                        int_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    img = Image.fromarray(int_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}-{i}.png"))

            base_count += 1


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
