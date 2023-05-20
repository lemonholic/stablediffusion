import argparse
import os
from contextlib import nullcontext
from shutil import rmtree
from time import time

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast

from ddim_ext import DDIMSamplerExt
from ldm.models.diffusion.ddpm import LatentDiffusion
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
    print(f"Model loaded: {model.__class__.__name__}")
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
        default="outputs",
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
        "--prompt_file",
        type=str,
        help="path to file with prompts",
        required=True,
    )
    return parser.parse_args()


def sample(opt, sampler, model, tc_config, start_codes):
    if tc_config.type not in ['generate', 'resume']:
        raise NotImplementedError(f'Testcase type "{tc_config.type}" is not supported. Check if the prompt file is correct.')

    if tc_config.type == 'generate':
        if tc_config.prompt in start_codes:
            start_code = start_codes[tc_config.prompt]
        else:
            start_code = torch.randn(
                (1, opt.C, opt.H // opt.f, opt.W // opt.f),
                device=torch.device("cuda"),
            )
            start_codes[tc_config.prompt] = start_code

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        c = model.get_learned_conditioning([tc_config.prompt])
        uc = model.get_learned_conditioning([""])
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
        time_start = time()
        samples, intermediates = sampler.sample(
            tc_config=tc_config,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            log_every_t=1,
            eta=opt.ddim_eta,
            x_T = None if tc_config.type == 'resume' else start_codes[tc_config.prompt],
        )
        time_end = time()
    return samples, intermediates, time_end - time_start



def main(opt):
    seed_everything(opt.seed)
    start_codes = {}

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda")
    model: LatentDiffusion = load_model_from_config(config, f"{opt.ckpt}", device)
    sampler = DDIMSamplerExt(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    prompt_config = OmegaConf.load(opt.prompt_file)

    for testcase_name in prompt_config:
        testcase_dir = os.path.join(sample_path, testcase_name)
        try:
            os.makedirs(testcase_dir, exist_ok=False)
        except OSError:
            rmtree(testcase_dir)
            os.makedirs(testcase_dir, exist_ok=False)
        tc_config = prompt_config[testcase_name]
        samples, intermediates, gen_time = sample(opt, sampler, model, tc_config, start_codes)
        print(f"Sampling took {gen_time:.2f} seconds.")

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = x_samples[0]
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(testcase_dir, f"result.png"))

        if 'save_intermediates' in tc_config and tc_config.save_intermediates == False:
            pass
        else:
            for i, int_t in enumerate(intermediates["x_inter"]):
                int_samples = model.decode_first_stage(int_t)
                int_samples = torch.clamp(
                    (int_samples + 1.0) / 2.0, min=0.0, max=1.0
                )
                int_sample = int_samples[0]
                int_sample = 255.0 * rearrange(
                    int_sample.cpu().numpy(), "c h w -> h w c"
                )
                img = Image.fromarray(int_sample.astype(np.uint8))
                if tc_config.type == 'generate':
                    img.save(os.path.join(testcase_dir, f"{i:04d}.png"))
                else:
                    img.save(os.path.join(testcase_dir, f"{i + tc_config.resume_from:04d}.png"))

        with open(os.path.join(testcase_dir, "result.txt"), "wb") as f:
            f.write(f'Sampling took {gen_time:.2f} seconds.'.encode())


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
