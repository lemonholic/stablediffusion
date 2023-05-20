import argparse
import os
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm
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
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters
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


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


class DDIMSamplerExt(DDIMSampler):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__(model, schedule=schedule, device=device, **kwargs)

    def make_schedule(self, num_last_steps, resume_from, num_extra_steps, ddim_eta=0.0, verbose=True):
        corrected_num_steps = round(num_extra_steps / (1 - resume_from / num_last_steps))
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=corrected_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod",
            to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps",
            sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self, S, batch_size, shape,
        resume_from=None,
        conditioning=None,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.,
        mask=None,
        x0=None,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        dynamic_threshold=None,
        ucg_schedule=None,
        **kwargs
    ):
        if resume_from is None:
            self.make_schedule(1, 0, S, ddim_eta=eta, verbose=verbose)
            self.saved_num_last_steps = S
            # self.saved_conditioning = conditioning
        else:
            self.make_schedule(self.saved_num_last_steps, resume_from, S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        samples, intermediates = self.ddim_sampling(
            conditioning, size,
            skips = 0 if resume_from is None else len(self.ddim_timesteps) - S,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask, x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T if resume_from is None else self.saved_intermediates['x_inter'][resume_from],
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule
        )
        if resume_from is None:
            self.saved_samples, self.saved_intermediates = samples, intermediates
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self, cond, shape,
        skips=0, x_T=None, ddim_use_original_steps=False,
        callback=None, timesteps=None, quantize_denoised=False,
        mask=None, x0=None, img_callback=None, log_every_t=100,
        temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
        unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
        ucg_schedule=None
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps - skips} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            # skip unnecessary steps
            if i < skips:
                continue
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(
                img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised, temperature=temperature,
                noise_dropout=noise_dropout, score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold
            )
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


def sample(opt, sampler, model, start_codes):
    prompt = input("Enter prompt: ")
    if prompt == "exit":
        exit()
    elif prompt == "resume":
        resume = True
    else:
        resume = False

    if resume:
        while (resume_from := int(input("Resume from: "))) not in range(50):
            print("Resume from must be in [0, 49]")
        prompt = input("Enter new prompt: ")
        while (steps := int(input("# of extra steps: "))) not in range(1, 51):
            print("# of extra steps must be in [1, 50]")
    else:
        if prompt in start_codes:
            start_code = start_codes[prompt]
        else:
            start_code = torch.randn(
                (1, opt.C, opt.H // opt.f, opt.W // opt.f),
                device=torch.device("cuda"),
            )
            start_codes[prompt] = start_code
        while (steps := int(input("# of inference steps: "))) not in range(1, 51):
            print("# of steps must be in [1, 50]")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        # Invoke model.cond_stage_model.<forward|encode|constructor>([prompt])
        # This is `ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder` with the default config.
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([""])
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
        samples, intermediates = sampler.sample(
            S=steps,
            resume_from = resume_from if resume else None,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            log_every_t=1,
            eta=opt.ddim_eta,
            x_T = None if resume else start_codes[prompt],
        )
    return samples, intermediates



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
        sampler = DDIMSamplerExt(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    while True:
        samples, intermediates = sample(opt, sampler, model, start_codes)

        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = x_samples[0]
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(sample_path, f"{base_count:05}.png"))

        if opt.store_intermediates:
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
                img.save(os.path.join(sample_path, f"{base_count:05}-{i}.png"))

        base_count += 1


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
