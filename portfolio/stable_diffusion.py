from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import googletrans
from googletrans import Translator
from datetime import datetime
import argparse, os, sys, glob
import PIL
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

class Txt2img:
    def txt2img_func(txt, current_user):
        now = datetime.now()
        # load safety model
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        def numpy_to_pil(images):
            """
            Convert a numpy image or a batch of images to a PIL image.
            """
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]

            return pil_images

        def load_model_from_config(config, ckpt, verbose=False):
            # print(f"Loading model from {ckpt}")
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

            model.cuda()
            model.eval()
            return model

        def put_watermark(img, wm_encoder=None):
            if wm_encoder is not None:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img = wm_encoder.encode(img, 'dwtDct')
                img = Image.fromarray(img[:, :, ::-1])
            return img

        def load_replacement(x):
            try:
                # hwc = x.shape
                # y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
                # y = (np.array(y) / 255.0).astype(x.dtype)
                # assert y.shape == x.shape
                # return y
                return x
            except Exception:
                return x

        def check_safety(x_image):
            safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
            x_checked_image, has_nsfw_concept = safety_checker(images=x_image,
                                                               clip_input=safety_checker_input.pixel_values)
            assert x_checked_image.shape[0] == len(has_nsfw_concept)
            for i in range(len(has_nsfw_concept)):
                if has_nsfw_concept[i]:
                    x_checked_image[i] = load_replacement(x_checked_image[i])
            return x_checked_image, has_nsfw_concept

        def kor_2_eng(prompt, lang='en'):
            translator = Translator()
            prompt = translator.translate(prompt, src='ko', dest=lang).text
            return prompt

        def eng_2_kor(prompt, lang='ko'):
            translator = Translator()
            prompt = translator.translate(prompt, src='en', dest=lang).text
            return prompt

        parser = argparse.ArgumentParser()

        prompt = txt

        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default=f"./media/{current_user}" 
        )
        parser.add_argument(
            "--skip_grid",
            action='store_true',
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )
        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save individual samples. For speed measurements.",
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--dpm_solver",
            action='store_true',
            help="use dpm_solver sampling",
        )
        parser.add_argument(
            "--laion400m",
            action='store_true',
            help="uses the LAION400M model",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=2,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=256,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=256,
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
            help="downsampling factor",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=1,
            help="how many samples to produce for each given prompt. A.k.a. batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=7.5,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="stable-diffusion-main/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
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
            default="autocast"
        )
        opt, _ = parser.parse_known_args()
        print(opt)

        if opt.laion400m:
            pass

        seed_everything(opt.seed)

        config = OmegaConf.load(f"{opt.config}")
        print('-----------------------')
        print(config)
        print('-----------------------')
        model = load_model_from_config(config, f"{opt.ckpt}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if opt.dpm_solver:
            sampler = DPMSolverSampler(model)
        elif opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = kor_2_eng(prompt)

        else:
            pass
        prompt_list = []
        prompt_list = prompt.split('.')
        new_path_list = []
        for prompt in prompt_list:
            prompt = eng_2_kor(prompt)
            prompt = kor_2_eng(prompt) #+ ',오일 페인팅 스타일'
            data = [batch_size * prompt]
            username_path = outpath #####
            os.makedirs(username_path, exist_ok=True)

            contents_count = len(os.listdir(username_path)) + 1
            contents_path = os.path.join(username_path, current_user + '_' + str(contents_count))

            os.makedirs(contents_path, exist_ok=True)
            base_count = len(os.listdir(contents_path))

            start_code = None
            if opt.fixed_code:
                start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        for n in trange(opt.n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if opt.scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = model.get_learned_conditioning(prompts)
                                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                 conditioning=c,
                                                                 batch_size=opt.n_samples,
                                                                 shape=shape,
                                                                 verbose=False,
                                                                 unconditional_guidance_scale=opt.scale,
                                                                 unconditional_conditioning=uc,
                                                                 eta=opt.ddim_eta,
                                                                 x_T=start_code)

                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                                if not opt.skip_save:
                                    dt_time = now.strftime('%Y%m%d%H%M%S')
                                    for x_sample in x_checked_image_torch:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        img = Image.fromarray(x_sample.astype(np.uint8))
                                        img = put_watermark(img, wm_encoder)
                                        img.save(os.path.join(contents_path, f"{dt_time}_{base_count}.png"))
                                        new_path = os.path.join(contents_path, f"{dt_time}_{base_count}.png")
                                        new_path = new_path[1:]
                                        new_path_list.append(new_path)
                                        base_count += 1

                                if not opt.skip_grid:
                                    pass

                        if not opt.skip_grid:
                            pass
                        toc = time.time()

        return new_path_list

class Img2img:
    def img2img_func(input_img, txt,current_user):

        input_img = input_img
        txt = txt
        now = datetime.now()

        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())

        def load_model_from_config(config, ckpt, verbose=False):
            # print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                # print(f"Global Step: {pl_sd['global_step']}")
                pass
            sd = pl_sd["state_dict"]
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                # print("missing keys:")
                # print(m)
                pass
            if len(u) > 0 and verbose:
                # print("unexpected keys:")
                # print(u)
                pass

            model.cuda()
            model.eval()
            return model

        def load_img(path):
            image = Image.open(path).convert("RGB")
            w, h = image.size
            # print(f"loaded input image of size ({w}, {h}) from {path}")
            w = 512
            h = 512
            print(f"Resized image of size ({w}, {h}) from {path}")
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            image = image.resize((w, h), resample=PIL.Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return 2. * image - 1.

        def kor_2_eng(prompt, lang='en'):
            translator = Translator()
            prompt = translator.translate(prompt, src='ko', dest=lang).text
            return prompt

        init_img = str(input_img)
        print('------------------------')
        print(init_img)
        print(type(init_img))
        print('------------------------')
        prompt = txt

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default=f"./media/{current_user}"
        )

        parser.add_argument(
            "--skip_grid",
            action='store_true',
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )

        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save indiviual samples. For speed measurements.",
        )

        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )

        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across all samples ",
        )

        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
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
            "--n_samples",
            type=int,
            default=2,
            help="how many samples to produce for each given prompt. A.k.a batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=5.0,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )

        parser.add_argument(
            "--strength",
            type=float,
            default=0.75,
            help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="stable-diffusion-main/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
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
            default="autocast"
        )

        opt, _ = parser.parse_known_args()
        seed_everything(opt.seed)

        config = OmegaConf.load(f"{opt.config}")
        model = load_model_from_config(config, f"{opt.ckpt}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if opt.plms:
            raise NotImplementedError("PLMS sampler not (yet) supported")
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = kor_2_eng(prompt)
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        username_path = os.path.join(outpath, current_user)  #####
        os.makedirs(username_path, exist_ok=True)

        contents_count = len(os.listdir(username_path)) + 1
        contents_path = os.path.join(username_path, current_user + '_' + str(contents_count))

        os.makedirs(contents_path, exist_ok=True)
        base_count = len(os.listdir(contents_path))

        #assert os.path.isfile(init_img)
        init_image = load_img(init_img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc, )

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                dt_time = now.strftime('%Y%m%d%H%M%S')
                                new_path_list = []
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(contents_path, f"{dt_time}_{base_count}.png"))
                                    new_path_list.append(os.path.join(contents_path, f"{dt_time}_{base_count}.png"))
                                    base_count += 1

                    if not opt.skip_grid:
                        pass

        return new_path_list