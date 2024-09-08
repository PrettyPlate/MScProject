import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import random

import torch
import numpy as np
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
from torchvision import utils as tvu

from runners.diffusion import Diffusion
from models.diffusion import Model

from hps import add_arguments, setup_hparams
from train_setup import setup_datasets

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
             "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def preprocess_batch(device, batch):
    batch["x"] = (batch["x"].to(device).float() - 127.5) / 127.5  # [-1, 1]
    batch["pa"] = batch["pa"].to(device).float()
    return batch


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


def load_vae(vae_path):
    print(f'\nLoading VAE checkpoint: {vae_path}')
    vae_checkpoint = torch.load(vae_path)
    vae_args = Hparams()
    vae_args.update(vae_checkpoint['hparams'])
    vae_args.data_dir = 'your dataset dir here'

    # init model
    assert vae_args.hps == 'morphomnist'
    if not hasattr(vae_args, 'vae'):
        vae_args.vae = 'simple'

    if vae_args.vae == 'hierarchical':
        from src.vae import HVAE
        vae = HVAE(vae_args).cuda()
    elif vae_args.vae == 'simple':
        from src.simple_vae import VAE
        vae = VAE(vae_args).cuda()
    else:
        NotImplementedError
    vae.load_state_dict(vae_checkpoint['ema_model_state_dict'])
    return vae, vae_args


def vae_preprocess(pa, input_res: int = 32):
    # concatenate parents and expand to input resolution for vae input
    pa = torch.cat([
        pa[k] if len(pa[k].shape) > 1 else pa[k].unsqueeze(-1) for k in pa.keys()
    ], dim=1)
    return pa[..., None, None].repeat(1, 1, *(input_res,) * 2)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    cf_args = setup_hparams(parser)

    datasets = setup_datasets(cf_args)

    # save_dir = os.path.join("./datasets/morphomnist_2048")
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # for i, batch in enumerate(datasets["train"]):
    #     if i > 2048:
    #         break
    #     x = batch["x"]
    #     x = x.to(dtype=torch.float)
    #     tvu.save_image(
    #         x, os.path.join(save_dir, f"train_{i}.png"), nrow=1
    #     )

    runner = Diffusion(args, config, dataset=datasets)
    model = Model(config)
    model = runner.load_state_dict(model)
    model.to(device)

    vae_path = 'exp/logs/hvae/checkpoint.pt'
    vae, vae_args = load_vae(vae_path)

    fixed_val_num = 6*25
    val_loader = DataLoader(
        datasets["valid"], batch_size=fixed_val_num, shuffle=False, num_workers=0, pin_memory=True
    )
    save_dir = os.path.join("samples")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # c = 0
    # for i, batch in enumerate(val_loader):
    #     batch = preprocess_batch(device, batch)
    #     pa = {k: v for k, v in batch.items() if k != 'x'}
    #     _pa = vae_preprocess({k: v.clone() for k, v in pa.items()})
    #     y = batch["pa"].to(device)
    #
    #     model.eval()
    #     vae.eval()
    #     with torch.no_grad():
    #         x = vae.sample(parents=_pa)[0]
    #         x_vae = (torch.clamp(x, min=-1, max=1) + 1) / 2
    #
    #         noise = torch.randn(
    #             fixed_val_num,
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=device,
    #             requires_grad=False
    #         )
    #         x = runner.sample_image(noise, model, y=y)
    #         x_ddim = (x.clamp(min=-1, max=1) + 1) / 2
    #
    #         for i in range(fixed_val_num):
    #             tvu.save_image(
    #                 x_vae[i], os.path.join(save_dir, 'vae_fid', f"{c + 1}.png")
    #             )
    #             tvu.save_image(
    #                 x_ddim[i], os.path.join(save_dir, 'ddim_fid', f"{c + 1}.png")
    #             )
    #             c += 1

    fixed_val_noise = torch.randn(
        fixed_val_num,
        config.data.channels,
        config.data.image_size,
        config.data.image_size,
        device=device,
        requires_grad=False
    )

    for i, fixed_val_batch in enumerate(val_loader):
        if i != 1:
            continue
        fixed_val_batch = preprocess_batch(device, fixed_val_batch)
        pa = {k: v for k, v in fixed_val_batch.items() if k != 'x'}
        _pa = vae_preprocess({k: v.clone() for k, v in pa.items()})
        fixed_val_y = fixed_val_batch["pa"].to(device)
        fixed_val_x = fixed_val_batch["x"].to(device)

        save_dir = os.path.join("samples")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fixed_val_x = (fixed_val_x.clamp(min=-1, max=1) + 1) / 2
        tvu.save_image(
            fixed_val_x, os.path.join(save_dir, f"fixed_val_x_{i + 1}.png"), nrow=6
        )
        model.eval()
        vae.eval()
        with torch.no_grad():
            x = vae.sample(parents=_pa)[0]
            x = (torch.clamp(x, min=-1, max=1) + 1) / 2
            tvu.save_image(
                x, os.path.join(save_dir, f"{i + 1}_{vae_args.vae}.png"), nrow=6
            )

            # runner.args.timesteps = 50
            runner.args.skip_type = "uniform"
            x = runner.sample_image(fixed_val_noise, model, y=fixed_val_y)
            x = (x.clamp(min=-1, max=1) + 1) / 2

            tvu.save_image(
                x, os.path.join(save_dir, f"{i + 1}_{runner.args.timesteps}_{runner.args.skip_type}.png"), nrow=6
            )

            runner.args.skip_type = "quad"
            x = runner.sample_image(fixed_val_noise, model, y=fixed_val_y)
            x = (x.clamp(min=-1, max=1) + 1) / 2

            tvu.save_image(
                x, os.path.join(save_dir, f"{i + 1}_{runner.args.timesteps}_{runner.args.skip_type}.png"), nrow=6
            )

            # runner.args.timesteps = 1000
            # runner.args.skip_type = "uniform"
            # x = runner.sample_image(fixed_val_noise, model, y=fixed_val_y)
            # x = (x.clamp(min=-1, max=1) + 1) / 2
            #
            # tvu.save_image(
            #     x, os.path.join(save_dir, f"{i + 1}_{runner.args.timesteps}.png"), nrow=6
            # )

    return 0


if __name__ == "__main__":
    sys.exit(main())
