import multiprocessing
import sys

from models.diffusion import Model
from runners.diffusion import Diffusion

sys.path.append('.')
sys.path.append('./pgm')
sys.path.append('./morphomnist')

from typing import Dict, IO, Optional, Tuple, List
import argparse
import os
import shutil
import logging
import yaml
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
from torch import nn, Tensor
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from pgm.train_pgm import setup_dataloaders, preprocess
from pgm.flow_pgm import MorphoMNISTPGM

from morphomnist.morpho import ImageMorphology

torch.set_printoptions(sci_mode=False)


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


def get_intensity(x, threshold=0.5):
    x = x.detach().cpu().numpy()[:, 0]
    x_min, x_max = x.min(axis=(1, 2), keepdims=True), x.max(axis=(1, 2), keepdims=True)
    mask = (x >= x_min + (x_max - x_min) * threshold)
    return np.array([np.median(i[m]) for i, m in zip(x, mask)])


def img_thickness(img, threshold, scale):
    return ImageMorphology(np.asarray(img), threshold, scale).mean_thickness


def unpack(args):
    return img_thickness(*args)


def get_thickness(x, threshold=0.5, scale=4, pool=None, chunksize=100):
    imgs = x.detach().cpu().numpy()[:, 0]
    args = ((img, threshold, scale) for img in imgs)
    if pool is None:
        gen = map(unpack, args)
    else:
        gen = pool.imap(unpack, args, chunksize=chunksize)
    results = tqdm(gen, total=len(imgs), unit='img', ascii=True)
    return list(results)


def vae_preprocess(pa: Dict[str, Tensor], input_res: int = 32) -> Tensor:
    # concatenate parents and expand to input resolution for vae input
    pa = torch.cat([
        pa[k] if len(pa[k].shape) > 1 else pa[k].unsqueeze(-1) for k in pa.keys()
    ], dim=1)
    return pa[..., None, None].repeat(1, 1, *(input_res,) * 2)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


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


def preprocess_batch(device, batch):
    batch["x"] = (batch["x"].to(device).float() - 127.5) / 127.5  # [-1, 1]
    batch["pa"] = batch["pa"].to(device).float()
    return batch


def cf_epoch(
        model,
        runners,
        pgm,
        predictor,
        dataloaders,
) -> Tuple[Tensor, Tensor, Tensor]:
    def pa_preprocess(pa):
        # concatenate parents and expand to input resolution for vae input
        pa = torch.cat([
            pa[k] if len(pa[k].shape) > 1 else pa[k].unsqueeze(-1) for k in pa.keys()
        ], dim=1)
        return pa

    save_path = './samples/morphomnist_cf/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    pgm.eval()
    predictor.eval()
    dag_vars = list(pgm.variables.keys())
    train_set = copy.deepcopy(dataloaders['train'].dataset.samples)
    loader = tqdm(enumerate(dataloaders['test']), total=len(
        dataloaders['test']), mininterval=0.1)
    model.to(runners.device)

    for _, batch in loader:
        if _ < 120:
            continue
        if _ > 180:
            break
        bs = batch['x'].shape[0]
        batch = preprocess(batch)
        batch = {k: v.to(runners.device) for k, v in batch.items()}
        pa = {k: v for k, v in batch.items() if k != 'x'}

        # 如果不是8或者9，就跳过
        if torch.argmax(pa['digit']) not in [8]:
            continue

        orig_x = batch['x'].to('cpu')
        batch_dos = []
        x_counterfactuals = []

        # 每次只对一个变量进行干预
        for k in dag_vars:
            do = {}
            if k == 'intensity' or k == 'thickness':
                idx = torch.randperm(train_set[k].shape[0])
                do[k] = train_set[k].clone()[idx][:bs]
                do = preprocess(do)
                batch_dos.append(do)
                # infer counterfactual parents
                cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
                _pa = pa_preprocess({k: v.clone() for k, v in pa.items()})
                _cf_pa = pa_preprocess({k: v.clone() for k, v in cf_pa.items()})
                # abduct exogenous noise x_T
                x_T = runners.abduction(model, batch['x'], _pa).to(runners.device)
                # rec = self.sample_image(x_T, model, y=_pa)
                # abduct counterfactual noise x_cf
                x_cf = runners.sample_image(x_T, model, y=_cf_pa).to(runners.device)

                cfs = {'x': (x_cf.clamp(-1, 1) + 1) / 2}
                x_counterfactuals.extend(cfs['x'])
            else:
                for digit in range(10):
                    # 每个数字都进行一次，one-hot编码
                    do[k] = torch.zeros(bs, 10).to(runners.device)
                    do[k][:, digit] = 1
                    do = preprocess(do)
                    batch_dos.append(do.copy())
                    # infer counterfactual parents
                    cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
                    _pa = pa_preprocess({k: v.clone() for k, v in pa.items()})
                    _cf_pa = pa_preprocess({k: v.clone() for k, v in cf_pa.items()})
                    # abduct exogenous noise x_T
                    x_T = runners.abduction(model, batch['x'], _pa).to(runners.device)
                    # rec = self.sample_image(x_T, model, y=_pa)
                    # abduct counterfactual noise x_cf
                    x_cf = runners.sample_image(x_T, model, y=_cf_pa).to(runners.device)

                    cfs = {'x': (x_cf.clamp(-1, 1) + 1) / 2}
                    x_counterfactuals.extend(cfs['x'])
        x_counterfactuals = torch.stack(x_counterfactuals).cpu()

        # save counterfactuals
        save_image(x_counterfactuals, os.path.join(save_path, f'cf_{_}.png'), nrow=12)
        # save pa and do as one file
        with open(os.path.join(save_path, f'pa_do_{_}.txt'), 'w') as f:
            f.write(str(pa))
            f.write('\n')
            f.write(str(batch_dos))

        # save differece
        orig_x = (orig_x.clamp(-1, 1) + 1) / 2

        save_image(orig_x, os.path.join(save_path, f'orig_{_}.png'), nrow=12)

        direct_effect = x_counterfactuals - orig_x
        norm = direct_effect.abs() / direct_effect.abs().max()

        # 根据正负设置不同的颜色，正为红色，负为蓝色
        colored = torch.zeros_like(direct_effect).repeat(1, 3, 1, 1)
        colored[:, :1] = (direct_effect > 0.001).float() * norm
        colored[:, 2:] = (direct_effect < -0.001).float() * norm

        save_image(colored, os.path.join(save_path, f'diff_{_}.png'), nrow=12)


def cf_vis(
        model,
        runners,
        pgm,
        predictor,
        dataloaders,
) -> Tuple[Tensor, Tensor, Tensor]:
    def pa_preprocess(pa):
        # concatenate parents and expand to input resolution for vae input
        pa = torch.cat([
            pa[k] if len(pa[k].shape) > 1 else pa[k].unsqueeze(-1) for k in pa.keys()
        ], dim=1)
        return pa

    save_path = './samples/morphomnist_cf_vis/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    pgm.eval()
    predictor.eval()
    dag_vars = list(pgm.variables.keys())
    train_set = copy.deepcopy(dataloaders['train'].dataset.samples)
    loader = tqdm(enumerate(dataloaders['test']), total=len(
        dataloaders['test']), mininterval=0.1)
    model.to(runners.device)

    seq = [0, 1, 10, 20, 30, 40, 50]
    rev_seq = [10, 20, 30, 40, 49, 50]

    for _, batch in loader:
        if _ < 50:
            continue
        if _ > 90:
            break
        bs = batch['x'].shape[0]
        batch = preprocess(batch)
        batch = {k: v.to(runners.device) for k, v in batch.items()}
        pa = {k: v for k, v in batch.items() if k != 'x'}
        # randomly intervene on a single parent do(pa_k), pa_k ~ p(pa_k)
        do = {}

        for k in dag_vars:
            if torch.rand(1) > 0.5:  # coin flip to intervene on pa_k
                idx = torch.randperm(train_set[k].shape[0])
                do[k] = train_set[k].clone()[idx][:bs]

        # 只有thickness减少才继续
        if 'thickness' not in do or 'thickness' in do and torch.mean(do['thickness']) > torch.mean(pa['thickness']):
            continue

        do = preprocess(do)
        # infer counterfactual parents
        cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
        _pa = pa_preprocess({k: v.clone() for k, v in pa.items()})
        _cf_pa = pa_preprocess({k: v.clone() for k, v in cf_pa.items()})
        # abduct exogenous noise x_T

        # rec = self.sample_image(x_T, model, y=_pa)
        # abduct counterfactual noise x_cf
        strategy = ['uniform', 'quad']
        for s in strategy:
            runners.args.skip_type = s
            x_Ts = runners.abduction(model, batch['x'], _pa, last=False)
            x_Ts[0] = x_Ts[0].to('cpu')
            x_T = x_Ts[-1].to(runners.device)
            x_cfs = runners.sample_image(x_T, model, y=_cf_pa, last=False)[0]
            x_cfs[0] = x_cfs[0].to('cpu')

            # save x_Ts and x_cfs
            x_ts = torch.cat(x_Ts).cpu()
            x_cfs = torch.cat(x_cfs).cpu()
            x_ts = (x_ts.clamp(-1, 1) + 1) / 2
            x_cfs = (x_cfs.clamp(-1, 1) + 1) / 2

            # save sequence
            x_abduct = x_ts[seq]
            x_cf = x_cfs[rev_seq]

            saveimg = torch.cat([x_abduct, x_cf], dim=0)
            save_image(saveimg, os.path.join(save_path, f'cf_{runners.args.skip_type[:4]}_{_}.png'), nrow=13)


def cf_ablation(
        model,
        runners,
        pgm,
        predictor,
        dataloaders,
) -> Tuple[Tensor, Tensor, Tensor]:
    def pa_preprocess(pa):
        # concatenate parents and expand to input resolution for vae input
        pa = torch.cat([
            pa[k] if len(pa[k].shape) > 1 else pa[k].unsqueeze(-1) for k in pa.keys()
        ], dim=1)
        return pa

    save_path = './samples/morphomnist_cf_ablation/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    pgm.eval()
    predictor.eval()
    dag_vars = list(pgm.variables.keys())
    train_set = copy.deepcopy(dataloaders['train'].dataset.samples)
    loader = tqdm(enumerate(dataloaders['test']), total=len(
        dataloaders['test']), mininterval=0.1)
    model.to(runners.device)

    for _, batch in loader:
        if _ > 0:
            break
        bs = batch['x'].shape[0]
        batch = preprocess(batch)
        batch = {k: v.to(runners.device) for k, v in batch.items()}
        pa = {k: v for k, v in batch.items() if k != 'x'}
        # randomly intervene on a single parent do(pa_k), pa_k ~ p(pa_k)
        do = {}

        for k in ['thickness']:
            # if torch.rand(1) > 0.5:  # coin flip to intervene on pa_k
            idx = torch.randperm(train_set[k].shape[0])
            do[k] = train_set[k].clone()[idx][:bs]

        do = preprocess(do)
        # infer counterfactual parents
        cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)

        # directly change the thickness
        cf_pa_undo = pa.copy()
        cf_pa_undo['thickness'] = do['thickness']

        _pa = pa_preprocess({k: v.clone() for k, v in pa.items()})
        _cf_pa = pa_preprocess({k: v.clone() for k, v in cf_pa.items()})
        _cf_pa_undo = pa_preprocess({k: v.clone() for k, v in cf_pa_undo.items()})
        # abduct exogenous noise x_T

        # rec = self.sample_image(x_T, model, y=_pa)
        # abduct counterfactual noise x_cf
        x_T = runners.abduction(model, batch['x'], _pa).to(runners.device)
        x_cf = runners.sample_image(x_T, model, y=_cf_pa)
        x_cf_undo = runners.sample_image(x_T, model, y=_cf_pa_undo)

        # cat
        x_cfs = torch.cat([x_cf, x_cf_undo], dim=0)
        x_cfs = (x_cfs.clamp(-1, 1) + 1) / 2

        save_image(x_cfs, os.path.join(save_path, f'cf_{_}.png'), nrow=2)

        # save orig
        orig_x = (batch['x'].to('cpu').clamp(-1, 1) + 1) / 2
        save_image(orig_x, os.path.join(save_path, f'orig_{_}.png'), nrow=1)

        # save pa and do thickness as one file
        with open(os.path.join(save_path, f'pa_do_{_}.txt'), 'w') as f:
            f.write(str(pa['thickness']))
            f.write('\n')
            f.write(str(do['thickness']))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args, config = parse_args_and_config()

    predictor_path = './exp/logs/morphomnist_ddim/sup_aux/checkpoint.pt'
    print(f'\nLoading predictor checkpoint: {predictor_path}')
    predictor_checkpoint = torch.load(predictor_path)
    predictor_args = Hparams()
    predictor_args.update(predictor_checkpoint['hparams'])
    assert predictor_args.dataset == 'morphomnist'
    predictor = MorphoMNISTPGM(predictor_args).cuda()
    predictor.load_state_dict(predictor_checkpoint['ema_model_state_dict'])
    predictor.to(device)

    pgm_path = './exp/logs/morphomnist_ddim/sup_pgm/checkpoint.pt'
    print(f'\nLoading PGM checkpoint: {pgm_path}')
    pgm_checkpoint = torch.load(pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint['hparams'])
    assert pgm_args.dataset == 'morphomnist'
    pgm = MorphoMNISTPGM(pgm_args).cuda()
    pgm.load_state_dict(pgm_checkpoint['ema_model_state_dict'])
    pgm.to(device)

    file = open(f'./eval.txt', 'a')
    pgm_args.data_dir = './datasets/morphomnist'
    pgm_args.bs = 1
    dataloaders = setup_dataloaders(pgm_args)

    runners = Diffusion(args, config)
    model = Model(config)
    model = runners.load_state_dict(model)
    model.to(device)

    # cf_epoch(model, runners, pgm, predictor, dataloaders)
    # cf_vis(model, runners, pgm, predictor, dataloaders)
    cf_ablation(model, runners, pgm, predictor, dataloaders)