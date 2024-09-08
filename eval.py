import sys

sys.path.append('.')
sys.path.append('./pgm')
sys.path.append('./morphomnist')
from typing import Dict, IO, Optional, Tuple, List
import argparse
import os
import shutil
import logging
import yaml

import gc
import copy
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import multiprocessing
import numpy as np
from tqdm import tqdm

from runners.diffusion import Diffusion
from models.diffusion import Model
from models.ema import EMAHelper

from pgm.train_pgm import setup_dataloaders, preprocess
from pgm.flow_pgm import MorphoMNISTPGM

from morphomnist.morpho import ImageMorphology


# Refer to https://github.com/dccastro/Morpho-MNIST for details on Morpho-MNIST

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


@torch.no_grad()
def cf_epoch(
        model,
        runners,
        pgm,
        predictor,
        dataloaders,
        do_pa=None,
        te_cf=False
) -> Tuple[Tensor, Tensor, Tensor]:
    def pa_preprocess(pa):
        # concatenate parents and expand to input resolution for vae input
        pa = torch.cat([
            pa[k] if len(pa[k].shape) > 1 else pa[k].unsqueeze(-1) for k in pa.keys()
        ], dim=1)
        return pa

    model.eval()
    pgm.eval()
    predictor.eval()
    dag_vars = list(pgm.variables.keys())
    preds = {k: [] for k in dag_vars}
    targets = {k: [] for k in dag_vars}
    x_counterfactuals = []
    train_set = copy.deepcopy(dataloaders['train'].dataset.samples)
    loader = tqdm(enumerate(dataloaders['test']), total=len(
        dataloaders['test']), mininterval=0.1)

    model.to(runners.device)

    for _, batch in loader:
        bs = batch['x'].shape[0]
        batch = preprocess(batch)
        batch = {k: v.to(runners.device) for k, v in batch.items()}
        pa = {k: v for k, v in batch.items() if k != 'x'}
        # randomly intervene on a single parent do(pa_k), pa_k ~ p(pa_k)
        do = {}
        if do_pa is not None:
            idx = torch.randperm(train_set[do_pa].shape[0])
            do[do_pa] = train_set[do_pa].clone()[idx][:bs]
        else:  # random interventions
            while not do:
                for k in dag_vars:
                    if torch.rand(1) > 0.5:  # coin flip to intervene on pa_k
                        idx = torch.randperm(train_set[k].shape[0])
                        do[k] = train_set[k].clone()[idx][:bs]
        do = preprocess(do)
        # infer counterfactual parents
        cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
        _pa = pa_preprocess({k: v.clone() for k, v in pa.items()})
        _cf_pa = pa_preprocess({k: v.clone() for k, v in cf_pa.items()})
        # abduct exogenous noise x_T
        x_T = runners.abduction(model, batch['x'], _pa).to(runners.device)
        # rec = self.sample_image(x_T, model, y=_pa)
        # abduct counterfactual noise x_cf
        x_cf = runners.sample_image(x_T, model, y=_cf_pa).to(runners.device)

        cfs = {'x': torch.clamp(x_cf, min=-1, max=1)}
        cfs.update(cf_pa)
        x_counterfactuals.extend(cfs['x'])
        # predict labels of inferred counterfactuals
        preds_cf = predictor.predict(**cfs)
        for k, v in preds_cf.items():
            preds[k].extend(v)
        # targets are the interventions and/or counterfactual parents
        for k in targets.keys():
            t_k = do[k].clone() if k in do.keys() else cfs[k].clone()
            targets[k].extend(t_k)
    for k, v in targets.items():
        targets[k] = torch.stack(v).squeeze().cpu()
        preds[k] = torch.stack(preds[k]).squeeze().cpu()
    x_counterfactuals = torch.stack(x_counterfactuals).cpu()
    return targets, preds, x_counterfactuals


def eval_cf_loop(
        model, runners, pgm, predictor, dataloaders, file, seeds, total_effect=False
):
    for do_pa in ['thickness', 'intensity', 'digit', None]:  # "None" is for random interventions
        acc_runs = []
        mae_runs = {
            'thickness': {'predicted': [], 'measured': []},
            'intensity': {'predicted': [], 'measured': []}
        }

        for seed in seeds:
            print(f'do({(do_pa if do_pa is not None else "random")}), seed {seed}:')
            # assert vae.cond_prior if total_effect else True
            targets, preds, x_cfs = cf_epoch(model, runners, pgm, predictor, dataloaders, do_pa, total_effect)
            acc = (targets['digit'].argmax(-1).numpy() == preds['digit'].argmax(-1).numpy()).mean()
            print(f'predicted digit acc:', acc)
            # evaluate inferred cfs using true causal mechanisms
            measured = {}
            measured['intensity'] = torch.tensor(get_intensity((x_cfs + 1.0) * 127.5))
            with multiprocessing.Pool() as pool:
                measured['thickness'] = torch.tensor(get_thickness((x_cfs + 1.0) * 127.5, pool=pool, chunksize=250))

            mae = {'thickness': {}, 'intensity': {}}
            for k in ['thickness', 'intensity']:
                min_max = dataloaders['train'].dataset.min_max[k]
                _min, _max = min_max[0], min_max[1]
                preds_k = ((preds[k] + 1) / 2) * (_max - _min) + _min
                targets_k = ((targets[k] + 1) / 2) * (_max - _min) + _min
                mae[k]['predicted'] = (targets_k - preds_k).abs().mean().item()
                mae[k]['measured'] = (targets_k - measured[k]).abs().mean().item()
                print(f'predicted {k} mae:', mae[k]['predicted'])
                print(f'measured {k} mae:', mae[k]['measured'])

            acc_runs.append(acc)
            for k in ['thickness', 'intensity']:
                mae_runs[k]['predicted'].append(mae[k]['predicted'])
                mae_runs[k]['measured'].append(mae[k]['measured'])

            file.write(
                f'\ndo({(do_pa if do_pa is not None else "random")}) | digit acc: {acc}, ' +
                f'thickness mae (predicted): {mae["thickness"]["predicted"]}, ' +
                f'thickness mae (measured): {mae["thickness"]["measured"]}, ' +
                f'intensity mae (predicted): {mae["intensity"]["predicted"]}, ' +
                f'intensity mae (measured): {mae["intensity"]["measured"]} | seed {seed}'
            )
            file.flush()
            gc.collect()

        v = 'Total effect: ' + str(total_effect)
        file.write(
            f'\n{(v if config.model.cond_prior else "")}\n' +
            f'digit acc | mean: {np.array(acc_runs).mean()} - std: {np.array(acc_runs).std()}\n' +
            f'thickness mae (predicted) | mean: {np.array(mae_runs["thickness"]["predicted"]).mean()} - std: {np.array(mae_runs["thickness"]["predicted"]).std()}\n' +
            f'thickness mae (measured) | mean: {np.array(mae_runs["thickness"]["measured"]).mean()} - std: {np.array(mae_runs["thickness"]["measured"]).std()}\n' +
            f'intensity mae (predicted) | mean: {np.array(mae_runs["intensity"]["predicted"]).mean()} - std: {np.array(mae_runs["intensity"]["predicted"]).std()}\n' +
            f'intensity mae (measured) | mean: {np.array(mae_runs["intensity"]["measured"]).mean()} - std: {np.array(mae_runs["intensity"]["measured"]).std()}\n'
        )
        file.flush()
    return


if __name__ == '__main__':
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
    pgm_args.bs = 32
    dataloaders = setup_dataloaders(pgm_args)

    runners = Diffusion(args, config)
    model = Model(config)
    model = runners.load_state_dict(model)
    model.to(device)

    eval_cf_loop(model, runners, pgm, predictor, dataloaders, file, seeds=[1])
    file.close()
