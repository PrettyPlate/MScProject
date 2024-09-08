import os
import logging
import time
import glob
import gc
import copy

from tqdm import tqdm

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from functions.ckpt_util import get_ckpt_path
from functions.denoising import abduction_steps

import torchvision.utils as tvu


def preprocess_batch(device, batch):
    batch["x"] = (batch["x"].to(device).float() - 127.5) / 127.5  # [-1, 1]
    batch["pa"] = batch["pa"].to(device).float()
    return batch


def get_attr_max_min(attr: str):
    # some ukbb dataset (max, min) stats
    if attr == "age":
        return 73, 44
    elif attr == "brain_volume":
        return 1629520, 841919
    elif attr == "ventricle_volume":
        return 157075, 7613.27001953125
    else:
        NotImplementedError


def preprocess(
        batch, dataset: str = "ukbb", split: str = "l"
):
    if "x" in batch.keys():
        batch["x"] = (batch["x"].float().cuda() - 127.5) / 127.5  # [-1,1]
    # for all other variables except x
    not_x = [k for k in batch.keys() if k != "x"]
    for k in not_x:
        if split == "u":  # unlabelled
            batch[k] = None
        elif split == "l":  # labelled
            batch[k] = batch[k].float().cuda()
            if len(batch[k].shape) < 2:
                batch[k] = batch[k].unsqueeze(-1)
        else:
            NotImplementedError
    if "ukbb" in dataset:
        for k in not_x:
            if k in ["age", "brain_volume", "ventricle_volume"]:
                k_max, k_min = get_attr_max_min(k)
                batch[k] = (batch[k] - k_min) / (k_max - k_min)  # [0,1]
                batch[k] = 2 * batch[k] - 1  # [-1,1]
    return batch


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, dataset=None, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.dataset = dataset

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def load_state_dict(self, model):
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)
        return model

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        # dataset, test_dataset = get_dataset(args, config)

        train_loader = data.DataLoader(
            self.dataset["train"],
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        fixed_val_num = 10 ** 2
        val_loader = data.DataLoader(
            self.dataset["valid"],
            batch_size=fixed_val_num,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info("Resuming training from checkpoint. Resuming at Epoch: {}, Step: {}".format(start_epoch, step))

        fixed_val_noise = torch.randn(
            fixed_val_num,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
            requires_grad=False
        )

        fixed_val_batch = next(iter(val_loader))
        fixed_val_batch = preprocess_batch(self.device, fixed_val_batch)
        fixed_val_y = fixed_val_batch["pa"].to(self.device)
        fixed_val_x = fixed_val_batch["x"].to(self.device)

        # save fixed val x
        save_dir = os.path.join(self.args.log_path, "samples")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fixed_val_x = (fixed_val_x.clamp(min=-1, max=1) + 1) / 2
        tvu.save_image(
            fixed_val_x, os.path.join(save_dir, "fixed_val_x.png"), nrow=10
        )

        for epoch in range(start_epoch, self.config.training.n_epochs):
            model.train()
            data_start = time.time()
            data_time = 0
            for i, batch in enumerate(train_loader):
                batch = preprocess_batch(self.device, batch)
                x = batch["x"].to(self.device)
                y = batch["pa"].to(self.device)

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                # x = x.to(self.device)
                # x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b, y=y)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"epoch: {epoch} / {self.config.training.n_epochs}, step: {step}, loss: {loss.item()}, "
                    f"data time: {data_time / (i + 1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1 or epoch == self.config.training.n_epochs - 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    logging.info("Checkpoint saved at step {}!".format(step))

                if step % self.config.training.validation_freq == 0:
                    save_dir = os.path.join(self.args.log_path, "samples")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # sample from the model
                    model.eval()
                    with torch.no_grad():
                        x = self.sample_image(fixed_val_noise, model, y=fixed_val_y)
                        x = (x.clamp(min=-1, max=1) + 1) / 2

                        tvu.save_image(
                            x, os.path.join(save_dir, f"{step}.png"), nrow=10
                        )

                data_start = time.time()

    def abduction(self, model, x_0, pa, last=True):
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        # # save fixed val x
        # save_dir = os.path.join(self.args.log_path, "abudction")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        with torch.no_grad():
            if self.args.sample_type == "generalized":
                if self.args.skip_type == "uniform":
                    skip = self.num_timesteps // self.args.timesteps
                    seq = range(0, self.num_timesteps, skip)
                elif self.args.skip_type == "quad":
                    seq = (
                            np.linspace(
                                0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                            )
                            ** 2
                    )
                    seq = [int(s) for s in list(seq)]
                else:
                    raise NotImplementedError

                xs = abduction_steps(x_0, seq, model, self.betas, eta=self.args.eta, y=pa)
                # x_T = xs[-1]
                # x_T_clamp = (x_T.clamp(min=-1, max=1) + 1) / 2
                # xs = (torch.cat(xs[1:]).clamp(min=-1, max=1) + 1) / 2
            else:
                raise NotImplementedError
        if last:
            return xs[-1]
        return xs

    @torch.no_grad()
    def cf_epoch(
            self,
            model,
            pgm,
            predictor,
            dataloaders,
            do_pa=None,
            te_cf=False
    ):
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

        for _, batch in loader:
            bs = batch['x'].shape[0]
            batch = preprocess(batch)
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
            x_T = self.abduction(model, batch['x'], _pa)
            # rec = self.sample_image(x_T, model, y=_pa)
            # abduct counterfactual noise x_cf
            x_cf = self.sample_image(x_T, model, y=_cf_pa)

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

    def eval_cf_loop(self, pgm, predictor, dataloaders, file, seeds, total_effect=False):
        import multiprocessing
        from tqdm import tqdm
        from morphomnist.morpho import ImageMorphology

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

        args, config = self.args, self.config
        # dataset, test_dataset = get_dataset(args, config)

        model = Model(config)

        model = self.load_state_dict(model)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        for do_pa in ['thickness', 'intensity', 'digit', None]:  # "None" is for random interventions
            acc_runs = []
            mae_runs = {
                'thickness': {'predicted': [], 'measured': []},
                'intensity': {'predicted': [], 'measured': []}
            }

            for seed in seeds:
                print(f'do({(do_pa if do_pa is not None else "random")}), seed {seed}:')
                # assert vae.cond_prior if total_effect else True
                targets, preds, x_cfs = self.cf_epoch(model, pgm, predictor, dataloaders, do_pa, total_effect)
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
                f'\n{(v if model.cond_prior else "")}\n' +
                f'digit acc | mean: {np.array(acc_runs).mean()} - std: {np.array(acc_runs).std()}\n' +
                f'thickness mae (predicted) | mean: {np.array(mae_runs["thickness"]["predicted"]).mean()} - std: {np.array(mae_runs["thickness"]["predicted"]).std()}\n' +
                f'thickness mae (measured) | mean: {np.array(mae_runs["thickness"]["measured"]).mean()} - std: {np.array(mae_runs["thickness"]["measured"]).std()}\n' +
                f'intensity mae (predicted) | mean: {np.array(mae_runs["intensity"]["predicted"]).mean()} - std: {np.array(mae_runs["intensity"]["predicted"]).std()}\n' +
                f'intensity mae (measured) | mean: {np.array(mae_runs["intensity"]["measured"]).mean()} - std: {np.array(mae_runs["intensity"]["measured"]).std()}\n'
            )
            file.flush()
        return

    def sample(self):
        model = Model(self.config)

        model = self.load_state_dict(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                # x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        # x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                    torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                    + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i: i + 8], model))
        # x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, y=None, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta, y=y)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas, y=y)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
