import argparse
import logging
import json

import torch
import tqdm
import numpy as np
import wandb

import robot_calibration.sim as sim


logger = logging.getLogger(__name__)


def get_pen_positions(Tb, Te, Tr, Tp):
    """Computes the xyz position of the tip of the pen"""
    T = Tb @ Te @ Tr @ Tp
    b = T.shape[0]
    p = torch.cat([torch.zeros(b, 3), torch.ones(b, 1)], axis=1).double()
    return torch.einsum('b m n, b n -> b m', T, p)[:, :3]

def to_tensor(x):
    """Converts numpy array to double precision torch tensor"""
    return torch.tensor(x, dtype=torch.float64)

def min_ang(a):
    """Reduces the absolute value of an angle"""
    return min([a, a % (2 * np.pi), a % (-2 * np.pi)], key=abs)

class Dataset:

    def __init__(self, sim_data, obs_noise=0.0):
        assert sim_data.ndim == 4
        self.obs_noise = obs_noise
        self.sim_data_raw = sim_data
        self.sim_data_torch = to_tensor(sim_data)
        self.Tb = self.sim_data_torch[:, 0]
        self.Te = self.sim_data_torch[:, 1]
        self.Tr = self.sim_data_torch[:, 2]
        self.Tp = self.sim_data_torch[:, 3]
        self.pen_xyz_raw = get_pen_positions(self.Tb, self.Te, self.Tr, self.Tp)
        self.pen_xyz = self.pen_xyz_raw + obs_noise * torch.randn_like(self.pen_xyz_raw)

    def __len__(self):
        return len(self.sim_data_raw)

    @property
    def fit_data(self):
        return self.Tb, self.Tr, self.pen_xyz

def get_true_params(sim_params):
    return dict(
        x=sim_params['err_x'],
        y=sim_params['err_y'],
        z=sim_params['err_z'],
        a=sim_params['err_a'],
        p=sim_params['err_p'],
    )


class Estimator(torch.nn.Module):

    def __init__(self, x0=0, y0=0, z0=0, a0=0, p0=0.2):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(x0).double())
        self.y = torch.nn.Parameter(torch.tensor(y0).double())
        self.z = torch.nn.Parameter(torch.tensor(z0).double())
        self.a = torch.nn.Parameter(torch.tensor(a0).double())
        self.p = torch.nn.Parameter(torch.tensor(p0).double())

    @property
    def params(self):
        with torch.no_grad():
            return dict(
                x=self.x.item(),
                y=self.y.item(),
                z=self.z.item(),
                a=min_ang(self.a.item()),
                p=self.p.item(),
            )

    def save_params(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.params, f)

    def param_errors(self, x, y, z, a, p):
        with torch.no_grad():
            return dict(
                x=(self.x - x).item(),
                y=(self.y - y).item(),
                z=(self.z - z).item(),
                a=(self.a - a).item(),
                p=(self.p - p).item(),
            )

    @property
    def transforms(self):
        return torch.stack([self.Te, self.Tp], axis=0).detach().numpy()

    @property
    def Te(self):
        """
        c := cos(a); s := sin(a)
            [ c, -s, 0, x]
            [ s,  c, 0, y]
            [ 0,  0, 1, z]
            [ 0,  0, 0, 1]
        """
        cos_a, sin_a = torch.cos(self.a), torch.sin(self.a)
        Te = torch.eye(4).double()
        Te[0, 0] = cos_a
        Te[0, 1] = -sin_a
        Te[1, 0] = sin_a
        Te[1, 1] = cos_a
        Te[0, 3] = self.x
        Te[1, 3] = self.y
        Te[2, 3] = self.z
        return Te

    @property
    def Tp(self):
        """
            [ 1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, p],
            [ 0, 0, 0, 1],
        """
        Tp = torch.eye(4).double()
        Tp[2, 3] = self.p
        return Tp

    def forward(self, Tb, Tr):
        assert Tb.ndim == Tr.ndim == 3
        n = Tb.shape[0]
        Te = torch.tile(self.Te, (n, 1, 1))
        Tp = torch.tile(self.Tp, (n, 1, 1))
        return get_pen_positions(Tb=Tb, Te=Te, Tr=Tr, Tp=Tp)

    def loss(self, Tb, Tr, pen_xyz):
        pen_xyz_pred = self(Tb, Tr)
        err_xyz = pen_xyz_pred - pen_xyz
        return err_xyz.pow(2).sum(axis=-1).mean()  # Mean squared euc dist


    @classmethod
    def fit(cls, data, init=None, lr=1.0, iters=10, pbar=True, cb=None):

        Tb, Tr, pen_xyz = data

        estimator = cls(**(init or {}))
        # optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
        optimizer = torch.optim.LBFGS(estimator.parameters(), lr=lr)

        pbar = tqdm.tqdm(range(iters), disable=(not pbar))
        loss = None

        for step in pbar:

            def closure():
                return estimator.loss(Tb=Tb, Tr=Tr, pen_xyz=pen_xyz)

            # Do one step of batch gradient descent
            optimizer.zero_grad()
            loss = closure()
            loss.backward()
            optimizer.step(closure)
            loss = loss.detach().item()

            # Call callback if provided and possibly stop loop
            if not cb is None and cb(step, estimator, loss):
                break

            # Upbdate pbar
            pbar.set_postfix(dict(loss=loss))

        return estimator, loss


# -- CLI

parser = argparse.ArgumentParser()
parser.add_argument('sim_data')
parser.add_argument('sim_params')
parser.add_argument('--wandb')
parser.add_argument('--tag')
parser.add_argument('--out')
parser.add_argument('--pbar', action='store_true')

parser.add_argument('--noise', default=0.0, type=float)

parser.add_argument('--num', type=int)
parser.add_argument('--split', type=float)
parser.add_argument('--iters', default=10, type=int)
parser.add_argument('--lr', default=1.0, type=float)


def main(args):

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s: %(message)s'
    )

    # Load data
    sim_data = sim.load_sim_data(args.sim_data)[:args.num]
    logger.info('Loaded %s sim data points from %s', len(sim_data), args.sim_data)
    sim_params = sim.load_sim_params(args.sim_params)
    logger.info('Loaded sim params from %s', args.sim_params)

    # Grab the true params (x,y,z,a,p)
    true_params = get_true_params(sim_params)
    logger.info('True params: %s', true_params)

    # Split and create datasets
    if args.split:
        n_train = int(round(args.split * len(sim_data)))
        data_train = Dataset(sim_data[:n_train], obs_noise=args.noise)
        data_valid = Dataset(sim_data[n_train:])
        logger.info('Split data into %s train and %s valid', len(data_train), len(data_valid))
    else:
        n_train = len(sim_data)
        data_train = Dataset(sim_data, obs_noise=args.noise)
        data_valid = None

    # Initialize wandb if configured
    if args.wandb:
        wandb.init(project=args.wandb, config={**sim_params, **vars(args)})
        wandb.config.num_train = len(data_train)

    def evaluate(estimator):
        info = {}

        if data_valid is not None:
            loss_valid = estimator.loss(*data_valid.fit_data).detach().item()
            info['Loss/valid'] = loss_valid

        param_errors = estimator.param_errors(**true_params)
        info.update({f'AbsErr/{k}': abs(v) for k, v in param_errors.items()})
        info.update({f'Err/{k}': v for k, v in param_errors.items()})

        param_value = estimator.params
        info.update({f'Est/{k}': v for k, v in estimator.params.items()})

        return info

    def callback(step, estimator, loss):
        if args.wandb:
            info = {'Loss/train': loss}
            info.update(evaluate(estimator))
            wandb.log(info, step=step)
            # TODO: Log wandb


    logger.info('Begin training')
    estimator, final_loss = Estimator.fit(
        data=data_train.fit_data,
        iters=args.iters,
        lr=args.lr,
        cb=callback,
        pbar=args.pbar,
    )
    logger.info('Finished training. Final loss: %.10f', final_loss)

    info = {'Loss/train': estimator.loss(*data_train.fit_data).detach().item()}
    info.update(evaluate(estimator))
    logger.info('Final results: %s', info)

    if args.wandb:
        for k, v in info.items():
            wandb.run.summary[k] = v

    if args.out:
        estimator.save_params(args.out)
        logger.info('Saved params to %s', args.out)

    logger.info('Done!')


if __name__ == '__main__':
    main(parser.parse_args())
