import argparse
import json
import sys

import pandas as pd
import numpy as np
import math3d as m3d
import tqdm


class WaveData:

    def __init__(self, data, name=None):
        self.data = data
        self.name = name

    def normalize(self, scale=1, shift=None):
        data = self.data.copy()
        if shift is None:
            pass
        elif shift == 'center':
            mean = data.body_pos.mean()
            data.body_pos -= mean
            data.target_pos -= mean
        else:
            data.body_pos += shift
            data.target_pos += shift
        data.body_pos *= scale
        data.target_pos *= scale
        return WaveData(data, self.name)

    @classmethod
    def load_hdf(cls, filepath, conditions):
        return cls(pd.read_hdf(filepath, key=conditions), conditions)

    @classmethod
    def load_csv(cls, filepath):
        return cls(pd.read_csv(
            filepath, header=[0, 1], parse_dates=[0], index_col=0
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return WaveData(self.data.iloc[i], self.name)
        step = self.data.iloc[i]
        pos = m3d.Vector(step.body_pos.values)
        angles = step.body_rot[list('zyx')].values
        rot = m3d.Orientation.new_euler(angles, 'ZYX')
        time = step.name / pd.Timedelta(seconds=1)
        return time, m3d.Transform(rot, pos)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f'<WaveData num={len(self)} name={self.name}>'


class Kinematics:

    def __init__(
        self,
        stationary=False,
        err_x=0,
        err_y=0,
        err_z=0,
        err_a=0,
        err_p=0.2,
    ):
        self.param_names = set(locals()) - {'self'}
        self.stationary = stationary
        self.err_x=err_x
        self.err_y=err_y
        self.err_z=err_z
        self.err_a=err_a
        self.err_p=err_p

    @property
    def params(self):
        return {k: getattr(self, k) for k in self.param_names}

    def get_Tb(self, time, platform_transform=None):
        """Platform to robot-base (ish)"""
        if platform_transform is None or self.stationary:
            return m3d.Transform()
        else:
            return platform_transform.copy()

    def get_Te(self, time):
        """Mocap base estimate to actual robot base (our eyeballing error)"""
        return m3d.Transform(
             [self.err_x, self.err_y, self.err_z] + [0, 0, self.err_a]
        )

    def get_Tr(self, time):
        """Robot base to end-effector (robot forward kinematics)"""
        raise NotImplementedError()

    def get_Tp(self, time):
        """End-effector to pen tip (extends from center of ee along z-axis)"""
        return m3d.Transform([0, 0, self.err_p] + [0, 0, 0])


    def get(self, time, platform_transform=None):

        Tb = self.get_Tb(time, platform_transform)
        Te = self.get_Te(time)
        Tr = self.get_Tr(time)
        Tp = self.get_Tp(time)

        return Tb, Te, Tr, Tp


class PeriodicKinematics(Kinematics):

    def __init__(self, length_1=2.0, length_2=1.0, speed=0.0, **kwargs):
        super().__init__(**kwargs)
        self.length_1 = length_1
        self.length_2 = length_2
        self.speed = speed
        self.param_names |= {'length_1', 'length_2', 'speed'}

    def get_Tr(self, time):
        """Robot base to end-effector (robot forward kinematics)"""
        # Displace end effector from middle joint
        T_l2 = m3d.Transform([0, 0, self.length_2] + [0, 0, 0])
        # Bend middle joint around y axis
        range = np.pi / 8 #+/-
        angle = range * np.sin(self.speed * time / (2 * range))
        T_r2 = m3d.Transform([0, 0, 0] + [0, angle, 0])
        # Displace middle joint from base
        T_l1 = m3d.Transform([0, 0, self.length_1] + [0, 0, 0])
        # Rotate around base
        range = np.pi
        angle = range * np.sin(self.speed * time / (2 * range))
        T_r1 = m3d.Transform([0, 0, 0] + [0, 0, angle])

        return T_r1 * T_l1 * T_r2 * T_l2


def run_simulation(wave_data, kin):
    data = []
    for time, platform_transform in tqdm.tqdm(wave_data):
        data.append(
            np.stack(
                [np.array(T.matrix) for T in kin.get(time, platform_transform)],
                axis=0
            )
        )
    return np.stack(data, axis=0)

def save_sim_data(filepath, sim_data, n_transforms=4):
    assert sim_data.ndim == 4
    assert sim_data.shape[1:] == (n_transforms, 4, 4)
    np.savetxt(filepath, sim_data.reshape(-1, n_transforms * 4 * 4))

def load_sim_data(filepath, n_transforms=4):
    sim_data = np.loadtxt(filepath)
    return sim_data.reshape(-1, n_transforms, 4, 4)


def save_sim_params(filepath, **params):
    with open(filepath, 'w') as f:
        json.dump(params, f)

def load_sim_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


parser = argparse.ArgumentParser()

parser.add_argument('data_path')
parser.add_argument('--scale', default=0.1, type=float)
parser.add_argument('--shift', default='center')
parser.add_argument('--stationary', action='store_true')

parser.add_argument('--length_1', default=2, type=float)
parser.add_argument('--length_2', default=1, type=float)
parser.add_argument('--speed', default=0, type=float)

parser.add_argument('--err_x_scale', default=0, type=float)
parser.add_argument('--err_y_scale', default=0, type=float)
parser.add_argument('--err_z_scale', default=0, type=float)
parser.add_argument('--err_a_scale', default=0, type=float)
parser.add_argument('--err_p_scale', default=0, type=float)

parser.add_argument('--seed', default=np.random.randint(1, 1<<32 - 1), type=int)
parser.add_argument('--sim_data')
parser.add_argument('--sim_params')

def main(args):

    np.random.seed(args.seed)

    wave_data = WaveData.load_csv(args.data_path)
    wave_data = wave_data.normalize(scale=args.scale, shift=args.shift)

    kinematics = PeriodicKinematics(
        stationary=args.stationary,
        err_x=np.random.uniform(-args.err_x_scale, args.err_x_scale),
        err_y=np.random.uniform(-args.err_y_scale, args.err_y_scale),
        err_z=np.random.uniform(-args.err_z_scale, args.err_z_scale),
        err_a=np.random.uniform(-args.err_a_scale, args.err_a_scale),
        err_p=0.2 + np.random.uniform(-args.err_p_scale, args.err_p_scale),
        length_1=args.length_1,
        length_2=args.length_2,
        speed=args.speed,
    )

    sim_data = run_simulation(wave_data, kinematics)

    save_sim_data(args.sim_data or sys.stdout, sim_data)
    if args.sim_params:
        save_sim_params(args.sim_params, **{**vars(args), **kinematics.params})

if __name__ == '__main__':
    main(parser.parse_args())
