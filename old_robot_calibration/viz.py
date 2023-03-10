import argparse

import numpy as np
import math3d as m3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.artist as artist
import mpl_toolkits.mplot3d as mplot3d


class Arrow3D(patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self.update_xyz(xs, ys, zs)

    def update_xyz(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
        super().draw(renderer)



class Frame:

    def __init__(self, transform=None, scale=1, ax=None, **kwargs):
        kwargs['mutation_scale'] = kwargs.pop('head_size', 10)
        kwargs['arrowstyle'] = kwargs.pop('style', '-|>')
        kwargs['linewidth'] = kwargs.pop('line_width', 2)
        self.scale = scale
        self.xyz_arrows = [
            Arrow3D((0, 1), (0, 0), (0, 0), color=color, **kwargs)
            for color in ['red', 'green', 'blue']
        ]
        self.transform = transform
        if not ax is None:
            self.add(ax)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = m3d.Transform(np.eye(4) if transform is None else transform)
        for i, arrow in enumerate(self.xyz_arrows):
            xyz_a = self.transform * m3d.Vector(np.zeros(3))
            xyz_b = self.transform * m3d.Vector(self.scale * np.arange(3) == i)
            arrow.update_xyz(
                (xyz_a.x, xyz_b.x),
                (xyz_a.y, xyz_b.y),
                (xyz_a.z, xyz_b.z),
            )

    def add(self, ax):
        for arrow in self.xyz_arrows:
            ax.add_artist(arrow)


def plot_frames(ax, Tb, Te, Tr, Tp, area=3):
    T = m3d.Transform()

    T = T * Tb
    platform = Frame(ax=ax, transform=T)

    T = T * Te
    base = Frame(ax=ax, transform=T)

    T = T * Tr
    ee = Frame(ax=ax, transform=T)

    T = T * Tp
    pen = Frame(ax=ax, transform=T)

    ax.set_xlim(-area, area)
    ax.set_ylim(-area, area)
    ax.set_zlim(-area, area)
    ax.view_init(elev=35., azim=120)
    ax.dist = 10

    return platform, base, ee, pen


class Animation:

    def __init__(self, ax, sim_data, **frame_kwargs):
        self.ax = ax
        self.sim_data = sim_data
        self.b_frame = Frame(ax=ax, **frame_kwargs)
        self.e_frame = Frame(ax=ax, **frame_kwargs)
        self.r_frame = Frame(ax=ax, **frame_kwargs)
        self.p_frame = Frame(ax=ax, **frame_kwargs)

    def update(self, num):

        # Extract current
        Tb, Te, Tr, Tp = map(m3d.Transform, self.sim_data[num])

        # Initial world frame
        T = m3d.Transform()

        # Base on platform
        T = T * Tb
        self.b_frame.transform = T

        # Error of estimate of platform
        T = T * Te
        self.e_frame.transform = T

        # Robot transform from base to end-effector
        T = T * Tr
        self.r_frame.transform = T

        # Pen offset from end effector
        T = T * Tp
        self.p_frame.transform = T

        return self.b_frame


    @classmethod
    def run(cls, sim_data, figsize=(10, 10), rate=5e-3, speed=1, area=3, **kwargs):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        animation = cls(ax, sim_data, **kwargs)
        origin, = ax.plot([0], [0], [0])

        ax.set_xlim3d([-area, area])
        ax.set_xlabel('X')
        ax.set_ylim3d([-area, area])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-area, area])
        ax.set_zlabel('Z')
        ax.view_init(elev=40, azim=140)

        def upd(num):
            origin.set_data([[0], [0]])
            animation.update(num)

        N = len(sim_data)
        interval = 1e3 / (rate * speed)  # Delay between updates
        ani = mpl.animation.FuncAnimation(fig, upd, N, interval=interval)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('sim_data')
parser.add_argument('--area', type=float, default=3)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--rate', type=float, default=2) # Hz
parser.add_argument('--speed', type=float, default=1.0)


def main(args):
    import robot_calibration.sim as sim
    sim_data = sim.load_sim_data(args.sim_data)
    Animation.run(
        sim_data,
        area=args.area, rate=args.rate, speed=args.speed, alpha=args.alpha
    )


if __name__ == '__main__':
    main(parser.parse_args())
