"""Same concept as plot3 except we try to save it as a gif"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from gaussian_spheres.pwl import GaussianSpheresPWLP
from matplotlib.animation import FuncAnimation
import shared.filetools
import os
import pytweening

NUM_POINTS = 1000
NUM_CLUSTERS = 50
NUM_LABELS = 2

def main():
    """Main runner"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = torch.zeros((NUM_POINTS, 3), dtype=torch.double)
    labels = torch.zeros(NUM_POINTS, dtype=torch.long)
    pwl = GaussianSpheresPWLP.create(NUM_POINTS, 3, NUM_LABELS, 2, NUM_CLUSTERS, 0.04, 0, 0.08)
    pwl.fill(points, labels)

    ax.scatter(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), s=1, c=labels.numpy(), label='3d clusters')
    ax.legend(loc=1)

    def update(time_ms: float):
        print(f'time_ms={time_ms}')
        duration = 12000                    # 12000
        if time_ms < duration: # spin
            progress = time_ms / duration
            rotation = pytweening.easeInOutSine(progress) * 360
            ax.view_init(30, rotation)
            return ax
        time_ms -= duration
        duration = 3000                     # 15000
        if time_ms < duration: # wait
            if time_ms - 16.67 <= 0:
                ax.view_init(30, 0)
                return ax
            return tuple()
        time_ms -= duration
        duration = 3000                     # 18000
        if time_ms < duration: # elev from 30 to -30
            progress = time_ms / duration
            elev = 30 - 60 * pytweening.easeInOutBack(progress)
            ax.view_init(elev, 0)
            return ax
        duration = 1500                     # 19500
        if time_ms < duration: # wait
            if time_ms - 16.67 <= 0:
                ax.view_init(-30, 0)
                return ax
            return tuple()
        time_ms -= duration
        duration = 3000                     # 22500
        if time_ms < duration: # elev from -30 to 30
            progress = time_ms / duration
            elev = -30 + 60 * pytweening.easeInOutBack(progress)
            ax.view_init(elev, 0)
            return ax
        time_ms -= duration
        # wait rest                         # 24000
        if time_ms - 16.67 <= 0:
            ax.view_init(30, 0)
            return ax
        return tuple()

    os.makedirs(shared.filetools.savepath(), exist_ok=True)

    anim = FuncAnimation(fig, update, frames=np.arange(0, 24000, 16.67), interval=16.67)
    anim.save(os.path.join(shared.filetools.savepath(), 'out.mp4'), dpi=100, writer='ffmpeg')


if __name__ == '__main__':
    main()
