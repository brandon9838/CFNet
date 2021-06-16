from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.switch_backend('agg')

def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5),bbx=None,mass_center=None):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 6, 18))
    my_angles=[(30,-45),(30,45),(30,135)]
    #my_angles=[(0,0),(0,90),(90,0)]
    if mass_center is None:
        mass_center=[0,0,0]
    if bbx is None:
        bbx=np.array([[0.499,0.499,0.499],
         [0.499,0.499,-0.499],
         [0.499,-0.499,0.499],
         [0.499,-0.499,-0.499],
         [-0.499,0.499,0.499],
         [-0.499,0.499,-0.499],
         [-0.499,-0.499,0.499],
         [-0.499,-0.499,-0.499]
        ])
    for i in range(3):
        elev = my_angles[i][0]
        azim = my_angles[i][1]
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = np.ones(shape=pcd.shape[0])*0.15
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-0.5, vmax=0.5)
            #ax.scatter(mass_center[0], mass_center[1], mass_center[2], zdir=zdir, c=[0.1], s=10.0, cmap='Blues', vmin=-0.5, vmax=0.5)
            #ax.scatter(bbx[:, 0], bbx[:, 1], bbx[:, 2], zdir=zdir, c= np.ones(shape=(8))*0.5, s=10.0, cmap='Blues', vmin=-1.0, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
def plot_pcd_nn_dist(filename, pcds, nn_dists, titles, suptitle='', sizes=None, cmap='rainbow', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]

    fig = plt.figure(figsize=(len(pcds) * 3, 9))

    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = nn_dists[j]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            nn_plt = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=0.0, vmax=0.04)

            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_title(titles[j])
            plt.colorbar(nn_plt, ax=ax)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
def plot_pcd_three_views_color_differ(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    print(pcds[0].shape)
    print(pcds[1].shape)
    print(pcds[2].shape)
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            if j==1:
                cmap = 'Blues'
            else:
                cmap = 'Reds'
            size = 0.5
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
