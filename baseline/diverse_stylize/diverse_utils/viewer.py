import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def anim(frame, ax, positions, parents, color):
    plt.cla()
    skeleton = positions[frame, :, :]
    trajectory = positions[:, 0, [0, 2]]
    X = []
    Y = []
    Z = []

    for i in range(len(skeleton)):
        X.append(skeleton[i, 0])
        Y.append(skeleton[i, 2])
        Z.append(skeleton[i, 1])

        if parents[i] == -1:
            continue
        else:
            line_X = []
            line_Y = []
            line_Z = []
            line_X.append(skeleton[i, 0])
            line_X.append(skeleton[parents[i], 0])
            line_Y.append(skeleton[i, 2])
            line_Y.append(skeleton[parents[i], 2])
            line_Z.append(skeleton[i, 1])
            line_Z.append(skeleton[parents[i], 1])
            ax.plot(line_X, line_Y, line_Z, color=color)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 30)

    ax.scatter(X, Y, Z, c=color, s=25)
    # if show_traj:
    #     ax.plot(trajectory[:, 0], trajectory[:, 1], c='gray')
    ax.view_init(30, 120)


def visualize(pos, par, frame, col='b', interval=1, show_traj=False, save=False, filename=None, mode=None):
    """
    :param pos: positions of the skeleton (F, J, 3)
    :param par: an array of the parent's index for each joint
    :param frame: the number of frames (F)
    :param col: the color of the skeleton; blue by default
    :param save: if true, save the animation
    :param filename: if save, the animation will be saved as filename (.mp4)
    :param mode: if train, close the plot. Otherwise, show the plot
    :return: the animation
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    anime = animation.FuncAnimation(
        fig, anim, fargs=(ax, pos, par, col), interval=interval, frames=frame, repeat=False
    )

    # for saving animation
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2500)
        anime.save(filename, writer=writer)

    if mode is None:
        plt.show()
    else:
        plt.close()

    return anime  # need to return animation at last
