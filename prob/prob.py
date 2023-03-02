import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
import numpy as np
from scipy import signal


def aberrations(image):
    num = np.poly([0, -0.99, -0.99, 0.8])
    den = np.poly([0.95 * np.exp(1j * np.pi/8),  0.95 * np.exp(-1j * np.pi/8), 0.9 * np.exp(1j * np.pi/2),  0.9 * np.exp(-1j * np.pi/2)])

    image_filtree = signal.lfilter(num, den, image)
    return image_filtree

def rotation(image):
    image = np.mean(image, -1)
    x, y  = image.shape

    newImage = np.zeros((x, y))
    for col in range(x):
        for row in range(y):
            newImage[row][y - col - 1] = image[col][row]

    plt.imshow(newImage)
    plt.show()


def main():
    plt.gray()
    image = mpimg.imread("goldhill_rotate.png", 0)
    rotation(image)


































def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k




if __name__ == "__main__":
    main()