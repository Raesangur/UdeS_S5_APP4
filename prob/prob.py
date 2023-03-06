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

    return newImage

def rotation_base(image):
    #image = np.mean(image, -1)
    x, y  = image.shape

    newImage = np.zeros((x, y))
    for e1 in range(x):
        for e2 in range(y):
            u1 =  e2
            u2 = -e1
            newImage[u1][u2] = image[e1][e2]

    return newImage

def elliptic_filter(image):
    fe = 1600
    fc = 500
    filterOrder = 4
    rippleMax   = 0.2
    threshold   = 60

    hz_ellip = signal.ellip(filterOrder, rippleMax, threshold, 2 * fc/fe)

    return signal.lfilter(hz_ellip[0], hz_ellip[1], image)


def bilinear(image):
    f  = 500
    fe = 1600
    wd = 2 * np.pi * f / fe
    wa = 2 * fe * np.tan(wd / 2)

    alpha = 2 * fe / wa     # 2 / T / wc

    a = alpha ** 2 + alpha * np.sqrt(2) + 1
    b = -2 * alpha ** 2 + 2
    c = alpha ** 2 - alpha * np.sqrt(2) + 1

    num = [1, 2, 1]
    den = [a, b, c]

    image_filtree = signal.lfilter(num, den, image)
    return image_filtree

def compress(image, factor=0.5):
    covariance = np.cov(image)
    eigenvalues, eigenvector = np.linalg.eig(covariance)
    transferMatrix    = np.transpose(eigenvector)
    transferMatrixInv = np.linalg.inv(transferMatrix)

    Iv = transferMatrix.dot(image)
    size = len(Iv)
    Iv = [Iv[n] if n < size * factor else np.zeros(size) for n in range(size)]
    Io = transferMatrixInv.dot(Iv)

    return Io

def show(image):
    plt.imshow(image)
    plt.show()
    return image

def main():
    plt.gray()
    image = show(np.load("image_complete.npy"))
    image = show(aberrations(image))
    image = show(rotation_base(image))
    image = show(elliptic_filter(image))
    image = show(compress(image))
    
    




































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