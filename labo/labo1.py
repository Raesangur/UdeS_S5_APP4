import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
import numpy as np
from scipy import signal


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







# -----------------------------------------------------------------------------

num = np.poly([0.8j, -0.8j])
den = np.poly([0.95 * np.exp(1j * np.pi/8),  0.95 * np.exp(-1j * np.pi/8)])

#zp = zplane(num, den)

w, H = signal.freqz(num, den)
angles = np.unwrap(np.angle(H))

#fig, ax1 = plt.subplots()
#ax1.plot(w, 20*np.log10(np.abs(H)))
#ax2 = ax1.twinx()
#ax2.plot(w, angles, "g-")
#plt.show()


N = 200
impulse = [0 if n != 0 else 1 for n in range(-N, N)]
filtre = signal.lfilter(num, den, impulse)
z = signal.lfilter(den, num, filtre)

#plt.plot(filtre)
#plt.show()
#plt.plot(z)
#plt.show()



# -----------------------------------------------------------------------------
z0 = np.exp(1j * np.pi/16)
z1 = z0
p0 = 0.99 * np.exp(1j * np.pi/16)
p1 = p0

num = np.poly([z0, z1])
den = np.poly([p0, p1])
# zp = zplane(num, den)

sin = [np.sin(np.pi * n / 16) + np.sin(np.pi * n / 32) for n in range(N)]
y   = signal.lfilter(num, den, sin)
#plt.plot(y)
#plt.show()



# -----------------------------------------------------------------------------
fe = 48000
bande_passante = 2500
bande_coupe = 3500




# -----------------------------------------------------------------------------
plt.gray()
img_couleur = mpimg.imread('goldhill.png')

plt.imshow(img_couleur)
#plt.show()

x_s, y_s = img_couleur.shape
T = [[2,  0 ],                                                                 \
     [0, 1/2]]

img_t = [[None] * (y_s // 2 + 1)] * (x_s * 2)

for c1 in range(x_s):
    row = []
    for l1 in range(y_s):
        l2 = y_s - 1 - l1
        xt = 2 * c1
        yt = 1/2 * l2
        img_t[round(xt)][round(yt)] = img_couleur[c1, l1]

plt.imshow(img_t)
plt.show()
