
import matplotlib.pyplot as plt
import random
import warnings
import numpy as np

__CMAP__ = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',  'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap','cubehelix', 
            'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu','RdYlGn', 'Spectral', 
            'coolwarm', 'bwr', 'seismic', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool','Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            ]
# see color in: https://matplotlib.org/stable/tutorials/colors/colormaps.html

def rgba(r, const=10., cmap="jet"):
    try:
        assert cmap in __CMAP__
    except AssertionError:
        ori_cmap = cmap
        cmap = __CMAP__[random.randint(0, len(__CMAP__)-1)]
        warnings.warn(str(ori_cmap) + " is not valid in TUtils, random sample valid value: " + str(cmap))
        
    c = plt.get_cmap(cmap)((r % const) / const) 
    c = list(c)
    c[-1] = 0.5  # alpha

    return c

def disp2rgb(disp):
    disp = disp.squeeze()
    H = disp.shape[0]
    W = disp.shape[1]
    I = disp.flatten()

    map = np.array([[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]])
    bins = map[:-1,3]
    cbins = np.cumsum(bins)
    bins = bins/cbins[-1]
    cbins = cbins[:-1]/cbins[-1]

    ind = np.minimum(np.sum(np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:, None],
                                    I.shape[0], axis=1), axis=0), 6)
    bins = np.reciprocal(bins)
    cbins = np.append(np.array([[0]]), cbins[:, None])

    I = np.multiply(I - cbins[ind], bins[ind])
    I = np.minimum(np.maximum(np.multiply(map[ind,0:3], np.repeat(1-I[:,None], 3, axis=1)) \
         + np.multiply(map[ind+1,0:3], np.repeat(I[:,None], 3, axis=1)),0),1)
    I = np.reshape(I, [H, W, 3]).astype(np.float32)
    return I


def range_color(r):
    pass