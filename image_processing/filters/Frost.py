import numpy as np
import math as m
import copy


def Frost(image, Damp_fact=1, sz=5):
    nimg = copy.copy(image)
    for color in range(3):
        image = nimg[:, :, color]
        ima_fi = np.zeros(image.shape)
        mn = round((sz-1)/2)
        EImg = np.pad(image, mn)
        x1 = range(int(-mn), int(mn+1))
        y1 = range(int(-mn), int(mn+1))
        [x, y] = np.meshgrid(x1, y1)
        S = np.zeros(x.shape)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                S[i][j] = m.sqrt(x[i][j]**2 + y[i][j]**2)

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                K = EImg[i:i+sz, j:j+sz]
                meanV = np.mean(K[:])
                varV = np.var(K[:])
                B = Damp_fact * (varV / (meanV * meanV))
                Weigh = np.zeros(S.shape)
                sum_ch, sum_zn = 0, 0
                for z in range(0, x.shape[0]):
                    for l in range(0, x.shape[1]):
                        Weigh[z][l] = m.exp(-S[z][l]*B)
                        sum_ch += K[z][l] * Weigh[z][l]
                        sum_zn += Weigh[z][l]

                ima_fi[i, j] = sum_ch / sum_zn
        nimg[:, :, color] = ima_fi
    return nimg