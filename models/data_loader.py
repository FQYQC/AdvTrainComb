import torch
import numpy as np


class MyDataLoader(torch.utils.data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = len(data)

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return self.len


def pca(mat, rank=10):
    channels = mat.shape[0]
    if(channels > 4):
        print('too many channels')
        return
    IMG_HEIGHT = mat.shape[1]
    IMG_WIDTH = mat.shape[2]
    image = mat
    recon_image = np.zeros(image.shape)
    for chan in range(channels):
        U, S, V = np.linalg.svd(image[chan,:, :])
        for rk in range(rank):
            recon_image[chan,:, :] += S[rk]*np.dot(U[:, rk].reshape(IMG_HEIGHT, 1),
                                                    V[rk, :].reshape(1, IMG_WIDTH))
    return recon_image
