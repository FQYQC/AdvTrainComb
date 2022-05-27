import torch


class MyDataLoader(torch.utils.data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = len(data)

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return self.len
