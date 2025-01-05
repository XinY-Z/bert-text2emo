import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def read_data(config, use='train'):
    if use == 'train':
        data = pd.read_csv(config['train_path'])
    elif use == 'dev':
        data = pd.read_csv(config['dev_path'])
    elif use == 'test':
        data = pd.read_csv(config['test_path'])
    else:
        raise ValueError('use should be either train, dev, or test.')

    if config['num_labels'] == 1:
        torch_data = MyDataset(data[config['x']], torch.Tensor(data[config['y']]).unsqueeze(1))
    else:
        # assume y is a list
        data[config['y']] = data[config['y']].apply(ast.literal_eval)
        torch_data = MyDataset(data[config['x']], torch.Tensor(data[config['y']]))

    torch_dataloader = DataLoader(torch_data, batch_size=config['batch_size'], shuffle=True)

    return torch_dataloader
