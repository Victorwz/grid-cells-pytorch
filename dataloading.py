"""
Pytorch data loader for the converted TFRecord data
"""
import torch
from torch.utils import data
from torch import Tensor
import glob

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_loc='../data/torch/', batch_size=256):
        'Initialization'
        self.file_list = glob.glob(data_loc + '*' )
        self.j = 0
        self.temp = 0
        self.data = None
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the total number of samples'
        return int(1e6)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        index = index % 10000

        loaded_data = self.loaded_data()
        X = (
            Tensor(loaded_data['init_pos'][index]) ,
            Tensor(loaded_data['init_hd'][index]),
            Tensor(loaded_data['ego_vel'][index])
            )

        y = (Tensor(loaded_data['target_pos'][index]) ,
                Tensor(loaded_data['target_hd'][index]))
        return X, y

    def loaded_data(self):
        if self.data == None or self.j % self.batch_size == 0:
            file_id = torch.randint(high=len(self.file_list), size=(1, ))
            ID = self.file_list[file_id]
            self.data = torch.load(ID)

        self.j += 1
        return self.data
