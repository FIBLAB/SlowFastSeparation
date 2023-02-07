import torch
import numpy as np
from torch.utils.data import Dataset


class _2S2FDataset(Dataset):

    def __init__(self, file_path, mode='train', length=None, model=None):
        super().__init__()
        
        self.length = length
        self.model = model

        # Search for txt files
        if length is None:
            self.data = np.load(file_path+f'/{mode}.npz')['data'] # (N, 2, 1, 3) or (N, 1, 1, 3)
        else:
            self.data = np.load(file_path+f'/{mode}_{length}.npz')['data']


    # 0 --> 1
    def __getitem__(self, index):

        trace = self.data[index]

        if self.model == 'cde':
            input = trace[:self.length-1]
            target = trace[self.length-1:]
            input = torch.from_numpy(input).float() # (1, channel, feature_dim)
            target = torch.from_numpy(target).float()
            return input, target
        
        input = trace[0]
        if self.length is None:
            target = trace[0] if len(trace)==1 else trace[1]
        else:
            target = trace[self.length-1]

        input = torch.from_numpy(input).float() # (1, channel, feature_dim)
        target = torch.from_numpy(target).float()

        if self.length is None:
            return input.unsqueeze(0), target.unsqueeze(0)
        else:
            return input.unsqueeze(0), target.unsqueeze(0), [torch.from_numpy(trace[i]).float().unsqueeze(0) for i in range(self.length)] 

    def __len__(self):
        return len(self.data)
    
#     def plot(self):
        
#         inputs = []
#         targets = []
#         for i in range(self.__len__()):
#             input, target, _ = self.__getitem__(i)
#             inputs.append(input[0,0,2])
#             targets.append(target[0,0,2])
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(inputs[:100], label='input')
#         plt.plot(targets[:100], label='target')
#         plt.legend()
#         plt.savefig('data.pdf', dpi=300)

# data_path = 'Data/data/tau_' + str(0.15)
# j = _2S2FDataset(data_path, 'val', length=10)
# j.plot()