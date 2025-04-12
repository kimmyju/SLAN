import torch
import torch.nn as nn
import torch.nn.functional as F

factor = 2

class AttentionModule(nn.Module) :

    def __init__(self) :
        super().__init__()
        temp_list = list(range(63))  
        partition_size = 9  
        self.atten_list = [temp_list[i:i+partition_size] for i in range(0, len(temp_list), partition_size)]
        self.conv1 = nn.Conv2d(63, 63, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(63)
        self.conv2 = nn.Conv2d(63, 63, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(63)
        self.relu = nn.ReLU()

    def forward(self, x) :
        input, light = x[0], x[1]
        alist = list()
        for i in range(7) : 
            alist.append(self.atten_list[i][light - 1])
        for channel in alist :
            input[:, channel, :, :] *= factor
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out 