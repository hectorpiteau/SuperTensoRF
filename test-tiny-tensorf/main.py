import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class TinyRFModel(nn.Module):
    def __init__(self, input_size, N):
        super(TinyRFModel, self).__init__()
        
        self.input_size = input_size
        self.N = N
        
        # First linear layer
        self.linears_1 = [] 
        self.linears_2 = [] 
        for i in range(0,N):
            self.linears_1.append(
                torch.tensor((input_size, 1))
            )
            self.linears_2.append(
                torch.tensor((1, input_size))
            )
        
    def forward(self, x):
        # Forward pass
        # First
        tmp = np.matmul(self.linears_1[0], self.linears_2[0])
        # Following
        for i in range(1, self.N):
            tmp2 = np.matmul(self.linears_1[i], self.linears_2[i])
            tmp += tmp2
        
        return tmp




img_size = (3, 400, 400)
start_img = torch.rand( img_size)

image_np = start_img.numpy()

plt.imshow(image_np.transpose(1, 2, 0))
plt.axis('off')
plt.show()