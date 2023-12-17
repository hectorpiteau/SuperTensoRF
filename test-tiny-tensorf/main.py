import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image_path = "img.png"
image = Image.open(image_path).convert("RGB")  # Convert to RGB if the image has an alpha channel

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # Normalize pixel values
])

normalized_image = transform(image)
print(normalized_image.shape)
# plt.imshow(normalized_image.numpy().transpose(1, 2, 0))
# plt.show()


class TinyRFModel(nn.Module):
    def __init__(self, input_size, N):
        super(TinyRFModel, self).__init__()
        
        self.input_size = input_size
        self.N = N
        
        # First linear layer
        # linears_1 = [] 
        # linears_2 = [] 
        # for i in range(0,N):
        #     linears_1.append(
        #         torch.tensor((input_size, 1))
        #     )
        #     linears_2.append(
        #         torch.tensor((1, input_size))
        #     )
        # self.linears_1 = nn.Parameter(linears_1)
        # self.linears_2 = nn.ParameterList(linears_2)
        self.l_1_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_1_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_2_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_2_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_3_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_3_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_4_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_4_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_5_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_5_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_6_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_6_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_7_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_7_2 = nn.Parameter(torch.randn(1, input_size))

        self.l_8_1 = nn.Parameter(torch.randn(input_size, 1))
        self.l_8_2 = nn.Parameter(torch.randn(1, input_size))

    def forward(self):
        # Forward pass
        # Forward pass
        # First
        # tmp = np.matmul(self.linears_1[0], self.linears_2[0])
        # Following
        # for i in range(1, self.N):
        #     tmp2 = np.matmul(self.linears_1[i], self.linears_2[i])
        #     tmp += tmp2
        tmp = torch.matmul(self.l_1_1, self.l_1_2)
        tmp += torch.matmul(self.l_2_1, self.l_2_2)
        tmp += torch.matmul(self.l_3_1, self.l_3_2)
        tmp += torch.matmul(self.l_4_1, self.l_4_2)
        tmp += torch.matmul(self.l_5_1, self.l_5_2)
        tmp += torch.matmul(self.l_6_1, self.l_6_2)
        tmp += torch.matmul(self.l_7_1, self.l_7_2)
        tmp += torch.matmul(self.l_8_1, self.l_8_2)

        
        return tmp

learning_rate = 0.001
num_epochs = 10000
model = TinyRFModel(400, 8)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


img_size = (3, 400, 400)
start_img = torch.rand(img_size)

image_np = start_img.numpy()

plt.imshow(image_np.transpose(1, 2, 0))
plt.axis('off')
plt.show()

for epoch in range(num_epochs):
    # Forward pass
    outputs = model()
    loss = criterion(outputs, normalized_image)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        plt.imshow(outputs.detach().cpu().numpy())
        plt.show()
print(outputs.shape)
plt.imshow(outputs.detach().cpu().numpy())
plt.axis('off')
plt.show()