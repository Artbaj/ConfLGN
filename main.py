import torch
import torchlogix
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# Create a simple logic layer
layer = torchlogix.layers.LogicDense(in_dim=10, out_dim=5)
x = torch.randn(32, 10)
output = layer(x)
print(f"Output shape: {output.shape}")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # np. (60000, 28, 28)
print(y_train.shape)  # (60000,)
plt.imshow(output, cmap='gray')
plt.title(f"Etykieta: {y_train[1]}")
plt.axis('off')

plt.show()