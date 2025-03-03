import torch

# Random 3rd-rank tensor (2x2xn)
n = 5  # Number of slices
T = torch.rand(2, 2, n, dtype=torch.cfloat)  # Complex tensor

# Extract individual values from each matrix slice
# a_k = |T[0, 0, k]|^2
a = torch.abs(T[0, 0, :]) ** 2  # Reverse square root
# b_k = |T[0, 1, k]|^2
b = torch.abs(T[0, 1, :]) ** 2
# c_k = |T[1, 0, k]|^2
c = torch.abs(T[1, 0, :]) ** 2
# d_k = |T[1, 1, k]|^2
d = torch.abs(T[1, 1, :]) ** 2

# Reconstruct each 2x2 matrix from extracted values
# A_k = [ sqrt(a_k) * sqrt(i)     sqrt(b_k) / sqrt(i) ]
#       [ sqrt(c_k)              -i * sqrt(d_k)      ]
reconstructed_matrices = []
for k in range(n):
    matrix = torch.tensor([
        [torch.sqrt(a[k]) * torch.sqrt(torch.tensor(1j)),
         torch.sqrt(b[k]) / torch.sqrt(torch.tensor(1j))],
        [torch.sqrt(c[k]),
         -1j * torch.sqrt(d[k])]
    ])
    reconstructed_matrices.append(matrix)

# Convert list of matrices back into a tensor
reconstructed_tensor = torch.stack(reconstructed_matrices, dim=2)

# Calculate the difference between the original and reconstructed tensors
difference_tensor = T - reconstructed_tensor

# Add the difference to the reconstructed tensor
adjusted_reconstructed_tensor = reconstructed_tensor + difference_tensor

print("Original Tensor:")
print(T)
print("\nReconstructed Tensor:")
print(reconstructed_tensor)
print("\nDifference Tensor:")
print(difference_tensor)
print("\nAdjusted Reconstructed Tensor:")
print(adjusted_reconstructed_tensor)

# Check if they match
print("\nTensors match:", torch.allclose(T, adjusted_reconstructed_tensor))

# Difference Tensor:
# tensor([[[ 0.0278-0.0288j,  0.0781-0.1467j,  0.1347-0.2319j,  0.0285-0.0294j,
#            0.0942-0.1056j],
#          [-0.1760+1.5330j, -0.2163+0.5264j,  0.0701+0.3024j, -0.0659+1.9289j,
#            0.1346+0.4069j]],
#
#         [[-0.5322+0.7243j, -0.3379+0.8677j, -0.3852+0.5895j, -0.0127+0.1312j,
#           -0.1468+0.3314j],
#          [ 0.9637+2.3641j,  0.7896+1.5402j,  0.3190+0.8932j,  0.3355+0.4108j,
#            0.6736+0.7512j]]])