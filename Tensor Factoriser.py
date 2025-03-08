import numpy as np
import torch
import math
import pandas as pd
class TensorConverter:
    def __init__(self, a, b, c, d, theta):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.theta = theta
        self.i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

    def create_matrix_A(self):
        A = torch.tensor([
            [torch.sqrt(torch.tensor(self.a, dtype=torch.complex64)) * torch.sqrt(self.i),
             torch.sqrt(torch.tensor(self.b, dtype=torch.complex64)) * torch.sqrt(self.i)],
            [torch.sqrt(torch.tensor(self.c, dtype=torch.complex64)),
             -self.i * torch.sqrt(torch.tensor(self.d, dtype=torch.complex64))]
        ], dtype=torch.complex64)
        return A

    def create_matrix_B(self, A):
        rotation = torch.tensor([
            [math.cos(self.theta), -math.sin(self.theta)],
            [math.sin(self.theta), math.cos(self.theta)]
        ], dtype=torch.float32)
        B = A @ rotation.to(A.dtype)
        return B

    def create_matrix_C(self, A, B):
        C = A @ B
        return C

    def create_tensor(self):
        A = self.create_matrix_A()
        B = self.create_matrix_B(A)
        C = self.create_matrix_C(A, B)
        tensor = torch.stack([A, B, C])
        return tensor

    @staticmethod
    def tensor_to_variables(tensor):
        A = tensor[0]
        a = (A[0, 0].real ** 2).item()
        b = (A[0, 1].real ** 2).item()
        c = (A[1, 0].real ** 2).item()
        d = (A[1, 1].imag ** 2).item()
        theta = math.atan2(tensor[1][1, 0].real, tensor[1][0, 0].real)
        return a*2, b*2, c, d, theta
class dataframeToTensor:
    df=[]
    def __init__(self, dataframe):
        df=dataframe
    @staticmethod
    def create_nth_tensor(self):
        allcolums = df.columns
        for i in allcolums:
            df[i]

# Example usage
a, b, c, d = 26, 2, 32, 0.0001
theta = math.pi / 4
print(a)
print(b)
print(c)
print(d)
print(theta)
converter = TensorConverter(a, b, c, d, theta)
result_tensor = converter.create_tensor()
print("Tensor:", result_tensor)

# Convert tensor back to variables
recovered_a, recovered_b, recovered_c, recovered_d, recovered_theta = TensorConverter.tensor_to_variables(result_tensor)
print("Recovered variables:", recovered_a, recovered_b, recovered_c, recovered_d, recovered_theta)




# Sample DataFrame
data = {
    'Column1': [1, 2, 3],
    'Column2': [4, 5, 6],
    'Column3': [7, 8, 9]
}
df = pd.DataFrame(data)

# Extracting all columns
all_columns = df.columns

