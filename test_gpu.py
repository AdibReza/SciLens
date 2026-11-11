import torch

print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():

    device = torch.device("mps")

    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    result = torch.matmul(x, y)

    print("\nGPU test successful!")
    print("Device:", result.device)
    print("Output shape:", result.shape)

else:
    print("\nMPS is not available.")