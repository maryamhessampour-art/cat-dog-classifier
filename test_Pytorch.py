import torch
import time

size = 5000

# CPU
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()

result_cpu = torch.matmul(x_cpu, y_cpu)

cpu_time = time.time() - start


# GPU
x_gpu = x_cpu.to("cuda")
y_gpu = y_cpu.to("cuda")

torch.cuda.synchronize()

start = time.time()

result_gpu = torch.matmul(x_gpu, y_gpu)

torch.cuda.synchronize()

gpu_time = time.time() - start


print("CPU time:", cpu_time)
print("GPU time:", gpu_time)