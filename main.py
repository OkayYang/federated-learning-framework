import torch


def check_cuda():
    """
    检查系统是否可以使用 CUDA（GPU 加速）以及可用的 GPU 设备。
    """
    print("Checking PyTorch and CUDA environment...\n")

    # 检查 PyTorch 是否可用
    print("PyTorch version:", torch.__version__)

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA is available!")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs detected: {gpu_count}")

        # 列出每个 GPU 的名称
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print(
            "CUDA is not available. Please check your CUDA installation and NVIDIA drivers."
        )


def check_nvidia_smi():
    """
    检查 `nvidia-smi` 命令是否能够正常工作，以确认 GPU 驱动程序是否正常运行。
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print("\nNVIDIA SMI output:\n", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except FileNotFoundError:
        print(
            "Error: `nvidia-smi` command not found. Please ensure that the NVIDIA drivers are installed and available in your system PATH."
        )


if __name__ == "__main__":
    check_cuda()  # 然后检查 CUDA 环境
    check_nvidia_smi()  # 最后检查 `nvidia-smi`
