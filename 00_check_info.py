'''
D:\Python\python11\python.exe F:\ST\code\00_check_info.py
========================================
      系统与硬件环境信息检测
========================================
【CPU 信息】
逻辑线程数 (Threads): 32
物理核心数 (Cores):   16

【内存信息 (RAM)】
总内存: 31.30GB
已使用: 23.91GB (76.4%)
可用  : 7.38GB

【GPU 与 PyTorch 支持】
PyTorch 版本: 2.10.0.dev20251206+cu128
GPU 加速可用 (CUDA): ✅ 支持
CUDA 版本: 12.8
cuDNN 版本: 91002

当前 PyTorch 编译支持的架构 (SM):
sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

发现 1 个 GPU 设备:
  [GPU 0] -> 型号: NVIDIA GeForce RTX 5070 Laptop GPU
      硬件算力 (Compute Capability): 12.0
      显存大小: 7.96GB
      测试: 张量写入显存成功

进程已结束，退出代码为 0

'''

import torch
import psutil
import os
import sys


def get_size(bytes, suffix="B"):
    """
    字节转换辅助函数，将字节转换为 GB/MB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def print_system_info():
    print("=" * 40)
    print("      系统与硬件环境信息检测")
    print("=" * 40)

    # 1. CPU 线程信息
    # os.cpu_count() 返回的是逻辑 CPU 数量 (线程数)
    # psutil.cpu_count(logical=False) 返回物理核心数
    print(f"【CPU 信息】")
    print(f"逻辑线程数 (Threads): {os.cpu_count()}")
    try:
        print(f"物理核心数 (Cores):   {psutil.cpu_count(logical=False)}")
    except:
        pass

    # 2. 内存信息
    svmem = psutil.virtual_memory()
    print(f"\n【内存信息 (RAM)】")
    print(f"总内存: {get_size(svmem.total)}")
    print(f"已使用: {get_size(svmem.used)} ({svmem.percent}%)")
    print(f"可用  : {get_size(svmem.available)}")

    # 3. GPU 和 PyTorch 信息
    print(f"\n【GPU 与 PyTorch 支持】")
    print(f"PyTorch 版本: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"GPU 加速可用 (CUDA): {'✅ 支持' if cuda_available else '❌ 不支持'}")

    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        # CUDNN 版本 (深度学习加速库)
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")

        # 4. PyTorch 支持的 SM 架构 (Software Support)
        # 这回答了 "torch支持sm到哪里" 的问题
        try:
            arch_list = torch.cuda.get_arch_list()
            print(f"\n当前 PyTorch 编译支持的架构 (SM):")
            # 格式化打印，避免太长
            print(", ".join(arch_list))
        except AttributeError:
            print("无法获取架构列表 (可能是旧版 PyTorch)")

        # 5. 具体的 GPU 硬件信息
        device_count = torch.cuda.device_count()
        print(f"\n发现 {device_count} 个 GPU 设备:")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            name = props.name
            # 计算能力 (Compute Capability) 即硬件的 SM 版本
            major = props.major
            minor = props.minor
            total_memory = props.total_memory

            print(f"  [GPU {i}] -> 型号: {name}")
            print(f"      硬件算力 (Compute Capability): {major}.{minor}")
            print(f"      显存大小: {get_size(total_memory)}")

            # 简单的张量测试
            try:
                x = torch.tensor([1.0]).to(f"cuda:{i}")
                print("      测试: 张量写入显存成功")
            except Exception as e:
                print(f"      测试: ❌ 写入失败 ({e})")
    else:
        print("\n未检测到可用 GPU，或未安装 CUDA 版本的 PyTorch。")


if __name__ == "__main__":
    print_system_info()