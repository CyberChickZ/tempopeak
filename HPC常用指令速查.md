# 常用指令（HPC / tempopeak）

## 地址与目录
- 项目目录：`/nfs/hpc/share/zhanhaoc/hpe/tempopeak`
- 进入项目：`cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak`

## 环境与 source
- 初始化 conda：`source /nfs/stak/users/zhanhaoc/hpc-share/conda/bin/activate`
- 激活环境：`conda activate sam_3d_body`
- 当前环境名：`sam_3d_body`

## 检查命令（版本/设备）
- 检查 mamba-ssm 版本：`python -c "import mamba_ssm; print(mamba_ssm.__version__)"`
- 检查 PyTorch/CUDA/GPU：`python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"`

## 运行文件与作用
- 编辑测试脚本：`code smoke_mamba.py`
  - 作用：打开并修改 Mamba 冒烟测试脚本。
- 运行冒烟测试：`python smoke_mamba.py`
  - 作用：验证 `Mamba(d_model=128)` 前向可运行，且输入输出形状一致。
- 运行性能基准：
  ```bash
  python - <<EOF
  import torch
  from mamba_ssm import Mamba
  import time

  B,T,D = 8,64,128
  x = torch.randn(B,T,D).cuda()
  m = Mamba(d_model=D).cuda()

  torch.cuda.synchronize()
  t0=time.time()
  for _ in range(50):
      y = m(x)
  torch.cuda.synchronize()
  print("avg time:", (time.time()-t0)/50)
  EOF
  ```
  - 作用：计算平均前向耗时，快速评估时序模块计算开销。

## 新增模型 (ResNet+BiMamba)
- 运行新模型冒烟测试：`python test_model.py`
  - 预期输出：`output shape: [B, T]`，`sum over time: 1.0`
