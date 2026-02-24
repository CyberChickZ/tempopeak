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
- 运行合成数据训练测试：`python train.py`
  - 预期：Loss 快速下降，Acc 接近 1.0

## SAM3 Mask 提取器
- 提取 mask/centroid/box/score（仅 JSON）：
  ```bash
  python scripts/sam3_mask_extractor.py
  ```
- 同时渲染带标注 MP4：
  ```bash
  python scripts/sam3_mask_extractor.py --vis
  ```
- 输出目录：`outputs/sam3_mask_extractor/`
  - `tracks.json`：每帧所有 object 的 `{prompt, score, centroid, box}`
  - `vis.mp4`：带红/橙圆点标注的可视化视频（仅 `--vis` 时生成）

## SAM3 Web 标注工具
- 启动无状态后端：
  ```bash
  cd /Users/harryzhang/git/tempopeak/sam3_annotator/backend
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8080
  ```
- 访问：打开浏览器 `http://127.0.0.1:8080/`。支持按照视频名称自动加载 `00001.mp4`, `00001.json`, `00001.npz`，并提供拖拽式标注可视化和修改。

## 单帧 PCS Text-Only 测试
- 对帧 `[0,5,10,15,20]` 检测 racket：
  ```bash
  python scripts/smoke_text_only.py
  ```
- 输出：`outputs/pcs_samples/frame_XXXX_text_only.jpg`
