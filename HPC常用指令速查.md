# 常用指令（HPC / tempopeak）

## 地址与目录
- 项目目录：`/nfs/hpc/share/zhanhaoc/hpe/tempopeak`
- 进入项目：`cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak`

## 环境与 source
- 初始化 conda：`source /nfs/stak/users/zhanhaoc/hpc-share/conda/bin/activate`
- 激活环境：`conda activate sam_3d_body`
- 当前环境名：`sam_3d_body`
- 复制：
```bash
source /nfs/stak/users/zhanhaoc/hpc-share/conda/bin/activate
conda activate sam_3d_body
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak

srun --gres=gpu:1 --mem=64G --pty bash
```

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

## SAM3 Mask 提取器（单视频推理）

脚本：`scripts/sam3_mask_extractor.py`（单视频模式，每次处理一个 mp4）

### 单视频运行命令
```bash
CUDA_VISIBLE_DEVICES=0 python /nfs/hpc/share/zhanhaoc/hpe/tempopeak/scripts/sam3_mask_extractor.py \
  --hf_local_model /nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7 \
  --video_name 00001 \
  --video_path /nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4 \
  --out_dir /nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_mask_extractor \
  --prompts ball racket \
  --dtype bf16 \
  --tracker_score_min 0.10 \
  --mask_area_min 1 \
  --print_every 30
```

### 开启后处理（含 mask IoU 静态检测）
```bash
CUDA_VISIBLE_DEVICES=0 python /nfs/hpc/share/zhanhaoc/hpe/tempopeak/scripts/sam3_mask_extractor.py \
  --hf_local_model /nfs/hpc/share/zhanhaoc/hpe/tempopeak/models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7 \
  --video_name 00001 \
  --video_path /nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00001.mp4 \
  --out_dir /nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/sam3_mask_extractor \
  --prompts ball racket \
  --dtype bf16 \
  --tracker_score_min 0.10 \
  --mask_area_min 1 \
  --post_process_rm --rm_min_len 15 --rm_static_px 5.0 --rm_static_iou 0.85 \
  --post_process_fusion --fusion_max_gap 5 --fusion_skip_unknown \
  --post_process_predict --predict_max_gap 15 \
  --print_every 30 \
  --vis
```

### 后处理关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rm_min_len` | 15 | track 帧数少于此值直接删除 |
| `--rm_static_px` | 5.0 | centroid 逐帧平均移动 ≤ 此值视为静止（兜底） |
| `--rm_static_iou` | 0.85 | **相邻帧 mask IoU ≥ 此值视为静止并删除**（主判断）。设为 1.0 可禁用 |
| `--fusion_max_gap` | 5 | 同 label track 间隔 < 此帧数则合并 |

### 输出文件

| 文件 | 内容 |
|---|---|
| `{video_name}.json` | 含 `_meta` + 每帧 track 数据 |
| `{video_name}.npz` | `masks [M,H,W] bool` + `frame_indices [M]` + `object_ids [M]` |
| `{video_name}_vis.mp4` | 可视化视频（仅 `--vis` 时生成） |

## YOLO Clip Extractor (自动截取击球片段)
此脚本会自动扫描文件夹中的 mp4，使用 YOLO 识别球和球拍的距离，检测击球事件 (hit event)，并自动切出包含击球瞬间的前后短视频 (clip)。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/yolo_clip_extractor.py \
  --folder /nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve \
  --tmp_dir /nfs/hpc/share/zhanhaoc/hpe/tempopeak/outputs/yolo_clips \
  --model yolo26x.pt \
  --conf 0.25
```
- 默认会跳过非 15/30/60 fps 的视频。
- 切好的 clip 会存放在 `--tmp_dir` 目录下。

## SAM3 Web 标注工具
- 启动无状态后端：
  ```bash
  cd /Users/harryzhang/git/tempopeak/sam3_annotator/backend
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload
  ```

  ```bash
  cd /Users/harryzhang/git/tempopeak/sam3_annotator/backend
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload
  ```
- 访问：打开浏览器 `http://127.0.0.1:8080/`。支持按照视频名称自动加载 `00001.mp4`, `00001.json`, `00001.npz`，并提供拖拽式标注可视化和修改。

## 单帧 PCS Text-Only 测试
- 对帧 `[0,5,10,15,20]` 检测 racket：
  ```bash
  python scripts/smoke_text_only.py
  ```
- 输出：`outputs/pcs_samples/frame_XXXX_text_only.jpg`
