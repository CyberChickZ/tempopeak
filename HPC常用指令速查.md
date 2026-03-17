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

## Backbone 权重缓存
- 缓存目录：`/nfs/stak/users/zhanhaoc/.cache/torch/hub/checkpoints/`
- `resnet34-b627a593.pth` (83.3M) — 首次 `extract_features.py --backbone=resnet34` 时自动下载
- `vit_b_16-c867db91.pth` (330M) — 首次 `extract_features.py --backbone=vit_small` 时自动下载
- ResNet-18 权重也已缓存（之前训练时下载）

## 数据部署（Mac → HPC）
`datasets/v1/` 在 `.gitignore` 中，数据不走 git，通过**网盘**传输。

### Mac 端打包
```bash
cd /Users/harryzhang/git/tempopeak
# 只打 frames + annot.json，排除 features/*.pt（HPC CUDA 重新提取更快）
COPYFILE_DISABLE=1 tar -czf datasets_v1_export.tar.gz --exclude='features' datasets/v1/export/
# 输出: datasets_v1_export.tar.gz (~2.6 GB)
```
⚠️ 不加 `COPYFILE_DISABLE=1` 会产生 macOS xattr 警告（`LIBARCHIVE.xattr.com.apple.provenance`），不影响数据但 HPC 会刷屏 warning。

### HPC 端解压
```bash
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak
tar -xzf datasets_v1_export.tar.gz
# 验证
ls datasets/v1/export/0001/ | wc -l    # 应为 27
find datasets/v1/export -name "*.jpg" | wc -l   # 应为 14946
```

### 数据规模
| 内容 | 数量 | 大小 |
|---|---|---|
| clips | 27 | — |
| JPG frames | 14946 (含双 hitter) | ~5.2 GB |
| annot.json | 27 | ~3.2 MB |
| tar.gz 压缩包 | 1 | ~2.6 GB |

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

## 训练 Pipeline — Temporal Head 比较实验

### 文件结构
| 文件 | 作用 |
|---|---|
| `train.py` | 训练入口 |
| `dataloader.py` | ClipDataset（支持 annot.json 导出格式 + 原始 .json+.mp4 格式） |
| `model.py` | TempoPeakModel：frozen ResNet-18 → temporal head → Linear → logits |
| `temporal_heads.py` | 6 种 temporal head：identity, bilstm, mstcn, transformer, mamba2, bimamba2 |
| `eval.py` | 指标计算：MAE, Acc@1/3/5, Entropy；Gaussian soft CE loss |
| `config.py` | argparse 配置 |

### Mac 本地冒烟测试
```bash
cd /Users/harryzhang/git/tempopeak

# 单头快速验证
python3 train.py --temporal_head=identity --t_max=32 --data_dir=datasets/v1/clips --epochs=3 --batch_size=4

# 全部 6 heads 冒烟 (1 epoch each)
for head in identity bilstm mstcn transformer mamba2 bimamba2; do
    echo "=== Testing $head ==="
    python3 train.py --temporal_head=$head --t_max=32 --data_dir=datasets/v1/clips --epochs=1 --batch_size=4
done
```

### HPC 训练命令

#### Exp A — 6 Temporal Heads 对比 (t_max=32, 100 epochs)
```bash
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak
for head in identity bilstm mstcn transformer mamba2 bimamba2; do
    echo "=== Training $head ==="
    python train.py --temporal_head=$head --t_max=32 --data_dir=datasets/v1/clips --epochs=100
done
```

#### Exp B — T_max Sweep (best head)
```bash
for tmax in 16 32 64; do
    python train.py --temporal_head=BEST_HEAD --t_max=$tmax --data_dir=datasets/v1/clips --epochs=100
done
```

### train.py 参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--temporal_head` | `identity` | `{identity, bilstm, mstcn, transformer, mamba2, bimamba2}` |
| `--t_max` | `32` | clip 长度 `{16, 32, 64}` |
| `--target_type` | `gaussian` | `{onehot, gaussian}` |
| `--sigma` | `2.0` | Gaussian soft label 标准差 |
| `--lr` | `1e-3` | 学习率 |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `8` | batch size |
| `--data_dir` | (required) | 数据根目录 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | `auto` = cuda if available else cpu |

### 输出
- Checkpoint 保存在 `checkpoints/`
  - `best_{head}_{t_max}.pt` — 按 **Acc@1** 选出的最优模型
  - `last_{head}_{t_max}.pt` — 最后一个 epoch
- Metrics 每 epoch 打印：`train_loss | MAE Acc@1 Acc@3 Acc@5 Entropy | best_Acc@1`

## 旧版冒烟测试 (已被 train.py 取代)
- ~~运行新模型冒烟测试：`python test_model.py`~~
- ~~运行合成数据训练测试：旧 `python train.py`~~

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
  cd /Users/harryzhang/git/tempopeak/sam3_annotator/server
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload
  ```

  ```bash
  cd /Users/harryzhang/git/tempopeak/sam3_annotator/server
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload
  ```
- 访问：打开浏览器 `http://127.0.0.1:8080/`。支持按照视频名称自动加载 `00001.mp4`, `00001.json`, `00001.npz`，并提供拖拽式标注可视化和修改。

## 单帧 PCS Text-Only 测试
- 对帧 `[0,5,10,15,20]` 检测 racket：
  ```bash
  python scripts/smoke_text_only.py
  ```
- 输出：`outputs/pcs_samples/frame_XXXX_text_only.jpg`

## Temporal Head 训练 (SSM Validation)

### Mac 本地冒烟 (CPU, Feature-Mode)
```bash
cd /Users/harryzhang/git/tempopeak

# 单 head 冒烟 (resnet18 features)
python3 train.py --temporal_head=identity --t_max=32 \
  --data_dir=datasets/v1/export --backbone=resnet18 --epochs=1 --batch_size=4

# 4 heads × 指定 backbone 全部冒烟
for head in identity bilstm mamba2 bimamba2; do
    echo "=== $head ==="
    python3 train.py --temporal_head=$head --t_max=32 \
      --data_dir=datasets/v1/export --backbone=resnet18 --epochs=1 --batch_size=4
done

# 3 backbones × 4 heads 完整冒烟
for bb in resnet18 resnet34 vit_small; do
    for head in identity bilstm mamba2 bimamba2; do
        echo "=== $bb / $head ==="
        python3 train.py --temporal_head=$head --t_max=32 \
          --data_dir=datasets/v1/export --backbone=$bb --epochs=1 --batch_size=4
    done
done
```

### HPC 正式训练 (CUDA, Exp A — 4 heads × 3 backbones × 100 epochs)

先提取所有 backbone 的特征（HPC CUDA 上很快）：
```bash
source /nfs/stak/users/zhanhaoc/hpc-share/conda/bin/activate
conda activate sam_3d_body
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak
git pull
srun --gres=gpu:1 --mem=64G --pty bash

# 提取 3 种 backbone 特征
for bb in resnet18 resnet34 vit_small; do
    echo "=== Extracting $bb ==="
    python extract_features.py --data_dir=datasets/v1/export --backbone=$bb --batch_size=64
done
```

然后训练（特征模式，秒级/epoch）：
```bash
# Exp A v1: 4 heads × 3 backbones (已完成 resnet18 × 4 heads)
for bb in resnet18 resnet34 vit_small; do
    for head in identity bilstm mamba2 bimamba2; do
        echo "=== $bb / $head ==="
        python train.py --temporal_head=$head --t_max=32 \
          --data_dir=datasets/v1/export --backbone=$bb --epochs=100 --batch_size=8
    done
done
```

### HPC 正式训练 (Exp A v3 — DataLoader v3 + Cosine LR)

```bash
source /nfs/stak/users/zhanhaoc/hpc-share/conda/bin/activate
conda activate sam_3d_body
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak
git pull
srun --gres=gpu:1 --mem=64G --pty bash

# Exp A v3: vit_small × 4 heads × 100 epochs
# Feature preload (RAM) + Acc@1 best criterion + CosineAnnealingLR(3e-3 → 1e-5)
# batch_size=256 充分利用 H100，lr=3e-3（linear scaling: batch ×8 → lr ×√8 ≈ ×3）
for head in identity bilstm mamba2 bimamba2; do
    echo "=== vit_small / $head ==="
    python train.py --temporal_head=$head --t_max=32 \
      --data_dir=datasets/v1/export --backbone=vit_small \
      --epochs=100 --batch_size=256 --lr=3e-3
done
```

### HPC Exp A v3.1 — LR 调优 + Gradient Clipping (30 epochs)

```bash
source /nfs/stak/users/zhanhaoc/hpc-share/conda/bin/activate
conda activate sam_3d_body
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak
git pull
srun --gres=gpu:1 --mem=64G --pty bash

# SSM heads 用更小 LR（3e-3 导致 BiMamba2 爆炸）
for head in mamba2 bimamba2; do
    echo "=== vit_small / $head (lr=3e-4) ==="
    python train.py --temporal_head=$head --t_max=32 \
      --data_dir=datasets/v1/export --backbone=vit_small \
      --epochs=30 --batch_size=256 --lr=3e-4
done

# BiLSTM 稍降 LR 防 epoch 24 爆炸
python train.py --temporal_head=bilstm --t_max=32 \
  --data_dir=datasets/v1/export --backbone=vit_small \
  --epochs=30 --batch_size=256 --lr=1e-3
```

注意：Identity 不用重跑，41.4% 是其 capacity ceiling。

### HPC Exp B — T_max Sweep (用 Exp A best head + best backbone)
```bash
BEST_HEAD=bimamba2   # ← 替换为 Exp A 的 best head
BEST_BB=vit_small    # ← 替换为 Exp A 的 best backbone
for tmax in 16 32 64; do
    echo "=== T_max=$tmax ==="
    python train.py --temporal_head=$BEST_HEAD --t_max=$tmax \
      --data_dir=datasets/v1/export --backbone=$BEST_BB --epochs=100 --batch_size=8
done
```

### 训练参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--temporal_head` | identity | `{identity, bilstm, mstcn, transformer, mamba2, bimamba2}` |
| `--t_max` | 32 | 每个样本的帧窗口长度 `{16, 32, 64}` |
| `--target_type` | gaussian | `{gaussian, onehot}` |
| `--sigma` | 2.0 | Gaussian soft label 标准差 |
| `--lr` | 1e-3 | AdamW 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--epochs` | 100 | |
| `--batch_size` | 8 | |
| `--data_dir` | (必填) | 数据根目录，支持 export 或 raw 格式 |
| `--seed` | 42 | 随机种子 |

### 新增参数 (v2+)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backbone` | `resnet18` | `{resnet18, resnet34, vit_small}` — 用于查找 features/ 目录 |
| `--no_features` | (flag) | 强制 image mode，不使用预提取特征 |

### DataLoader 版本说明

| 版本 | 特征 | Train Samples (resnet18) |
|---|---|---|
| v2 | `AUG_K_TRAIN=5`，每 hit 注册 5 个随机窗口 | ~890 |
| **v3 (当前)** | `_enumerate_windows()` 枚举所有合法窗口，`MAX_WINDOWS_PER_HIT=50` | **~5996** |

v3 变化：`__getitem__` 零随机性，所有 `(start, n)` 在 `_build_samples()` 时确定。Val 使用最长窗口 + hit 居中，每 hit 1 个确定性样本。

### DataLoader I/O 参数（硬编码，非命令行参数）

| 参数 | Train | Val | 说明 |
|---|---|---|---|
| `preload` | True | True | 启动时一次性加载所有 `.pt` 特征到 RAM（~92MB），`__getitem__` 零磁盘 I/O |
| `num_workers` | 0 | 0 | 数据全在内存，无需多进程预取 |
| `pin_memory` | True | True | 锁页内存 DMA 直传，CPU→GPU 传输快 30-50% |

preload + num_workers=0 对训练结果零影响，仅加速数据加载吞吐。Probe 数据集 `preload=False`（仅读 clip_ids，不需加载特征）。

### Scheduler + Gradient Clipping

- v3 起默认使用 `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)`
- LR 从 `--lr`（默认 1e-3）余弦衰减至 1e-5
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)` — SSM/RNN 标配，防止 BiMamba2 数值爆炸（loss.backward() 后、optimizer.step() 前）

### Checkpoint 输出
- Best model: `checkpoints/best_{head}_{t_max}.pt` — 按 **Acc@1** 选出（非 MAE）
- Last model: `checkpoints/last_{head}_{t_max}.pt`
- Epoch 日志格式：`train_loss | MAE Acc@1 Acc@3 Acc@5 Entropy | best_Acc@1`（MAE 仅诊断用，不决定 checkpoint）

## 预提取特征 (extract_features.py)

训练前先提取 backbone 特征，避免每 epoch 重复过 ResNet/ViT。

### 目录结构
```
export/0001/00001/
├── frames/                          # 原始 JPEG
├── features/
│   ├── resnet18/p1/*.pt             # ResNet-18 特征 [512]
│   ├── resnet18/p2/*.pt
│   ├── resnet34/p1/*.pt             # ResNet-34 特征 [512]
│   └── vit_small/p1/*.pt            # ViT-Small 特征 [768]
└── annot.json
```

### Mac 本地提取
```bash
cd /Users/harryzhang/git/tempopeak

# ResNet-18 (D=512)
python3 extract_features.py --data_dir=datasets/v1/export --backbone=resnet18 --batch_size=16

# ResNet-34 (D=512)
python3 extract_features.py --data_dir=datasets/v1/export --backbone=resnet34 --batch_size=16

# ViT-Small (D=768)
python3 extract_features.py --data_dir=datasets/v1/export --backbone=vit_small --batch_size=8
```

### HPC 提取 (CUDA)
```bash
cd /nfs/hpc/share/zhanhaoc/hpe/tempopeak
python extract_features.py --data_dir=datasets/v1/export --backbone=resnet18 --batch_size=64
python extract_features.py --data_dir=datasets/v1/export --backbone=vit_small --batch_size=32
```

### 特征模式训练
```bash
# 自动检测 features/ 目录（如果存在则跳过 backbone）
python train.py --temporal_head=identity --t_max=32 --data_dir=datasets/v1/export --backbone=resnet18

# 强制 image mode（忽略预提取特征）
python train.py --temporal_head=identity --t_max=32 --data_dir=datasets/v1/export --no_features
```

### extract_features.py 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | (必填) | 数据根目录，递归查找 annot.json |
| `--backbone` | `resnet18` | `{resnet18, resnet34, vit_small}` |
| `--batch_size` | `32` | 每批处理帧数 |
| `--device` | `auto` | `auto` = cuda > cpu |
