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

## SAM3 Mask 提取器（v2 — 全量重写）

脚本：`scripts/sam3_mask_extractor.py`（完整 argparse CLI，无需改源码即可调参）

### 最小运行（默认参数）
```bash
python scripts/sam3_mask_extractor.py
```

### 自定义视频 + 渲染可视化
```bash
python scripts/sam3_mask_extractor.py \
  --video_name 00002 \
  --video_path /nfs/hpc/share/zhanhaoc/hpe/tempopeak/datasets/serve/00002.mp4 \
  --vis
```

### 指定 ID→Label 映射（推荐，避免歧义）
```bash
python scripts/sam3_mask_extractor.py \
  --obj_id_to_label "0:ball,1:ball,2:racket" \
  --vis
```

### 开启空间跳变拦截（抗 ID Switch）
```bash
python scripts/sam3_mask_extractor.py \
  --max_jump_px 300 \
  --ema_alpha 0.7 \
  --predict_on_reject \
  --max_lost 3 \
  --vis
```

### 完整参数表

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hf_local_model` | HPC Snapshots 路径 | SAM3 本地权重目录 |
| `--video_name` | `00001` | 输出文件名前缀 |
| `--video_path` | HPC serve/00001.mp4 | 输入视频 |
| `--out_dir` | `outputs/sam3_mask_extractor/` | 输出目录 |
| `--vis` | off | 是否渲染带标注 MP4 |
| `--prompts` | `ball racket` | 传给 SAM3 PCS 的文字 prompt 列表 |
| `--dtype` | `bf16` | 推理精度（bf16/fp16/fp32） |
| `--max_frames` | `-1`（全视频）| 限制处理帧数，调试用 |
| `--tracker_score_min` | `0.10` | 低于此 tracker score 的 mask 丢弃 |
| `--static_score_min` | `-1.0`（关闭） | 低于此 static score 的 mask 丢弃 |
| `--quality_score_mode` | `mul` | `quality_score` 的计算方式 (`none`/`mul`/`min`) |
| `--mask_area_min` | `1` | mask 像素数低于此值的丢弃 |
| `--max_jump_px` | `-1`（关闭）| 相邻帧 centroid 欧氏距离超过此值则视为 ID Switch，丢弃 |
| `--ema_alpha` | `1.0`（关闭 EMA）| centroid EMA 平滑系数（0.5~0.8 为典型值） |
| `--max_lost` | `0` | 若为 0，则纯静态丢弃；若 >0 则允许在连续丢失 N 帧内通过速度外推状态 |
| `--predict_on_reject` | off | 当拒绝当前帧且 `max_lost>0` 时，利用上一帧的速度 (vx, vy) 外推 centroid |
| `--print_every` | `30` | 提取过程中每 N 帧打印一次进度和保留 mask 数量 |
| `--debug_first_frames` | `1` | 打印前 K 帧的 PP `prompt_to_obj_ids` 映射信息供调试 |

### 输出文件

| 文件 | 内容 |
|---|---|
| `{video_name}.json` | 含 `_meta`（完整运行参数），及 `"0"~"N"` 帧数据 |
| `{video_name}.npz` | `masks [M,H,W] bool` + `frame_indices [M] int32` + `object_ids [M] int32` |
| `{video_name}_vis.mp4` | 带遮罩/BBox/ID/score OSD 的可视化视频（仅 `--vis` 时） |

### JSON schema（单帧）
```json
{
  "_meta": { "tracker_score_min": 0.1, "max_jump_px": 300, ... },
  "42": {
    "0": {
      "label": "ball",
      "tracker_score": 0.823,
      "static_score": 0.911,
      "quality_score": 0.749753,
      "centroid": [312.5, 204.1],
      "box_xyxy": [298, 190, 327, 218],
      "mask_idx": 77
    }
  }
}
```

### Annotator 后端注意事项
`io_sam3.py` 已同步支持 v2 schema（`_meta` 跳过，`label`/`tracker_score`/`box_xyxy` 字段读写），同时向后兼容 v1 的 `prompt`/`score`/`box`。

### 已知 Bugfix（已入库）
- **`torch.where unpack error`**：模型输出的 `obj_id_to_mask[obj_id]` 实际 shape 为 `[1, H, W]`（带 batch 维），直接解包到 `ys, xs` 会 crash。
  修法：取 mask 时调用 `.squeeze()` 内联处理：
  ```python
  mask = obj_id_to_mask[obj_id].squeeze()  # Tensor[1,H,W] <BOOL> -> [H,W]
  ```

## SAM3 Web 标注工具
- 启动无状态后端：
  ```bash
  cd /Users/harryzhang/git/tempopeak/sam3_annotator/backend
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8080 --reload
  ```
- 访问：打开浏览器 `http://127.0.0.1:8080/`。支持按照视频名称自动加载 `00001.mp4`, `00001.json`, `00001.npz`，并提供拖拽式标注可视化和修改。

## 单帧 PCS Text-Only 测试
- 对帧 `[0,5,10,15,20]` 检测 racket：
  ```bash
  python scripts/smoke_text_only.py
  ```
- 输出：`outputs/pcs_samples/frame_XXXX_text_only.jpg`
