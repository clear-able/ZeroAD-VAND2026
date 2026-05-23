# ZeroAD 零样本：基于 CLIP 特征的异常检测

VAND 4.0 挑战赛 - 工业赛道（零样本类别）官方提交。

## 方法概述

ZeroAD 是一个参数高效的零样本异常检测框架，使用冻结的 CLIP ViT-L/14@336px 特征。方法：

1. **冻结特征提取**：从 CLIP 的 5 个中间层（12, 15, 18, 21, 24）提取多尺度特征
2. **轻量评分头**：仅训练 10 个小线性层（768→1），用于异常评分：
   - 5 个图像级分类器（每层一个）
   - 5 个像素级检测器（每层一个）
3. **多层融合**：对 5 个特征层的异常分数取平均
4. **双分支推理**：融合图像级分类和像素级分割分数

**总可训练参数量：7,690（0.0077M）**

## 零样本声明

- **训练过程中未使用任何 MVTec AD 2 图像**
- 预训练模型：CLIP ViT-L/14@336px（OpenAI，在 LAION-400M / WebImageText 上预训练）
- 训练数据：仅使用 MVTec AD（测试集）或 VisA（测试集）
- 超参数：固定值（未在 MVTec AD 2 上调参）

## 环境安装

### 依赖
- Python 3.9+
- CUDA 11.8+（或 CPU 模式）
- GPU：推荐 RTX 3090 / A100 或同等显存

### 安装步骤

```bash
git clone https://github.com/clear-able/ZeroAD-VAND2026.git
cd zero-shot
pip install -r requirements.txt
```

CLIP 模型（约 890MB）首次运行时会自动下载到 `~/.cache/clip/`。

## 快速开始

### 在 MVTec2 上测试（计算 SegF1）

```bash
# 使用 MVTec 训练的权重（推荐）
python test.py \
  --weight weights/uniadet_mvtec_final.pth \
  --data_dir /path/to/mvtec2_dataset \
  --image_size 518

# 输出：每个类别的 SegF1 和平均值
```

### 生成提交文件

```bash
# 对 test_private 和 test_private_mixed 推理，保存 float16 .tiff 异常图
python submit.py \
  --weight weights/uniadet_mvtec_final.pth \
  --data_dir /path/to/mvtec2_dataset \
  --output_dir ./submission

# 输出目录结构: submission/anomaly_images/{类别}/{split}/*.tiff
# 压缩提交: tar -czf submission.tar.gz -C submission anomaly_images/
```

### 从零训练（可选）

```bash
# 在 MVTec AD 测试集上训练
python train.py \
  --dataset mvtec \
  --test_dataset mvtec \
  --data_dir /path/to/datasets \
  --epoch 15 \
  --learning_rate 0.001 \
  --batch_size 8

# 在 VisA 测试集上训练
python train.py \
  --dataset visa \
  --test_dataset mvtec \
  --data_dir /path/to/datasets
```

## 结果

### MVTec2 test_public 上的 SegF1

| 训练数据 | 平均 SegF1 | 最好类别 | 最差类别 |
|---|---|---|---|
| MVTec AD（测试集） | **22.9%** | rice (43.8%) | can (0.4%) |
| VisA（测试集） | 4.6% | vial (13.9%) | can (0.1%) |

### 各类别详细结果（MVTec 训练）

| 类别 | SegF1 |
|---|---|
| can | 0.4% |
| fabric | 7.1% |
| fruit_jelly | 32.4% |
| rice | 43.8% |
| sheet_metal | 15.6% |
| vial | 29.7% |
| wallplugs | 8.0% |
| walnuts | 46.0% |
| **平均** | **22.9%** |

## 架构细节

### 特征提取
- **骨干网络**：CLIP ViT-L/14@336px（冻结）
- **使用层**：[12, 15, 18, 21, 24]
- **特征维度**：768（投影后）
- **Patch 网格**：37×37 = 1,369 patches/图

### 评分模块
每层有：
```
AnomalyScorer_CLS:  Linear(768 → 1)  # 图像级二分类
AnomalyScorer_SEG:  Linear(768 → 1)  # 每 patch 异常评分
```

### 推理流程
1. 从 5 层提取特征
2. L2 归一化 CLS 和 patch 特征
3. 每层独立分类
4. 生成像素级异常图（上采样到 518×518）
5. 对 5 层分数取平均
6. 融合图像级和像素级分数：`(1-0.5)×cls + 0.5×max(seg_map)`
7. 高斯平滑（σ=4）

## 文件结构

```
zero-shot/
├── README.md                      # 英文文档
├── README_zh.md                   # 中文文档（本文件）
├── LICENSE                        # CC BY-NC 4.0
├── requirements.txt               # Python 依赖
├── model.py                       # ZeroAD 模型定义
├── train.py                       # 训练和评估入口
├── test.py                        # MVTec2 test_public 评测
├── submit.py                      # 生成提交 .tiff 文件
├── clip/                          # CLIP 视觉编码器
│   ├── __init__.py
│   ├── clip.py                    # CLIP 加载器（自动下载 ViT-L/14@336px）
│   └── model.py                   # CLIP 架构（支持中间层特征提取）
├── weights/
│   ├── README.md                  # 权重说明
│   ├── uniadet_mvtec_final.pth    # MVTec 训练权重（38 KB）
│   └── uniadet_visa_final.pth     # VisA 训练权重（38 KB）
└── report/
    ├── main.tex                   # 技术报告 LaTeX 源码
    ├── main.bib                   # 参考文献
    ├── preamble.tex               # LaTeX 配置
    └── cvpr.sty                   # CVPR 2026 样式文件
```

## 可调参数

### test.py
- `--weight`：权重文件路径
- `--image_size`：输入图像尺寸（默认 518）
- `--sigma`：高斯平滑核大小（默认 4）
- `--data_dir`：数据集根目录

### train.py
- `--dataset`：训练数据集（`mvtec` 或 `visa`）
- `--test_dataset`：测试数据集（`mvtec` 或 `visa`）
- `--epoch`：训练轮数（默认 15）
- `--learning_rate`：Adam 学习率（默认 0.001）
- `--batch_size`：批大小（默认 8）
- `--save_freq`：保存频率（默认每 5 轮）

## 许可证

代码采用 **CC BY-NC 4.0** 许可证。详见 `LICENSE`。

CLIP 模型由 OpenAI 以 MIT 许可证发布。

## 联系方式

Long Chen  
广东人工智能与数字经济实验室（深圳）/ 深圳大学  
chenlong@gml.ac.cn
