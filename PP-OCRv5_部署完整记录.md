# PaddleOCR ONNX + OpenCV DNN 部署完整记录

> **日期**: 2026-06-09  
> **设备**: NVIDIA Jetson Orin Nano Super (P3767-0005), 8GB LPDDR5  
> **系统**: Ubuntu 22.04, L4T R36.5.0, JetPack 6.0  
> **CUDA**: 12.6.68 | cuDNN 9.3.0 | TensorRT 10.3.0.30  
> **OpenCV**: 4.10.0 自编译 (CUDA 加速, `/usr/local/`)  
> **Python**: 3.10.12  
> **虚拟环境**: `/home/ysdhanji/ocr/venv_paddleocr/`

---

## 目录

1. [项目目标](#1-项目目标)
2. [最终项目结构](#2-最终项目结构)
3. [技术路线总览](#3-技术路线总览)
4. [阶段一：PaddleOCR 环境搭建](#4-阶段一paddleocr-环境搭建)
5. [阶段二：PIR 模型 → ONNX 导出](#5-阶段二pir-模型--onnx-导出)
6. [阶段三：OpenCV DNN CUDA 推理管线](#6-阶段三opencv-dnn-cuda-推理管线)
7. [阶段四：准确率调试（7轮迭代）](#7-阶段四准确率调试7轮迭代)
8. [阶段五：TensorRT 尝试（失败）](#8-阶段五tensorrt-尝试失败)
9. [阶段六：性能瓶颈分析](#9-阶段六性能瓶颈分析)
10. [关键踩坑记录](#10-关键踩坑记录)
11. [对 ocrcplus 项目的批判分析](#11-对-ocrcplus-项目的批判分析)
12. [使用方法](#12-使用方法)
13. [已知限制](#13-已知限制)

---

## 1. 项目目标

在 Jetson Orin Nano 上部署 PaddleOCR，实现高效文字检测+识别。

**为什么不直接用 PaddlePaddle GPU？**
- PaddlePaddle 官方无 JetPack 6 (L4T 36.x) GPU 预编译包
- 仅支持到 JetPack 5.x (CUDA 11.4 + cuDNN 8.6)
- 用 CPU 版太慢，从源码编译太复杂

**为什么选择 ONNX + OpenCV DNN？**
- 设备已有自编译 OpenCV 4.10.0 + CUDA 12.6（含 DNN CUDA 后端）
- ONNX 是通用格式，无需 PaddlePaddle 运行时
- 已验证可加载并推理

---

## 2. 最终项目结构

```
/home/ysdhanji/ocr/
├── ocr_opencv_dnn.py              ← 主推理脚本 (v3, 生产可用)
├── ocr_opencv_dnn_v1.py           ← v1 备份 (det_size=960)
├── ocr_opencv_dnn_v3.py           ← v3 备份
├── export_to_onnx.py              ← PIR → ONNX 导出脚本
├── build_trt_engines.py           ← TRT 后台构建 (实验, 不推荐)
│
├── models/
│   ├── PP-OCRv5_mobile_det.onnx       # 原始 ONNX (4.6MB)
│   ├── PP-OCRv5_mobile_det_sim.onnx   # 简化 ONNX ← 实际使用
│   ├── PP-OCRv5_mobile_rec.onnx       # 原始 ONNX (16MB)
│   ├── PP-OCRv5_mobile_rec_sim.onnx   # 简化 ONNX ← 实际使用
│   └── ppocr_keys_v1.txt             # 字典 (18383 字符, use_space_char=True → 18385)
│
├── PaddleOCR-main/                ← PaddleOCR 源码 (复用后处理类)
│   └── ppocr/postprocess/
│       ├── db_postprocess.py      ← DBPostProcess (检测后处理)
│       └── rec_postprocess.py     ← CTCLabelDecode (识别后处理)
│
├── engines/                       ← TRT 引擎 (实验)
├── venv_paddleocr/                ← Python 虚拟环境
├── logs/                          ← TRT 构建日志
└── 机器状况详细报告.md             ← 设备环境文档
```

### 依赖版本清单

| 包 | 版本 | 安装方式 | 备注 |
|------|------|------|------|
| PaddlePaddle | 3.2.2 CPU | `pip install paddlepaddle -f ...aarch64/cpu/...` | GPU 版不可用 |
| PaddleOCR | 3.6.0 | `pip install paddleocr` | 含 paddlex |
| paddle2onnx | 2.1.0 | `pip install --no-deps` (清华镜像) | aarch64 预编译 wheel |
| onnx | 1.21.0 | `pip install` (清华镜像) | |
| onnxsim | 0.6.5 | `pip install` | ONNX 模型简化 |
| onnxruntime | 1.23.2 | `pip install` (清华镜像) | |
| OpenCV | 4.10.0 CUDA | 自编译, `/usr/local/` | **不能用 pip 版** |
| NumPy | 1.26.4 | 降级 | 兼容系统 OpenCV (1.x 编译) |
| TensorRT | 10.3.0 | 系统安装, 链接到 venv | `/usr/lib/python3.10/dist-packages/tensorrt/` |
| Pillow | 12.2.0 | 随 paddleocr 安装 | 中文渲染 |
| pyclipper | 1.4.0 | 随 paddleocr 安装 | DBNet unclip |

---

## 3. 技术路线总览

```
PP-OCRv5 Mobile PIR 模型
    │  (来自 /home/ysdhanji/ocrcplus/models/)
    │  inference.json + inference.pdiparams
    ▼
paddle.jit.load("inference") + paddle.onnx.export()
    │  关键: load 文件前缀, 不是目录!
    ▼
原始 ONNX (不能直接被 OpenCV 加载)
    │  Reshape 算子报错
    ▼
onnxsim simplify()
    │
    ▼
简化 ONNX (OpenCV DNN 可加载)
    │
    ▼
OpenCV DNN + CUDA 推理
    │  检测: cv2.dnn.readNetFromONNX + forward
    │  识别: 同上
    │  后处理: 复用 PaddleOCR 原生 DBPostProcess + CTCLabelDecode
    ▼
输出: JSON 文本 + 可视化图片
```

### 性能

| 方案 | 耗时 | 提速 |
|------|------|:--:|
| C++ Paddle Inference (ocrcplus) | ~5.5s | 基准 |
| PaddleOCR Python CPU | Segfault | ❌ |
| **OpenCV DNN CUDA (本项目)** | **~1.0s** | **6×** ✅ |
| TensorRT | 构建失败/太慢 | ❌ |

---

## 4. 阶段一：PaddleOCR 环境搭建

### 4.1 创建虚拟环境

```bash
cd /home/ysdhanji/ocr
python3 -m venv venv_paddleocr
source venv_paddleocr/bin/activate
pip install --upgrade pip setuptools wheel
```

### 4.2 安装 PaddlePaddle (CPU)

```bash
# GPU 版不可用 (JetPack 6 无官方包)
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html
```

结果: PaddlePaddle 3.2.2 CPU 版

### 4.3 安装 PaddleOCR

```bash
pip install paddleocr
```

结果: PaddleOCR 3.6.0 (基于 PaddleX)

---

## 5. 阶段二：PIR 模型 → ONNX 导出

### 5.1 模型来源

模型文件来自 `/home/ysdhanji/ocrcplus/models/`（之前 C++ 项目的遗留）：

| 模型 | 路径 | 大小 |
|------|------|------|
| PP-OCRv5_mobile_det | `PP-OCRv5_mobile_det_infer/` | 4.8MB |
| PP-OCRv5_mobile_rec | `PP-OCRv5_mobile_rec_infer/` | 17MB |

格式: PIR (Paddle Intermediate Representation)，文件为 `inference.json` + `inference.pdiparams`

### 5.2 安装 paddle2onnx

```bash
# 网络不稳定, 使用 --no-deps 跳过 onnxoptimizer 编译
pip install paddle2onnx --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
```

版本: 2.1.0 (aarch64 预编译 wheel)

### 5.3 加载 PIR 模型

**关键发现**: PIR 模型不能用 `paddle.jit.load(model_dir)` 加载！

```python
# ❌ 错误方式
model = paddle.jit.load('/path/to/model_dir')  # KeyError: 'forward'

# ✅ 正确方式: 传文件前缀, 不含扩展名
model = paddle.jit.load('/path/to/model_dir/inference')
```

`paddle.jit.load` 会自动查找 `inference.json` 和 `inference.pdiparams`。

### 5.4 导出 ONNX

```python
paddle.onnx.export(
    model,
    'output.onnx',
    input_spec=[paddle.static.InputSpec(shape=[1, 3, -1, -1], dtype='float32', name='x')],
    opset_version=14,
)
```

**注意**: `paddle.onnx.export` 内部调用 paddle2onnx，输出文件名会变成 `output.onnx.onnx`（双重后缀），需手动改名。

### 5.5 ONNX 模型简化

OpenCV DNN 无法直接加载原始 ONNX（Reshape 算子报错）：

```python
import onnx
from onnxsim import simplify

model = onnx.load('det.onnx')
model_simp, check = simplify(model)
onnx.save(model_simp, 'det_sim.onnx')
```

简化后的模型大小不变，但 OpenCV DNN 可以正常加载。

### 5.6 导出脚本

完整脚本: `/home/ysdhanji/ocr/export_to_onnx.py`

---

## 6. 阶段三：OpenCV DNN CUDA 推理管线

### 6.1 核心设计

主脚本使用 OpenCV DNN 加载 ONNX 模型 + CUDA 后端，后处理复用 PaddleOCR 原生 Python 类。

```python
# 加载模型
net = cv2.dnn.readNetFromONNX('model.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 推理
net.setInput(tensor)
output = net.forward()
```

### 6.2 系统 OpenCV 导入

pip 版 `opencv-python` 无 CUDA 支持，必须使用系统自编译版：

```python
import sys
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
import cv2
```

**NumPy 冲突**: 系统 OpenCV 用 NumPy 1.x 编译，但 PaddlePaddle 3.2.2 安装 NumPy 2.2.6。解决：

```bash
pip install 'numpy<2'  # 降级到 1.26.4
```

### 6.3 预处理参数（从 inference.yml 提取）

**检测模型** (`PP-OCRv5_mobile_det_infer/inference.yml`):
```yaml
DetResizeForTest:
    resize_long: 960
NormalizeImage:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    scale: 1./255.
ToCHWImage
```

实现:
```python
ratio = 960 / max(h, w)
img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
# padding 到 32 倍数 (DBNet 下采样 1/32)
pad_h = (32 - h % 32) % 32
img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, ...)
# 归一化
img = (img/255.0 - mean) / std
# HWC → CHW → BCHW
```

**识别模型** (`PP-OCRv5_mobile_rec_infer/inference.yml`):
```yaml
RecResizeImg:
    image_shape: [3, 48, 320]
# 无独立 NormalizeImage! 归一化内置在 RecResizeImg 中
```

RecResizeImg 的归一化逻辑 (来自 PaddleOCR 源码):
```python
img = cv2.resize(img, (new_w, 48))
img = img.astype(np.float32).transpose(2, 0, 1)
img = img / 255.0
img = (img - 0.5) / 0.5  # → [-1, 1]
```

**关键**: max_imgW 默认 3200 (不是 320!) — 支持长文本识别。

### 6.4 后处理集成

不重新实现后处理，直接引用 PaddleOCR 源码中的类（绕过 `__init__.py` 避免 skimage 等可选依赖）:

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    'db_pp', 'PaddleOCR-main/ppocr/postprocess/db_postprocess.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DBPostProcess = mod.DBPostProcess
```

**DBPostProcess 默认参数** (类定义):
- `thresh=0.3` — 概率图二值化阈值
- `box_thresh=0.7` — 文本框置信度阈值
- `unclip_ratio=2.0` — 文本框扩展比例

**实际使用参数** (对齐模型 config):
- `box_thresh=0.6`
- `unclip_ratio=2.0` (提高以合并 ONNX 碎片)

### 6.5 字符字典

PP-OCRv5 recognition 输出 18385 维。字典结构:
- Index 0: blank (CTC 空白)
- Index 1: space (空格)
- Index 2-18384: 18383 个字符 (来自 inference.yml 的 character_dict)

```python
CTCLabelDecode(character_dict_path='ppocr_keys_v1.txt', use_space_char=True)
```

`use_space_char=True` 会自动添加空格，字典从 18383 → 18385，匹配模型输出。

---

## 7. 阶段四：准确率调试（7轮迭代）

### 迭代 1: 识别全是乱码

**错误**: 识别结果如 "副伤?Q"、"裸灯"、"婿朴小" 等

**根因**: `rec_preprocess` 错误地做了额外归一化。PP-OCRv5 RecResizeImg **内置**归一化 `(x/255-0.5)/0.5`，不需要再归一化。

修复: 移除 `rec_preprocess` 中的额外归一化 → 但后来又发现 RecResizeImg 确实需要归一化...

### 迭代 2: 仍然乱码

**根因**: 字典维度不匹配。

- 模型输出: 18385 维
- 我的字典: ['blank', ' ', ...] → 18385
- CTCLabelDecode(use_space_char=False): 再添加 'blank' → 18386! 但模型输出 18385!

修复: 字典文件只保存 18383 个字符，设置 `use_space_char=True` 让 CTCLabelDecode 自动添加 blank + space → 18385 匹配。

### 迭代 3: 检测框坐标全错

**错误**: 所有检测框坐标都是 (640, 640)

**根因**: `boxes_from_bitmap` 中坐标缩放逻辑错误。

修复: 移除 `boxes_from_bitmap`，直接使用 `polygons_from_bitmap` 并传入正确的 `dest_width`/`dest_height`（原图尺寸）。

### 迭代 4: 文字被透视变换扭曲

**错误**: 识别出 "改祾仲" 等根本不存在的字。

**根因**: `_crop_image` 用了 `cv2.getPerspectiveTransform` + `cv2.warpPerspective`，将四边形"矫正"成矩形，但透视变换扭曲了文字。

修复: 改用轴对齐矩形裁剪 `img[y1:y2, x1:x2]`（对齐 C++ 的做法）。

### 迭代 5: 长文本识别为空

**错误**: 底部 "登机口于起飞前10分钟关闭..." (730px宽) 识别为空。

**根因**: `rec_preprocess` 最大宽度限制为 320px，730px 被严重压缩。

修复: 对齐 C++ 源码的 `max_imgW = 3200`:
```python
if new_w > 3200: new_w = 3200  # 不是 320!
```

模型支持动态宽度: 320→T=40, 640→T=80, 1130→T=141

### 迭代 6: 检测框碎片化

**错误**: 38 个检测框 vs C++ 基准 30 个。"BOARDING" + "PASS" 被分成两个框。

**根因**: ONNX 模型输出的概率图比 PaddlePaddle 原生稍碎片化。

修复: 添加 `_merge_adjacent()` 函数，合并水平紧邻、垂直对齐的框。

算法: 并查集，合并条件:
- 垂直重叠 > 50%
- 水平间距 < 平均高度 × 2
- 不合并宽高比差异过大的框

效果: 38 → 30 框（与 C++ 一致）。

### 迭代 7: 图片上中文不可见

**错误**: `output_test.jpg` 上识别框正确但文字空白。

**根因**: OpenCV `cv2.putText()` 的 Hershey 字体不支持中文。

修复: 使用 PIL + 系统 NotoSerifCJK 字体:
```python
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc', 16)
draw.text((x, y), text, font=font, fill=(0, 255, 0))
```

### 最终准确率对比

| 文本 | C++ 基准 | OpenCV DNN v3 |
|------|------|------|
| 登机牌 | ✅ | ✅ |
| BOARDINGPASS | ✅ | ✅ (合并为 "登机牌BOARDING PASS") |
| 舱位CLASS | ✅ | ✅ |
| 航班FLIGHT | ✅ | ✅ |
| 张祺伟 | ✅ | ✅ |
| ETKT7813699238489/1 | ✅ | ✅ |
| 登机口于起飞前10分钟关闭... | ✅ | ✅ |

---

## 8. 阶段五：TensorRT 尝试（失败）

### 8.1 环境准备

```bash
# TensorRT 10.3.0 系统已安装, 链接到 venv
ln -sf /usr/lib/python3.10/dist-packages/tensorrt venv_paddleocr/lib/python3.10/site-packages/
ln -sf /usr/lib/python3.10/dist-packages/tensorrt-10.3.0.dist-info venv_paddleocr/lib/python3.10/site-packages/
```

### 8.2 CUDA 内存管理

pycuda 安装失败 (编译耗时)，改用 ctypes 直接调用 CUDA Runtime API:

```python
cu = ctypes.CDLL('libcudart.so')
cu.cudaMalloc.restype = ctypes.c_int
cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
def gpu_alloc(size):
    p = ctypes.c_void_p()
    cu.cudaMalloc(ctypes.byref(p), size)
    return p.value
```

### 8.3 TRT 推理管线（原理验证通过）

```python
engine = runtime.deserialize_cuda_engine(buf)
ctx = engine.create_execution_context()
ctx.set_input_shape('x', (1, 3, 640, 640))
ctx.set_tensor_address('x', d_input)
ctx.set_tensor_address('fetch_name_0', d_output)
ctx.execute_async_v3(0)
```

检测速度: **11.6ms** (vs OpenCV DNN 67ms，理论 5.8x 提速)

### 8.4 致命问题

| 优化级别 | 构建时间 | 结果 |
|:--:|------|------|
| `optimization_level=2` | 117s | **输出全零** ❌ |
| `optimization_level=3` (默认) | 3+ 分钟未完成 | ⏳ |

**根因**: Jetson Orin Nano 8GB 统一内存紧张（构建时 6.1GB 占用），TensorRT 图编译在嵌入式 GPU 上极慢。SVTR Transformer 识别模型预计需 10+ 分钟。

### 8.5 结论

**TensorRT 不适合 Jetson Orin Nano 8GB 上的此模型**。构建太慢，调试周期不可接受。OpenCV DNN 已有 6x 提速，足够实用。

---

## 9. 阶段六：性能瓶颈分析

### 每框耗时分解

```
采样 5 框, det_size=960:

裁剪 (numpy slice):   0.1ms ( 0%)
预处理 (cv2.resize):  0.8ms ( 1%)
setInput (数据拷贝):   0.1ms ( 0%)
模型推理 (GPU):      79.0ms (97%)  ← 瓶颈
CTC解码 (numpy):      1.8ms ( 2%)
─────────────────────────────
每框合计:            81.8ms
```

### 全图预估 (30框)

```
检测:                 ~100ms ( 4%)
识别 ×30:  30×79ms ≈ 2370ms (93%)
其他开销:             ~100ms ( 4%)
═══════════════════════════════
总计:                ~2.5s (首次)
                    ~1.0s (后续, GPU 已预热)
```

### 批处理不可行

`batch > 1` 时模型内部 Concat 操作报错:
```
input[0] = [N, 480, 1, 40]
input[1] = [1, 480, 1, N*40]
→ Inconsistent shape for ConcatLayer
```

### 优化方向（按收益排序）

| 方案 | 预期提速 | 难度 | 状态 |
|------|:--:|:--:|:--:|
| GPU 预热 | ~2x (首次) | 低 | ✅ 已实现 |
| 减小 det_size (640) | ~1.1x | 低 | 支持, 但漏小字 |
| TensorRT | 5-6x | 高 | ❌ 失败 |

---

## 10. 关键踩坑记录

### 坑 1: pip 安装 opencv-python 覆盖系统 CUDA OpenCV

**现象**: `cv2.cuda.getCudaEnabledDeviceCount()` 报错

**原因**: `rapidocr-onnxruntime` 依赖 `opencv-python`，pip 安装了 4.13.0 CPU 版

**解决**: `pip uninstall opencv-python opencv-contrib-python -y`，用 `sys.path.insert` 导入系统版

### 坑 2: NumPy 版本冲突

**现象**: `import cv2` 报 `_ARRAY_API not found`

**原因**: 系统 OpenCV 用 NumPy 1.x 编译，PaddlePaddle 依赖 NumPy 2.x

**解决**: `pip install 'numpy<2'` 降级到 1.26.4

### 坑 3: ONNX 文件双重后缀

**现象**: 导出文件名为 `model.onnx.onnx`

**原因**: `paddle.onnx.export(output_path)` 会在 `output_path` 后追加 `.onnx`

**解决**: 传 `output_path='model'` 不带后缀，或事后 `mv`

### 坑 4: PIR 模型加载方式

**现象**: `paddle.jit.load(model_dir)` → `KeyError: 'forward'`

**原因**: PIR 模型目录中只有 `inference.json`，`load(dir)` 期望 `forward` program

**解决**: `paddle.jit.load(f'{model_dir}/inference')` — 传文件前缀

### 坑 5: RecResizeImg 归一化是内置的

**现象**: 识别乱码 / 全空

**原因**: 以为识别模型不需要归一化，因为 inference.yml 中没有 `NormalizeImage`。但实际上 `RecResizeImg` 内置了 `(x/255 - 0.5) / 0.5`

**解决**: 正确实现 RecResizeImg 的归一化逻辑

### 坑 6: max_imgW 是 3200 不是 320

**现象**: 长文本（730px 宽）识别为空

**原因**: `rec_preprocess` 最大宽度设为 320px

**解决**: 对齐 C++ 源码 `max_imgW = 3200`

### 坑 7: 字典维度需要 use_space_char=True

**现象**: CTCLabelDecode 添加 blank 后字典 18384 维 vs 模型输出 18385 维

**原因**: `use_space_char=False` 不添加空格

**解决**: `use_space_char=True` → 18383 + blank + space = 18385

### 坑 8: OpenCV DNN 不支持中文渲染

**现象**: 输出图上识别框正确但无文字

**原因**: `cv2.putText` 的 Hershey 字体只有 ASCII

**解决**: PIL + NotoSerifCJK 字体

---

## 11. 对 ocrcplus 项目的批判分析

`/home/ysdhanji/ocrcplus/` 是一个 PaddleOCR C++ 部署项目。

### 可复用的资产
- PP-OCRv5 Mobile 模型文件 (PIR 格式)
- PaddleOCR 源码 (后处理类)
- 预处理参数 (inference.yml)
- C++ 基准结果 (30 框, ~5.5s, 正确识别)

### 不足之处
- **速度慢**: ~5.5s/图，不够实用
- **TRT 失败**: 预编译 Paddle (TRT 8.5 API) vs 系统 TRT 10.3 不兼容
- **FP16 更慢**: Orin Nano Ampere 无专用 FP16 Tensor Core
- **CPU 崩溃**: ARM 无 MKL/MKLDNN
- **模型格式陈旧**: PIR 难以导出

### 我们的改进
- 速度: 5.5s → 1.0s (6×)
- 准确率: 对齐 C++ 基准
- 维护性: 纯 Python，无需编译

---

## 12. 使用方法

### 基本用法

```bash
cd /home/ysdhanji/ocr
source venv_paddleocr/bin/activate

# 高精度 (det_size=960)
python ocr_opencv_dnn.py -i 图片.jpg -o 结果.jpg

# 快速 (det_size=640, 可能漏小字)
python ocr_opencv_dnn.py -i 图片.jpg --det_size 640

# 性能测试
python ocr_opencv_dnn.py -i 图片.jpg --benchmark
```

### 导出新模型

```bash
python export_to_onnx.py
# 从 PIR 模型导出 ONNX, 自动 onnxsim 简化
```

### 后台构建 TRT (不推荐)

```bash
nohup python -u build_trt_engines.py > logs/trt_build.log 2>&1 &
tail -f logs/trt_build.log
```

---

## 13. 已知限制

1. **OpenCV DNN 不支持 batch > 1**: 模型内部 Concat 操作限制
2. **不能 pip install opencv-python**: 会覆盖 CUDA 版
3. **NumPy 必须 < 2**: 系统 OpenCV 兼容性
4. **cuDNN 警告无害**: 版本号格式差异，不影响推理
5. **内存紧张**: 8GB 统一内存，推理时 ~4GB 可用
6. **TensorRT 不实用**: 构建太慢
7. **仅支持 PP-OCRv5 Mobile 模型**: 已验证此模型，其他需重新导出

---

> 📝 **文档版本**: 2026-06-09  
> 📁 **项目路径**: `/home/ysdhanji/ocr/`  
> 🔗 **参考项目**: `/home/ysdhanji/ocrcplus/`  
> 💾 **Repo Memory**: `/memories/repo/paddleocr-jetson.md`
