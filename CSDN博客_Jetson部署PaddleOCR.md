# Jetson Orin Nano 部署 PaddleOCR 终极方案：ONNX + OpenCV DNN，6倍提速！

> **设备**: NVIDIA Jetson Orin Nano Super (8GB)  
> **效果**: 比 C++ Paddle Inference 快 **6 倍**，单图 ~1 秒  
> **核心**: PP-OCRv5 Mobile → ONNX → OpenCV DNN CUDA  
> **GitHub**: [项目地址](https://github.com/your/repo)

---

## 一、前言

在 Jetson Orin Nano 上部署 PaddleOCR 做文字识别，网上能找到的教程基本都指向 C++ Paddle Inference 方案。但实际跑下来，速度只有 **5.5 秒/图**，完全没法用。

PaddlePaddle 官方至今没有发布 JetPack 6 (L4T 36.x) 的 GPU 预编译包。没有 GPU 加速，OCR 在嵌入设备上就是慢动作。

经过一整天的折腾，我找到了一条可行的路：**将 PP-OCRv5 模型导出为 ONNX，用 OpenCV DNN + CUDA 做推理**。最终速度提升了 **6 倍**，从 5.5s 降到 1s 以内。

这篇文章记录了完整过程，包括 8 个大坑和解决方案。

---

## 二、环境说明

| 项目 | 参数 |
|------|------|
| 设备 | NVIDIA Jetson Orin Nano Super (P3767-0005) |
| CPU | 6核 ARM Cortex-A78AE |
| GPU | Ampere GA10B, 1024 CUDA cores, SM 8.7 |
| 内存 | 8GB LPDDR5 (CPU/GPU 统一) |
| 系统 | Ubuntu 22.04, L4T R36.5.0, JetPack 6.0 |
| CUDA | 12.6.68 |
| cuDNN | 9.3.0 |
| OpenCV | 4.10.0 (自编译, CUDA 加速) |
| Python | 3.10.12 |

---

## 三、为什么不用 PaddlePaddle GPU？

一句话：**PaddlePaddle 没有 JetPack 6 的 GPU 预编译包**。

| 平台 | PaddlePaddle GPU 预编译 |
|------|:--:|
| x86_64 Linux CUDA 12 | ✅ |
| Jetson JetPack 5.x (CUDA 11.4) | ✅ |
| **Jetson JetPack 6.x (CUDA 12.6)** | ❌ |

有人用 JetPack 5.x 的预编译包在 JetPack 6 上跑（利用 CUDA 向后兼容），但这种方式：
- TensorRT 不兼容（预编译 TRT 8.5 vs 系统 TRT 10.3）
- FP16 反而更慢（Orin Nano 无专用 FP16 Tensor Core）
- CPU 模式 Segfault（ARM 没有 Intel MKL）
- 速度 5.5s/图，太慢

从源码编译 PaddlePaddle 理论上可行，但耗时 4+ 小时且需要 8GB+ 内存，失败率高。

---

## 四、核心思路

```
PP-OCRv5 Mobile 模型 (PIR格式)
    │
    ▼
paddle.onnx.export()  ← 导出 ONNX
    │
    ▼
onnxsim simplify()    ← 简化模型
    │
    ▼
OpenCV DNN + CUDA     ← GPU 推理
    │
    ▼
PaddleOCR 原生后处理   ← 复用 DTPostProcess + CTCLabelDecode
    │
    ▼
输出: JSON 文本 + 可视化图片
```

### 为什么选 OpenCV DNN？

- Jetson 上 OpenCV 已编译 CUDA 加速版（含 DNN CUDA 后端）
- 无需额外依赖，无需 PaddlePaddle 运行时
- 推理速度实测 67ms/检测 + 25ms/识别框

### 为什么选 onnxsim？

OpenCV DNN 的 ONNX 解析器比较挑剔，部分算子不支持。原始 ONNX 加载直接报 `Reshape` 错误，`onnxsim` 简化后可以正常加载。

---

## 五、Step by Step 实战

### Step 1: 创建虚拟环境

```bash
cd /home/ysdhanji/ocr
python3 -m venv venv_paddleocr
source venv_paddleocr/bin/activate
pip install --upgrade pip
```

### Step 2: 安装依赖

```bash
# PaddlePaddle CPU 版 (GPU 不可用)
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html

# PaddleOCR
pip install paddleocr

# paddle2onnx (--no-deps 跳过 onnxoptimizer 编译)
pip install paddle2onnx --no-deps

# ONNX 工具链
pip install onnx onnxsim onnxruntime
```

### Step 3: 准备 PIR 模型

PP-OCRv5 模型可以从 PaddleOCR 官方下载：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar
tar xf PP-OCRv5_mobile_det_infer.tar
tar xf PP-OCRv5_mobile_rec_infer.tar
```

模型是 PIR 格式（PaddlePaddle 3.x 新格式），包含 `inference.json` + `inference.pdiparams`。

### Step 4: PIR → ONNX 导出

```python
import paddle
import os

os.makedirs('models', exist_ok=True)

# === 导出检测模型 ===
det_model = paddle.jit.load('PP-OCRv5_mobile_det_infer/inference')  # 注意: 文件前缀, 不是目录!
det_model.eval()

paddle.onnx.export(
    det_model,
    'models/PP-OCRv5_mobile_det',  # 不带.onnx后缀!
    input_spec=[paddle.static.InputSpec(shape=[1, 3, -1, -1], dtype='float32', name='x')],
    opset_version=14,
)

# === 导出识别模型 ===
rec_model = paddle.jit.load('PP-OCRv5_mobile_rec_infer/inference')
rec_model.eval()

paddle.onnx.export(
    rec_model,
    'models/PP-OCRv5_mobile_rec',
    input_spec=[paddle.static.InputSpec(shape=[1, 3, -1, -1], dtype='float32', name='x')],
    opset_version=14,
)
```

> ⚠️ **坑 1**: `paddle.jit.load()` 参数是**文件前缀** `inference`，不是目录！传目录会报 `KeyError: 'forward'`。  
> ⚠️ **坑 2**: `export()` 的第二个参数不要带 `.onnx` 后缀，否则文件会变成 `.onnx.onnx`。

### Step 5: ONNX 模型简化

```python
import onnx
from onnxsim import simplify

for name in ['PP-OCRv5_mobile_det', 'PP-OCRv5_mobile_rec']:
    model = onnx.load(f'models/{name}.onnx')
    model_simp, check = simplify(model)
    onnx.save(model_simp, f'models/{name}_sim.onnx')
    print(f'{name}: simplified, check={check}')
```

简化后 OpenCV DNN 才能加载。

### Step 6: 准备字符字典

从识别模型的 `inference.yml` 提取：

```python
import yaml

with open('PP-OCRv5_mobile_rec_infer/inference.yml') as f:
    config = yaml.safe_load(f)

# 只保存 18383 个字符, blank 和 space 由 CTCLabelDecode 自动添加
chars = config['PostProcess']['character_dict']
with open('models/ppocr_keys_v1.txt', 'w', encoding='utf-8') as f:
    for c in chars:
        f.write(c + '\n')
```

> ⚠️ **坑 3**: 字典文件只需 18383 字符。`CTCLabelDecode(use_space_char=True)` 会自动添加 blank + space → 18385 维，匹配模型输出。

### Step 7: 编写推理脚本

完整的推理脚本包含以下关键部分：

#### 7.1 系统 OpenCV 导入

pip 版 `opencv-python` 没有 CUDA！必须用系统自编译版：

```python
import sys
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
import cv2
```

> ⚠️ **坑 4**: 如果之前 pip install 过 opencv-python，会导致 NumPy 版本冲突。需要卸载 pip 版，降级 NumPy 到 1.x。

#### 7.2 检测预处理

```python
def det_preprocess(img, target_size=960):
    """PP-OCRv5 检测预处理: resize_long=960, NormalizeImage, padding到32倍数"""
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)

    # Padding 到 32 的倍数 (DBNet 下采样 1/32)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114,114,114))

    # 归一化: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC → CHW → BCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, (new_h, new_w), (h, w), (pad_h, pad_w)
```

#### 7.3 识别预处理

```python
def rec_preprocess(img, target_shape=(48, 320)):
    """PP-OCRv5 识别预处理: RecResizeImg 内置归一化"""
    h, w = img.shape[:2]
    target_h, target_w = target_shape

    # 保持宽高比, 高度缩放到 48
    ratio = target_h / h
    new_w = int(w * ratio)
    if new_w > 3200:          # ⚠️ 不是 320! C++ 默认 max_imgW=3200
        new_w = 3200
    img = cv2.resize(img, (new_w, target_h))

    # CHW + RecResizeImg 内置归一化: (x/255 - 0.5) / 0.5 → [-1, 1]
    img = img.astype(np.float32).transpose(2, 0, 1)
    img = img / 255.0
    img = (img - 0.5) / 0.5

    # 宽度填充 (仅当 < target_w)
    if new_w < target_w:
        pad = np.zeros((3, target_h, target_w - new_w), dtype=np.float32)
        img = np.concatenate([img, pad], axis=2)

    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img
```

> ⚠️ **坑 5**: 识别模型 `max_imgW` 默认 **3200**，不是 320！设置为 320 会导致长文本（如登机牌底部小字）被严重压缩，识别为空。  
> ⚠️ **坑 6**: 识别预处理中没有独立的 `NormalizeImage`，但 `RecResizeImg` **内置**了 `(x/255 - 0.5) / 0.5` 的归一化操作。不要漏掉，也不要多做。

#### 7.4 模型加载与 CUDA 后端

```python
det_net = cv2.dnn.readNetFromONNX('models/PP-OCRv5_mobile_det_sim.onnx')
rec_net = cv2.dnn.readNetFromONNX('models/PP-OCRv5_mobile_rec_sim.onnx')

# 设置 CUDA 后端
det_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
det_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
rec_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
rec_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

#### 7.5 后处理（复用 PaddleOCR 原生类）

不用自己写！直接引用 PaddleOCR 源码中的类：

```python
import importlib.util

# 绕过 __init__.py 避免 skimage 等可选依赖
spec = importlib.util.spec_from_file_location(
    'db_pp', 'PaddleOCR-main/ppocr/postprocess/db_postprocess.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DBPostProcess = mod.DBPostProcess

# 同样加载 CTCLabelDecode
spec2 = importlib.util.spec_from_file_location(
    'rec_pp', 'PaddleOCR-main/ppocr/postprocess/rec_postprocess.py')
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
CTCLabelDecode = mod2.CTCLabelDecode
```

#### 7.6 中文可视化

> ⚠️ **坑 7**: `cv2.putText()` 的 Hershey 字体**不支持中文**，中文字会空白！

```python
from PIL import Image, ImageDraw, ImageFont

# 系统自带 Noto Serif CJK 中文字体
font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc', 16)

# OpenCV BGR → PIL RGB → 绘制中文 → 转回 OpenCV BGR
vis_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vis_pil = Image.fromarray(vis_rgb)
draw = ImageDraw.Draw(vis_pil)
draw.text((x, y), text, font=font, fill=(0, 255, 0))
vis = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
```

### Step 8: 解决 ONNX 检测框碎片化

> ⚠️ **坑 8**: ONNX 输出的概率图比 PaddlePaddle 原生稍碎片化，导致多出 ~8 个检测框。

添加相邻框合并算法：

```python
def _merge_adjacent(boxes, scores):
    """并查集合并水平紧邻框"""
    n = len(boxes)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb: parent[pb] = pa
    
    for i in range(n):
        bi = np.array(boxes[i])
        yi_min, yi_max = bi[:,1].min(), bi[:,1].max()
        xi_max = bi[:,0].max()
        hi = yi_max - yi_min
        
        for j in range(i+1, n):
            bj = np.array(boxes[j])
            yj_min, yj_max = bj[:,1].min(), bj[:,1].max()
            xj_min = bj[:,0].min()
            hj = yj_max - yj_min
            
            # 垂直重叠 > 50%
            y_overlap = min(yi_max, yj_max) - max(yi_min, yj_min)
            if y_overlap < min(hi, hj) * 0.5: continue
            
            # 水平间距 < 平均高度 × 2
            x_gap = xj_min - xi_max
            avg_h = (hi + hj) / 2
            if 0 < x_gap < avg_h * 2:
                union(i, j)
    
    # ... 按组合并 ...
```

效果：38 框 → 30 框（对齐 C++ 基准）。

---

## 六、性能测试

### 测试环境

- 图片: 896×528, 30 个文本区域
- 预热 2 次后取 20 次平均

### 结果

| 方案 | 检测 | 识别 | **总耗时** | 提速 |
|------|------|------|----------|:--:|
| C++ Paddle Inference | ~2s | ~3s | **~5.5s** | 基准 |
| **OpenCV DNN CUDA (本项目)** | 100ms | 900ms | **~1.0s** | **6×** |

### 每框耗时分解

```
裁剪 (numpy):    0.1ms ( 0%)
预处理 (resize):  0.8ms ( 1%)
setInput (拷贝):  0.1ms ( 0%)
模型推理 (GPU):  79.0ms (97%)  ← 瓶颈
CTC解码 (numpy):  1.8ms ( 2%)
─────────────────────────
每框合计:        81.8ms
```

97% 的时间花在 GPU 推理上，CPU 处理几乎不占时间。

### 为什么不用 TensorRT？

试过了，在 Jetson Orin Nano 8GB 上：
- `optimization_level=2`: 构建 2 分钟，但输出全零（计算错误）
- `optimization_level=3`: 构建 3+ 分钟未完成
- 识别模型（SVTR Transformer）预计 10+ 分钟

**结论**: TensorRT 在此设备上不实用。编译太慢，调试周期不可接受。

---

## 七、效果展示

### 识别结果（登机牌测试图）

```
[0.996] 登机牌
[0.981] BOARDING PASS
[0.997] 票价FARE
[0.997] 张祺伟
[0.989] ZHANGQIWEI
[0.989] 姓名NAME
[0.999] 福州
[0.995] FUZHOU
[0.996] 航班FLIGHT
[0.998] 登机口
[0.999] 日期DATE
[1.000] 座位号
[0.959] 登机口于起飞前10分钟关闭 GATESCLOSE10MINUTESBEFORE DEPARTURE TIME
```

### 性能

```bash
$ python ocr_opencv_dnn.py -i test.png --benchmark

性能测试 (896x528, 20次):
  检测(det_size=960): 100ms (±8ms)
  识别(单框): 25.5ms
  识别(30框): 765ms
  总耗时: ~900ms
```

---

## 八、完整代码

完整推理脚本（200+ 行）可在项目中找到：

```bash
git clone <repo_url>
cd ocr
source venv_paddleocr/bin/activate
python ocr_opencv_dnn.py -i your_image.jpg
```

关键文件：
- `ocr_opencv_dnn.py` — 主推理脚本
- `export_to_onnx.py` — ONNX 导出脚本
- `models/` — ONNX 模型 + 字典
- `PP-OCRv5_部署完整记录.md` — 完整技术文档

---

## 九、踩坑速查表

| # | 现象 | 原因 | 解决 |
|:--:|------|------|------|
| 1 | `paddle.jit.load(dir)` 报 `KeyError: 'forward'` | PIR 格式需文件前缀 | `load(f'{dir}/inference')` |
| 2 | ONNX 文件名变 `xxx.onnx.onnx` | `export` 自动加后缀 | 传入不带后缀的路径 |
| 3 | `cv2.dnn.readNetFromONNX` 报 `Reshape` 错误 | OpenCV DNN 不支持某些算子 | onnxsim 简化 |
| 4 | 系统 `import cv2` 报 `_ARRAY_API` | NumPy 2.x vs 1.x 冲突 | `pip install 'numpy<2'` |
| 5 | pip opencv-python 覆盖 CUDA 版 | pip 依赖自动安装 | `pip uninstall opencv-python -y` |
| 6 | 识别全部乱码 | 字典维度不匹配 | `CTCLabelDecode(use_space_char=True)` |
| 7 | 图片上中文空白 | cv2.putText 不支持中文 | PIL + 中文字体 |
| 8 | 长文本识别为空 | max_imgW 设为 320 | 改为 3200 |
| 9 | 检测框比基准多 8 个 | ONNX 概率图碎片化 | 相邻框合并算法 |
| 10 | TensorRT 引擎输出全零 | `optimization_level=2` bug | 放弃 TRT, 用 OpenCV DNN |

---

## 十、总结

在 Jetson Orin Nano (JetPack 6) 上部署 PaddleOCR，**ONNX + OpenCV DNN** 是目前最实用的方案：

✅ **速度快**: 比 C++ Paddle Inference 快 6 倍  
✅ **精度好**: 对齐原生 PaddlePaddle 输出  
✅ **依赖少**: 不需要 PaddlePaddle 运行时  
✅ **纯 Python**: 不需要编译 C++ 代码  
✅ **可维护**: 代码清晰, 后处理复用 PaddleOCR 原生类  

❌ TensorRT 在此设备上不实用（编译太慢）  
❌ 批处理不支持（模型架构限制）  

完整文档和代码已开源，欢迎 Star ⭐ 

---

> **作者**: [你的名字]  
> **日期**: 2026-06-09  
> **设备**: NVIDIA Jetson Orin Nano Super  
> **项目地址**: [GitHub]
