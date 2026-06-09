# PP-OCRv5 ONNX + OpenCV DNN 推理引擎

[![Platform](https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-green)](https://github.com/tydfgt/ocronnx)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-brightgreen)](https://github.com/tydfgt/ocronnx)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0%20CUDA-blue)](https://github.com/tydfgt/ocronnx)
[![Speed](https://img.shields.io/badge/speed-6x%20faster-orange)](https://github.com/tydfgt/ocronnx)

> 🚀 在 NVIDIA Jetson Orin Nano 上用 ONNX + OpenCV DNN 实现 **6 倍于 C++ Paddle Inference** 的 OCR 推理速度。

**GitHub**: https://github.com/tydfgt/ocronnx

---

## 特性

- ⚡ **6x 提速**: ~1.0s/图 vs C++ Paddle Inference ~5.5s
- 🎯 **精度对齐**: 复用 PaddleOCR 原生后处理，30 个检测框与 C++ 版完全一致
- 📦 **轻量依赖**: 无需 PaddlePaddle GPU 运行时，仅需 ONNX + OpenCV
- 🐍 **纯 Python**: 无需编译 C++，pip install 即可
- 🇨🇳 **中文友好**: PIL + NotoSerifCJK 字体渲染中文标注

---

## 环境要求

| 项目 | 最低要求 | 本项目实测 |
|------|------|------|
| 设备 | Jetson Orin Nano / AGX | Orin Nano Super (8GB) |
| JetPack / L4T | 6.0 / R36.x | R36.5.0 |
| CUDA | 12.x | 12.6.68 |
| OpenCV | 4.5+ (CUDA) | 4.10.0 自编译 |
| Python | 3.10+ | 3.10.12 |
| 内存 | 8GB+ | 8GB (建议 11GB swap) |

> ⚠️ x86_64 Linux 也可以用（用 pip opencv-python + CUDA 或 OpenCV DNN CPU），但性能未测试。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/tydfgt/ocronnx.git
cd ocronnx
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 运行推理

```bash
python ocr_opencv_dnn.py -i test.jpg -o result.jpg
```

### 4. 性能测试

```bash
python ocr_opencv_dnn.py -i test.jpg --benchmark
```

---

## 模型准备

项目已包含预导出的 ONNX 模型 (`models/` 目录)。如需重新导出：

```bash
# 1. 下载 PP-OCRv5 Mobile PIR 模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar
tar xf PP-OCRv5_mobile_det_infer.tar
tar xf PP-OCRv5_mobile_rec_infer.tar

# 2. 导出 ONNX
python export_to_onnx.py

# 3. 自动 onnxsim 简化
# export_to_onnx.py 会自动调用 onnxsim
```

---

## 命令行参数

```
python ocr_opencv_dnn.py --help

  -i, --image    输入图片路径 (必填)
  -o, --output   输出图片路径 (默认: output.jpg)
  --det_size     检测 resize 边长 (默认: 960, 640 更快但可能漏小字)
  --benchmark    性能测试模式
  --cpu          使用 CPU 推理 (默认 CUDA)
```

---

## 项目结构

```
ocronnx/
├── ocr_opencv_dnn.py          # 主推理脚本
├── export_to_onnx.py          # PIR → ONNX 导出
├── requirements.txt           # Python 依赖
├── README.md                  # 本文件
├── models/
│   ├── PP-OCRv5_mobile_det_sim.onnx   # 检测 ONNX (4.6MB)
│   ├── PP-OCRv5_mobile_rec_sim.onnx   # 识别 ONNX (16MB)
│   └── ppocr_keys_v1.txt             # 字符字典
├── PaddleOCR-main/            # PaddleOCR 源码 (后处理复用)
├── PP-OCRv5_部署完整记录.md   # 详细技术文档
└── CSDN博客_Jetson部署PaddleOCR.md  # CSDN 博客原文
```

---

## 性能

| 方案 | 检测 | 识别 (30框) | **总耗时** | 提速 |
|------|------|------------|----------|:--:|
| C++ Paddle Inference | ~2s | ~3s | **~5.5s** | 基准 |
| **本项目** | 100ms | 900ms | **~1.0s** | **6×** |

每框耗时分解:
```
GPU 推理: 79ms (97%) ← 瓶颈
预处理:    1ms ( 2%)
CTC解码:   2ms ( 1%)
```

---

## TensorRT 支持

TensorRT 理论上可再提速 5-6 倍，但在 Jetson Orin Nano 8GB 上：
- 引擎编译极慢 (检测 ~2min, 识别 ~10min)
- `optimization_level=2` 产出错误输出
- 8GB 内存不足以高效编译

**不推荐在 8GB 设备上使用 TensorRT**。AGX (32GB+) 可尝试。

---

## 常见问题

**Q: `import cv2` 报 NumPy 错误？**  
A: 系统 OpenCV 用 NumPy 1.x 编译，降级 NumPy: `pip install 'numpy<2'`

**Q: 图片上中文不显示？**  
A: `cv2.putText` 不支持中文。已用 PIL + NotoSerifCJK 字体修复。

**Q: 长文本识别为空？**  
A: 确保 `rec_preprocess` 中 `max_width=3200`（不是 320）。

**Q: 检测框比 C++ 多？**  
A: ONNX 概率图稍碎片化，已内置相邻框合并算法。

**Q: 能批量推理吗？**  
A: 当前模型架构限制 batch > 1。已测试，OpenCV DNN 会报 Concat 维度错误。

---

## 引用

如果本项目对你有帮助，请 Star ⭐ 或引用：

```bibtex
@misc{ocronnx2026,
  title={PP-OCRv5 ONNX + OpenCV DNN for Jetson Orin Nano},
  author={Qu Xuesong},
  year={2026},
  url={https://github.com/tydfgt/ocronnx}
}
```

---

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — OCR 模型与后处理
- [OpenCV](https://opencv.org/) — DNN 推理引擎
- [onnxsim](https://github.com/daquexian/onnx-simplifier) — ONNX 模型简化

---

## License

Apache 2.0
