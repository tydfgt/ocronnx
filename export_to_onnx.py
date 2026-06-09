#!/usr/bin/env python3
"""
PaddleOCR 模型导出为 ONNX 格式
支持 PP-OCRv5 检测和识别模型 → ONNX → OpenCV DNN 推理
"""

import os
import sys
import argparse
import paddle
import numpy as np

# 添加虚拟环境路径
MODEL_CACHE = os.path.expanduser("~/.paddlex/official_models")

def export_det_model(model_dir: str, output_path: str, input_shape=(1, 3, 640, 640)):
    """导出文字检测模型 (DBNet) 为 ONNX"""
    print(f"[检测模型] 从 {model_dir} 加载...")

    # PaddleOCR 3.x 使用 PIR 格式，通过 PaddleX 加载
    from paddlex.modules.text_detection.model import PPDet
    import json

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 创建模型并加载权重
    model = PPDet(config=config)
    model.load_state_dict(paddle.load(os.path.join(model_dir, "inference.pdiparams")))
    model.eval()

    # 创建虚拟输入
    dummy_input = paddle.randn(input_shape)

    # 导出 ONNX
    print(f"[检测模型] 导出 ONNX 到 {output_path}...")
    paddle.onnx.export(
        model,
        output_path,
        input_spec=[paddle.static.InputSpec(shape=input_shape, dtype='float32')],
        opset_version=14,
    )
    print(f"[检测模型] 导出完成: {output_path}")


def export_rec_model(model_dir: str, output_path: str, input_shape=(1, 3, 48, 320)):
    """导出文字识别模型 (SVTR_HGNet) 为 ONNX"""
    print(f"[识别模型] 从 {model_dir} 加载...")

    from paddlex.modules.text_recognition.model import PPRec
    import json

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = PPRec(config=config)
    model.load_state_dict(paddle.load(os.path.join(model_dir, "inference.pdiparams")))
    model.eval()

    dummy_input = paddle.randn(input_shape)

    print(f"[识别模型] 导出 ONNX 到 {output_path}...")
    paddle.onnx.export(
        model,
        output_path,
        input_spec=[paddle.static.InputSpec(shape=input_shape, dtype='float32')],
        opset_version=14,
    )
    print(f"[识别模型] 导出完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR → ONNX 模型导出")
    parser.add_argument("--det_model", type=str,
                        default=f"{MODEL_CACHE}/PP-OCRv5_server_det",
                        help="检测模型目录")
    parser.add_argument("--rec_model", type=str,
                        default=f"{MODEL_CACHE}/PP-OCRv5_server_rec",
                        help="识别模型目录")
    parser.add_argument("--det_output", type=str,
                        default="models/det.onnx",
                        help="检测 ONNX 输出路径")
    parser.add_argument("--rec_output", type=str,
                        default="models/rec.onnx",
                        help="识别 ONNX 输出路径")
    parser.add_argument("--skip_det", action="store_true", help="跳过检测模型")
    parser.add_argument("--skip_rec", action="store_true", help="跳过识别模型")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    if not args.skip_det:
        export_det_model(args.det_model, args.det_output)
    if not args.skip_rec:
        export_rec_model(args.rec_model, args.rec_output)

    print("\n✅ 所有模型导出完成!")
    print(f"   检测模型: {args.det_output}")
    print(f"   识别模型: {args.rec_output}")


if __name__ == "__main__":
    main()
