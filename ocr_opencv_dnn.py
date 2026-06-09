#!/usr/bin/env python3
"""
OpenCV DNN + CUDA OCR 推理引擎 (使用 PaddleOCR 原生后处理)
模型: PP-OCRv5 Mobile (DBNet 检测 + SVTR_HGNet 识别)
后端: OpenCV 4.10.0 DNN + CUDA 12.6
后处理: PaddleOCR 原生 DBPostProcess + CTCLabelDecode

用法:
    python ocr_opencv_dnn.py --image test.jpg
    python ocr_opencv_dnn.py --image test.jpg --benchmark
"""

import sys
import os
import importlib.util
import argparse
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 确保使用系统 CUDA OpenCV
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
import cv2

# CJK 字体路径
_CJK_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'

# 加载 PaddleOCR 原生后处理模块 (绕过 __init__.py 避免依赖问题)
_PADDLEOCR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PaddleOCR-main')

def _load_module(filepath):
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(os.path.basename(filepath))[0], filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_db_pp = _load_module(os.path.join(_PADDLEOCR_DIR, 'ppocr/postprocess/db_postprocess.py'))
_rec_pp = _load_module(os.path.join(_PADDLEOCR_DIR, 'ppocr/postprocess/rec_postprocess.py'))
DBPostProcess = _db_pp.DBPostProcess
CTCLabelDecode = _rec_pp.CTCLabelDecode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 预处理 (对齐 PP-OCRv5 inference.yml 配置)
# ============================================================
def det_preprocess(img, target_size=960):
    """检测预处理: PP-OCRv5 配置: DetResizeForTest.resize_long=960, NormalizeImage, ToCHW
    需要 padding 到 32 倍数 (DBNet 下采样 1/32)
    """
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)

    # Padding 到 32 的倍数
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32

    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))

    padded_h, padded_w = new_h + pad_h, new_w + pad_w

    # NormalizeImage
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # ToCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img, (new_h, new_w), (h, w), (pad_h, pad_w)


def rec_preprocess(img, target_shape=(48, 320)):
    """识别预处理: RecResizeImg image_shape=[3,48,320] 内置归一化: x/255→[-1,1]
    max_width=3200 对齐 C++ max_imgW，支持长文本"""
    h, w = img.shape[:2]
    target_h, target_w = target_shape
    max_width = 3200

    # 保持宽高比，高度对齐 (C++ 默认 max_imgW=3200)
    ratio = target_h / h
    new_w = int(w * ratio)
    if new_w > max_width:
        new_w = max_width
    img = cv2.resize(img, (new_w, target_h))

    # 转 CHW + 归一化: (x/255 - 0.5) / 0.5 → [-1, 1]
    img = img.astype(np.float32).transpose(2, 0, 1)
    img = img / 255.0
    img = (img - 0.5) / 0.5

    # 宽度填充到 target_w (仅当 new_w < target_w)
    if new_w < target_w:
        pad = np.zeros((3, target_h, target_w - new_w), dtype=np.float32)
        img = np.concatenate([img, pad], axis=2)

    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


# ============================================================
# OCR 引擎
# ============================================================
class PPOCRv5OpenCVDNN:
    """PP-OCRv5 + OpenCV DNN CUDA 推理引擎 (v2: 批处理识别 + 预热)"""

    def __init__(self, det_onnx, rec_onnx, dict_path, use_cuda=True,
                 det_size=640):
        self.det_size = det_size

        # ONNX 模型
        self.det_net = cv2.dnn.readNetFromONNX(det_onnx)
        self._set_cuda(self.det_net, use_cuda)
        self.rec_net = cv2.dnn.readNetFromONNX(rec_onnx)
        self._set_cuda(self.rec_net, use_cuda)

        # PaddleOCR 原生后处理
        # PaddleOCR 原生后处理 (参数对齐 PP-OCRv5_mobile_det inference.yml)
        self.det_post = DBPostProcess(
            thresh=0.3, box_thresh=0.6, unclip_ratio=2.0,
            max_candidates=1000, box_type='quad'
        )
        self.rec_post = CTCLabelDecode(
            character_dict_path=dict_path, use_space_char=True
        )

        # 预热 (避免首次推理的 CUDA 初始化开销)
        if use_cuda:
            self._warmup()

        mode = f"CUDA det={det_size}"
        print(f"[PP-OCRv5] {mode}")

    def _warmup(self):
        """GPU 预热"""
        dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)
        self.det_net.setInput(dummy)
        self.det_net.forward()
        dummy2 = np.random.randn(1, 3, 48, 320).astype(np.float32)
        self.rec_net.setInput(dummy2)
        self.rec_net.forward()

    @staticmethod
    def _set_cuda(net, use):
        if use:
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                pass

    def detect(self, img):
        """文字检测 + 水平合并 (修复 ONNX 碎片化)"""
        h, w = img.shape[:2]
        input_tensor, (new_h, new_w), (orig_h, orig_w), (pad_h, pad_w) = \
            det_preprocess(img, target_size=self.det_size)

        self.det_net.setInput(input_tensor)
        pred = self.det_net.forward()[0, 0]
        pred = pred[:new_h, :new_w]

        bitmap = (pred > self.det_post.thresh).astype(np.uint8)
        boxes, scores = self.det_post.polygons_from_bitmap(
            pred, bitmap, orig_w, orig_h
        )

        # 合并水平相邻框 (ONNX 概率图比原生稍碎片化)
        boxes, scores = self._merge_adjacent(boxes, scores)
        return boxes, scores

    @staticmethod
    def _merge_adjacent(boxes, scores):
        """合并紧邻的水平框 (仅修复明显碎片, 比如 'BOARDING'+'PASS')"""
        if len(boxes) < 2:
            return boxes, scores

        merged = []
        merged_scores = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            bi = np.array(boxes[i])
            yi_min, yi_max = bi[:, 1].min(), bi[:, 1].max()
            xi_max = bi[:, 0].max()
            hi = yi_max - yi_min

            best_j = -1
            best_gap = float('inf')
            for j in range(len(boxes)):
                if i == j or used[j]:
                    continue
                bj = np.array(boxes[j])
                yj_min, yj_max = bj[:, 1].min(), bj[:, 1].max()
                xj_min = bj[:, 0].min()
                hj = yj_max - yj_min

                # 高度不能差太多
                if max(hi, hj) / max(min(hi, hj), 1) > 2.5:
                    continue

                # 垂直中心要对齐
                y_center_i = (yi_min + yi_max) / 2
                y_center_j = (yj_min + yj_max) / 2
                if abs(y_center_i - y_center_j) > min(hi, hj) * 0.4:
                    continue

                # 水平间距: 盒子 i 的右边到盒子 j 的左边
                gap = xj_min - xi_max
                if 1 < gap < max(hi, hj) * 1.5:  # 间距合理
                    if gap < best_gap:
                        best_gap = gap
                        best_j = j

            if best_j >= 0:
                # 合并 i 和 best_j
                bj = np.array(boxes[best_j])
                all_pts = np.concatenate([bi, bj])
                x_min = all_pts[:, 0].min(); y_min = all_pts[:, 1].min()
                x_max = all_pts[:, 0].max(); y_max = all_pts[:, 1].max()
                merged.append(np.array(
                    [[x_min, y_min], [x_max, y_min],
                     [x_max, y_max], [x_min, y_max]], dtype=np.float32))
                merged_scores.append((scores[i] + scores[best_j]) / 2)
                used[i] = True
                used[best_j] = True
            else:
                merged.append(boxes[i])
                merged_scores.append(scores[i])
                used[i] = True

        return merged, merged_scores

    def _crop_box(self, img, box):
        """四边形框 → 裁剪区域"""
        pts = np.array(box).reshape(-1, 2).astype(np.int32)
        x1, y1 = max(0, pts[:, 0].min()), max(0, pts[:, 1].min())
        x2, y2 = min(img.shape[1], pts[:, 0].max()), min(img.shape[0], pts[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def recognize_one(self, img, box):
        """识别单个文字区域"""
        crop = self._crop_box(img, box)
        if crop is None or crop.size == 0:
            return "", 0.0
        input_tensor = rec_preprocess(crop)
        self.rec_net.setInput(input_tensor)
        preds = self.rec_net.forward()
        results = self.rec_post(preds)
        return results[0] if results else ("", 0.0)

    def predict(self, img_path):
        """完整 OCR 流程"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取: {img_path}")
        print(f"[OCR] {img_path} ({img.shape[1]}x{img.shape[0]})")

        t0 = time.time()
        boxes, scores = self.detect(img)
        dt = time.time() - t0
        print(f"[检测] {len(boxes)} 区域, {dt*1000:.0f}ms")

        t0 = time.time()
        results = []
        for box in boxes:
            text, conf = self.recognize_one(img, box)
            results.append((text, conf, box))
        rt = time.time() - t0
        print(f"[识别] {len(results)} 条, {rt*1000:.0f}ms")

        return results

        return results

    def draw(self, img, results):
        """可视化结果 (PIL 渲染中文)"""
        # OpenCV BGR → PIL RGB
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vis_pil = Image.fromarray(vis)
        draw = ImageDraw.Draw(vis_pil)

        try:
            font = ImageFont.truetype(_CJK_FONT_PATH, 16)
        except Exception:
            font = ImageFont.load_default()
            print("[警告] 无法加载中文字体, 将使用默认字体")

        for text, score, box in results:
            if not text:
                continue
            pts = np.array(box).reshape(-1, 2).astype(np.int32)

            # 四边形边框
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            # PIL 绘制中文文本
            label = f"{text}"
            x, y = pts[0]
            # 文字背景
            bbox = draw.textbbox((x, y - 18), label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text((x, y - 18), label, font=font, fill=(0, 255, 0))

        # PIL RGB → OpenCV BGR
        vis = np.array(vis_pil)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        return vis


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="PP-OCRv5 OpenCV DNN CUDA OCR")
    p.add_argument("--image", "-i", required=True, help="输入图片")
    p.add_argument("--det_model", default=os.path.join(SCRIPT_DIR, "models/PP-OCRv5_mobile_det_sim.onnx"))
    p.add_argument("--rec_model", default=os.path.join(SCRIPT_DIR, "models/PP-OCRv5_mobile_rec_sim.onnx"))
    p.add_argument("--dict", default=os.path.join(SCRIPT_DIR, "models/ppocr_keys_v1.txt"))
    p.add_argument("--output", "-o", default="output.jpg")
    p.add_argument("--cpu", action="store_true", help="CPU 模式")
    p.add_argument("--det_size", type=int, default=960, help="检测 resize 边长 (640更快, 960更准)")
    p.add_argument("--benchmark", action="store_true", help="性能测试")
    args = p.parse_args()

    ocr = PPOCRv5OpenCVDNN(args.det_model, args.rec_model, args.dict,
                           use_cuda=not args.cpu,
                           det_size=args.det_size)

    if args.benchmark:
        _benchmark(ocr, args.image)
        return

    results = ocr.predict(args.image)
    print(f"\n{'='*50}")
    print(f"识别结果 ({len(results)} 条):")
    print(f"{'='*50}")
    for text, score, _ in results:
        print(f"  [{score:.3f}] {text}")

    img = cv2.imread(args.image)
    cv2.imwrite(args.output, ocr.draw(img, results))
    print(f"\n可视化: {args.output}")


def _benchmark(ocr, img_path, runs=20):
    img = cv2.imread(img_path)
    print(f"\n性能测试 ({img.shape[1]}x{img.shape[0]}, {runs}次):")

    # 检测
    times = []
    for _ in range(runs):
        t0 = time.time(); ocr.detect(img); times.append(time.time()-t0)
    print(f"  检测(det_size={ocr.det_size}): {np.mean(times)*1000:.0f}ms (±{np.std(times)*1000:.0f}ms)")

    # 识别 (单框)
    boxes, _ = ocr.detect(img)
    if boxes:
        times = []
        for _ in range(runs):
            t0 = time.time()
            ocr.recognize_one(img, boxes[0])
            times.append(time.time()-t0)
        per_box = np.mean(times) * 1000
        print(f"  识别(单框): {per_box:.1f}ms")
        print(f"  识别({len(boxes)}框): {per_box * len(boxes):.0f}ms")

    total_ms = np.mean(times[:runs])*1000*len(boxes) + np.mean(times[:runs])*1000 if boxes else 0
    print(f"  总耗时(估): ~{np.mean(times[:runs])*1000*len(boxes) + np.mean(times[:runs])*1000:.0f}ms")


if __name__ == "__main__":
    main()
