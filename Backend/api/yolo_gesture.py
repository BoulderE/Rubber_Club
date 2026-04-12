import os
import time
import json
from typing import Tuple, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

DET_MODEL_PATH = os.environ.get("DET_MODEL_PATH", "models/hand_detector.onnx")
CLS_MODEL_PATH = os.environ.get("CLS_MODEL_PATH", "models/gesture_classifier.onnx")  
DET_INPUT_SIZE = (640, 640)  
CLS_INPUT_SIZE = (160, 160)  
DET_CONF_TH = float(os.environ.get("DET_CONF_TH", "0.25"))
DET_IOU_TH = float(os.environ.get("DET_IOU_TH", "0.45"))
MAX_DETS = int(os.environ.get("MAX_DETS", "100"))
USE_CLASSIFIER = os.environ.get("USE_CLASSIFIER", "0") == "1"  

def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = nw - new_unpad[0], nh - new_unpad[1]
    dw //= 2
    dh //= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, nh - new_unpad[1] - dh, dw, nw - new_unpad[0] - dw,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float = 0.45, topk: int = 100) -> List[int]:
    if boxes.size == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < topk:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_th]
    return keep

class YOLOGesture:
    def __init__(self,
                 det_model_path: str = DET_MODEL_PATH,
                 cls_model_path: Optional[str] = CLS_MODEL_PATH,
                 det_input_size: Tuple[int, int] = DET_INPUT_SIZE,
                 cls_input_size: Tuple[int, int] = CLS_INPUT_SIZE,
                 conf_th: float = DET_CONF_TH,
                 iou_th: float = DET_IOU_TH,
                 max_dets: int = MAX_DETS):
        self.det_input_size = det_input_size
        self.cls_input_size = cls_input_size
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.max_dets = max_dets

        self.providers = self._init_providers()
        self.det_sess = ort.InferenceSession(det_model_path, providers=self.providers)
        self.det_in_name = self.det_sess.get_inputs()[0].name

        self.cls_sess = None
        self.cls_in_name = None
        if cls_model_path and os.path.exists(cls_model_path):
            try:
                self.cls_sess = ort.InferenceSession(cls_model_path, providers=self.providers)
                self.cls_in_name = self.cls_sess.get_inputs()[0].name
            except Exception as e:
                print(f"[WARN] 未能加载分类器模型：{e}")

    @staticmethod
    def _init_providers():
        avail = ort.get_available_providers()
        if "CoreMLExecutionProvider" in avail:
            coreml_opts = {
                "MLComputeUnits": "ALL",          
                "ModelFormat": "MLProgram",       
                "EnableOnSubgraphs": "1",        
                "RequireStaticInputShapes": "0"  
            }
            providers = [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"]
            print("[INFO] 使用 CoreMLExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
            print("[INFO] CoreML 不可用，使用 CPUExecutionProvider")
        return providers

    def preprocess_det(self, bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        size = self.det_input_size
        x, r, (left, top) = letterbox(rgb, size)
        x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        return x, r, left, top

    def postprocess_det(self, pred: np.ndarray, r: float, left: int, top: int,
                         orig_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        pred = pred[0]  
        boxes = pred[:, :4]
        obj_score = pred[:, 4]
        cls_scores = pred[:, 5:]
        cls_ids = cls_scores.argmax(axis=1)
        cls_max = cls_scores.max(axis=1)
        scores = obj_score * cls_max

        m = scores > self.conf_th
        boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes = xywh2xyxy(boxes)
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= r

        h0, w0 = orig_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0 - 1)

        keep = nms(boxes, scores, self.iou_th, self.max_dets)
        return boxes[keep].astype(np.float32), scores[keep].astype(np.float32), cls_ids[keep].astype(np.int64)

    def preprocess_cls(self, bgr_crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.cls_input_size[1], self.cls_input_size[0]), interpolation=cv2.INTER_LINEAR)
        x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        return x

    def classify(self, bgr: np.ndarray) -> Optional[int]:
        if self.cls_sess is None:
            return None
        x = self.preprocess_cls(bgr)
        out = self.cls_sess.run(None, {self.cls_in_name: x})
        prob = out[0].squeeze()
        cls_id = int(np.argmax(prob))
        return cls_id

    def detect(self, bgr: np.ndarray) -> List[dict]:
        inp, r, left, top = self.preprocess_det(bgr)
        out = self.det_sess.run(None, {self.det_in_name: inp})
        boxes, scores, cls_ids = self.postprocess_det(out[0], r, left, top, bgr.shape[:2])

        results = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            det = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(scores[i]),
                "cls": int(cls_ids[i])
            }
            if USE_CLASSIFIER and self.cls_sess is not None:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                crop = bgr[max(0, y1i):max(0, y2i), max(0, x1i):max(0, x2i)]
                if crop.size > 0:
                    det["gesture_id"] = self.classify(crop)
            results.append(det)
        return results

def draw_dets(img: np.ndarray, dets: List[dict]) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        score = d["score"]
        cls_id = d["cls"]
        label = f"id:{cls_id} {score:.2f}"
        if "gesture_id" in d and d["gesture_id"] is not None:
            label += f" g:{d['gesture_id']}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def main_camera():
    print("[INFO] 初始化 YOLO 手势识别...")
    yolo = YOLOGesture()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头 0")

    t_prev = None
    print("[INFO] 按 ESC 退出窗口")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = yolo.detect(frame)

        now = time.time()
        if t_prev is None:
            fps = 0.0
        else:
            fps = 1.0 / max(1e-3, now - t_prev)
        t_prev = now

        vis = draw_dets(frame, dets)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("YOLO Hand Detect (CoreML)", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_camera()