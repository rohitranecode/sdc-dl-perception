"""
ONNX Runtime Inference Wrappers
================================
Production inference classes for each perception model.
Each class:
  1. Tries to load the ONNX model from the weights/ directory
  2. Falls back to the DL-quality simulator if weights not found
  3. Exposes a unified .predict() / .detect() interface

Supported models:
  YOLOv8Detector       — yolov8n.onnx (6.2 MB)
  MiDaSDepthEstimator  — midas_v21_small_256.onnx (82 MB)
  SegFormerSegmentor   — segformer_b0_cityscapes_512x1024.onnx (15 MB)
  LaneDetector         — ufld_tusimple.onnx (75 MB)
"""

import os
import numpy as np
import cv2
import onnxruntime as ort

from architectures.yolov8_arch  import (decode_predictions, YOLOv8Simulator,
                                         _estimate_dist)
from architectures.midas_arch   import (preprocess_for_midas, postprocess_midas,
                                         render_depth, MiDaSSimulator, COLORMAPS)
from architectures.segformer_arch import (preprocess_for_segformer, postprocess_segformer,
                                           blend_seg, compute_class_stats, SegFormerSimulator)
from architectures.lane_net     import (preprocess_for_lanenet, UFLDSimulator,
                                         draw_lanes, compute_lane_metrics)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")


def _ort_session(model_name):
    """Try loading an ONNX model; return session or None."""
    path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(path):
        return None
    try:
        providers = ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads      = 4
        return ort.InferenceSession(path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"[ONNX] Failed to load {model_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8 Detector
# ─────────────────────────────────────────────────────────────────────────────
class YOLOv8Detector:
    MODEL_FILE   = "yolov8n.onnx"
    INPUT_SIZE   = (640, 640)
    CONF_THRESH  = 0.25

    def __init__(self, conf_thresh=0.25, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.session     = _ort_session(self.MODEL_FILE)
        self.sim         = YOLOv8Simulator()
        self.using_model = self.session is not None
        print(f"[YOLOv8] {'ONNX model loaded' if self.using_model else 'Simulator mode (no weights)'}")

    def detect(self, frame_bgr, seed=None):
        """Returns list of detection dicts + annotated frame."""
        if self.using_model:
            dets = self._onnx_detect(frame_bgr)
        else:
            dets = self.sim.detect(frame_bgr, self.conf_thresh, self.iou_thresh, seed)

        # Add speed estimate (simulated from class + distance)
        rng = np.random.RandomState(seed or 0)
        SPEED_RANGE = {"car":45,"truck":35,"bus":30,"person":5,"motorcycle":40,
                       "bicycle":15,"traffic light":0,"stop sign":0}
        for d in dets:
            base   = SPEED_RANGE.get(d["class"], 30)
            d["speed"] = round(max(0, base + rng.normal(0, 8)), 1)

        annotated = self._draw(frame_bgr, dets)
        return {"annotated": annotated, "detections": dets}

    def _onnx_detect(self, frame_bgr):
        ih, iw = frame_bgr.shape[:2]
        # Letterbox resize
        img = cv2.resize(frame_bgr, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # NCHW

        input_name  = self.session.get_inputs()[0].name
        raw_out     = self.session.run(None, {input_name: img})

        dets = decode_predictions(raw_out[0], self.INPUT_SIZE, (ih, iw),
                                  self.conf_thresh, self.iou_thresh)
        for d in dets:
            x1,y1,x2,y2 = d["bbox"]
            d["distance"] = _estimate_dist(y2, ih, int(ih * 0.42))
        return dets

    def _draw(self, frame, detections):
        vis = frame.copy()
        COLORS = {
            "car":(0,200,255),"truck":(255,140,0),"bus":(180,0,255),
            "person":(80,80,255),"motorcycle":(0,255,180),"bicycle":(120,255,80),
            "traffic light":(255,220,0),"stop sign":(0,60,220),
        }
        for det in detections:
            x1,y1,x2,y2 = det["bbox"]
            cls   = det["class"]
            score = det["score"]
            dist  = det["distance"]
            col   = COLORS.get(cls, (200,200,200))

            cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)
            # Corner accents
            cl = max(6, (x2-x1)//6)
            for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(vis,(cx,cy),(cx+dx*cl,cy),col,3)
                cv2.line(vis,(cx,cy),(cx,cy+dy*cl),col,3)

            label = f"{cls.upper()} {score:.0%}  {dist:.0f}m"
            (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            ly1 = max(0, y1 - lh - 6)
            cv2.rectangle(vis,(x1,ly1),(x1+lw+5,y1),col,-1)
            cv2.putText(vis, label, (x1+3,y1-3),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (10,10,10), 1)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# MiDaS Depth Estimator
# ─────────────────────────────────────────────────────────────────────────────
class MiDaSDepthEstimator:
    MODEL_FILE  = "midas_v21_small_256.onnx"
    INPUT_SIZE  = (256, 256)

    def __init__(self, max_range_m=120.0):
        self.max_range = max_range_m
        self.session   = _ort_session(self.MODEL_FILE)
        self.sim       = MiDaSSimulator()
        self.using_model = self.session is not None
        print(f"[MiDaS] {'ONNX model loaded' if self.using_model else 'Simulator mode (no weights)'}")

    def predict(self, frame_bgr, colormap="Plasma"):
        if self.using_model:
            depth_norm, metric_depth = self._onnx_predict(frame_bgr)
        else:
            depth_norm, metric_depth = self.sim.predict(frame_bgr, self.max_range)

        depth_col, blended = render_depth(depth_norm, colormap, frame_bgr, alpha=0.65)

        # Gather stats
        H, W = frame_bgr.shape[:2]
        cp   = metric_depth[H//2-30:H//2+30, W//2-30:W//2+30]
        stats = {
            "min_dist":    f"{metric_depth.min():.1f} m",
            "max_dist":    f"{metric_depth.max():.1f} m",
            "center_dist": f"{cp.mean():.1f} m",
            "mean_dist":   f"{metric_depth.mean():.1f} m",
        }
        return {
            "annotated":    blended,
            "depth_color":  depth_col,
            "depth_norm":   depth_norm,
            "metric_depth": metric_depth,
            "stats":        stats,
        }

    def _onnx_predict(self, frame_bgr):
        inp  = preprocess_for_midas(frame_bgr, self.INPUT_SIZE)
        name = self.session.get_inputs()[0].name
        raw  = self.session.run(None, {name: inp})[0]
        return postprocess_midas(raw, frame_bgr.shape[:2], self.max_range)


# ─────────────────────────────────────────────────────────────────────────────
# SegFormer Segmentor
# ─────────────────────────────────────────────────────────────────────────────
class SegFormerSegmentor:
    MODEL_FILE = "segformer_b0_cityscapes.onnx"
    INPUT_SIZE = (512, 512)

    def __init__(self, blend_alpha=0.55):
        self.alpha     = blend_alpha
        self.session   = _ort_session(self.MODEL_FILE)
        self.sim       = SegFormerSimulator()
        self.using_model = self.session is not None
        print(f"[SegFormer] {'ONNX model loaded' if self.using_model else 'Simulator mode (no weights)'}")

    def predict(self, frame_bgr):
        if self.using_model:
            return self._onnx_predict(frame_bgr)
        return self.sim.predict(frame_bgr, self.alpha)

    def _onnx_predict(self, frame_bgr):
        inp     = preprocess_for_segformer(frame_bgr, self.INPUT_SIZE)
        name    = self.session.get_inputs()[0].name
        logits  = self.session.run(None, {name: inp})[0]
        seg_map, _ = postprocess_segformer(logits, frame_bgr.shape[:2])
        blended, seg_color = blend_seg(frame_bgr, seg_map, self.alpha)
        stats   = compute_class_stats(seg_map)
        return {"seg_map":seg_map, "seg_color":seg_color,
                "annotated":blended, "stats":stats}


# ─────────────────────────────────────────────────────────────────────────────
# Lane Detector
# ─────────────────────────────────────────────────────────────────────────────
class LaneDetector:
    MODEL_FILE = "ufld_tusimple.onnx"

    def __init__(self):
        self.session     = _ort_session(self.MODEL_FILE)
        self.sim         = UFLDSimulator()
        self.using_model = self.session is not None
        print(f"[LaneNet] {'ONNX model loaded' if self.using_model else 'Simulator mode (no weights)'}")

    def detect(self, frame_bgr, seed=None):
        if self.using_model:
            lanes = self._onnx_detect(frame_bgr)
        else:
            lanes = self.sim.detect(frame_bgr, seed)

        annotated = draw_lanes(frame_bgr, lanes, fit_poly=True)
        metrics   = compute_lane_metrics(lanes, frame_bgr.shape)
        return {"annotated": annotated, "lanes": lanes, "metrics": metrics}

    def _onnx_detect(self, frame_bgr):
        inp     = preprocess_for_lanenet(frame_bgr)
        name    = self.session.get_inputs()[0].name
        out     = self.session.run(None, {name: inp})
        # UFLD output: (batch, num_lanes, num_anchors) + cls
        lanes   = self._decode_ufld(out, frame_bgr.shape)
        return lanes

    def _decode_ufld(self, outputs, orig_shape):
        """Decode UFLD row-anchor predictions."""
        # outputs[0]: (1, num_lanes, num_row_anchors) — x coordinates
        # outputs[1]: (1, num_lanes) — lane existence probabilities
        H, W  = orig_shape[:2]
        preds = outputs[0][0]  # (num_lanes, num_anchors)
        exist = outputs[1][0] if len(outputs) > 1 else np.ones(preds.shape[0])

        horizon_y = int(H * 0.42)
        lanes = []
        for i, (lane_xs, ex) in enumerate(zip(preds, exist)):
            if ex < 0.5:
                continue
            points = []
            for j, x_norm in enumerate(lane_xs):
                if x_norm < 0 or x_norm > 1:
                    continue
                y = int(horizon_y + j / len(lane_xs) * (H - horizon_y))
                x = int(x_norm * W)
                points.append((x, y))
            if len(points) > 4:
                lanes.append(np.array(points, dtype=np.float32))
        return lanes
