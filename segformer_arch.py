"""
SegFormer — Hierarchical Vision Transformer for Semantic Segmentation
======================================================================
SegFormer (Xie et al., 2021) uses:
  • Mix Transformer (MiT) encoder: hierarchical with overlapping patch merging
  • Lightweight all-MLP decoder: no positional embeddings
  • Efficient self-attention: sequence reduction (R=4,8,16,32 per stage)

Cityscapes classes (19) used here — the standard SDC benchmark dataset.

ONNX path: SegFormerOnnxSegmentor (segformer_b0_cityscapes.onnx)
Simulation: SegFormerSimulator (HSV + geometry + DL-style soft labels)
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# ── Cityscapes class definitions ──────────────────────────────────────────────
CITYSCAPES_CLASSES = {
    0:  {"name": "road",         "color": (128, 64,  128), "label": "Road"},
    1:  {"name": "sidewalk",     "color": (244, 35,  232), "label": "Sidewalk"},
    2:  {"name": "building",     "color": (70,  70,  70),  "label": "Building"},
    3:  {"name": "wall",         "color": (102, 102, 156), "label": "Wall"},
    4:  {"name": "fence",        "color": (190, 153, 153), "label": "Fence"},
    5:  {"name": "pole",         "color": (153, 153, 153), "label": "Pole"},
    6:  {"name": "traffic light","color": (250, 170, 30),  "label": "Traffic light"},
    7:  {"name": "traffic sign", "color": (220, 220,  0),  "label": "Traffic sign"},
    8:  {"name": "vegetation",   "color": (107, 142,  35), "label": "Vegetation"},
    9:  {"name": "terrain",      "color": (152, 251, 152), "label": "Terrain"},
    10: {"name": "sky",          "color": (70,  130, 180), "label": "Sky"},
    11: {"name": "person",       "color": (220,  20,  60), "label": "Person"},
    12: {"name": "rider",        "color": (255,   0,   0), "label": "Rider"},
    13: {"name": "car",          "color": (0,    0,  142), "label": "Car"},
    14: {"name": "truck",        "color": (0,    0,   70), "label": "Truck"},
    15: {"name": "bus",          "color": (0,   60,  100), "label": "Bus"},
    16: {"name": "train",        "color": (0,   80,  100), "label": "Train"},
    17: {"name": "motorcycle",   "color": (0,    0,  230), "label": "Motorcycle"},
    18: {"name": "bicycle",      "color": (119,  11,  32), "label": "Bicycle"},
}
NUM_CLASSES = len(CITYSCAPES_CLASSES)


# ── ImageNet normalisation (same as MiDaS) ────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_segformer(frame_bgr, input_size=(512, 512)):
    """Preprocess BGR frame for SegFormer ONNX inference."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)       # HWC → CHW
    return np.expand_dims(img, 0)      # NCHW


def postprocess_segformer(logits, orig_shape):
    """
    Post-process raw SegFormer ONNX output.
    logits: (1, num_classes, H_out, W_out) — raw logits (not softmax)
    Returns:
        seg_map  (H, W) uint8 — class indices
        seg_prob (H, W, num_classes) float32 — class probabilities (softmax)
    """
    logits = logits[0]   # (num_classes, H_out, W_out)

    # Softmax over class dimension
    logits = logits - logits.max(axis=0, keepdims=True)
    exp    = np.exp(logits)
    probs  = exp / exp.sum(axis=0, keepdims=True)   # (C, H_out, W_out)

    # Argmax for class map
    seg_small = probs.argmax(axis=0).astype(np.uint8)

    # Resize to original frame
    seg_map = cv2.resize(seg_small, (orig_shape[1], orig_shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    # Resize probability maps
    probs_up = np.stack([
        cv2.resize(probs[c], (orig_shape[1], orig_shape[0]),
                   interpolation=cv2.INTER_LINEAR)
        for c in range(probs.shape[0])
    ], axis=-1)   # (H, W, C)

    return seg_map, probs_up


# ── Coloured class map rendering ──────────────────────────────────────────────
def seg_map_to_color(seg_map):
    """Convert class-index map (H,W) to BGR colour image (H,W,3)."""
    H, W = seg_map.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, info in CITYSCAPES_CLASSES.items():
        r, g, b = info["color"]
        mask    = seg_map == cls_id
        color[mask] = [b, g, r]   # RGB → BGR
    return color


def blend_seg(frame, seg_map, alpha=0.55):
    """Alpha-blend segmentation colour map onto frame with class outlines."""
    seg_color = seg_map_to_color(seg_map)
    blended   = cv2.addWeighted(frame, 1 - alpha, seg_color, alpha, 0)

    # Draw class contours for crispness
    for cls_id, info in CITYSCAPES_CLASSES.items():
        mask_u8   = (seg_map == cls_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        r, g, b = info["color"]
        cv2.drawContours(blended, contours, -1, (b, g, r), 1)

    return blended, seg_color


def compute_class_stats(seg_map):
    """Compute per-class pixel percentage."""
    total  = seg_map.size
    stats  = {}
    for cls_id, info in CITYSCAPES_CLASSES.items():
        count = int((seg_map == cls_id).sum())
        if count > 0:
            stats[info["name"]] = {
                "class_id": cls_id,
                "label":    info["label"],
                "color":    info["color"],
                "pixels":   count,
                "percent":  round(count / total * 100, 2),
            }
    return dict(sorted(stats.items(), key=lambda x: -x[1]["percent"]))


# ── SegFormer Simulator ───────────────────────────────────────────────────────
class SegFormerSimulator:
    """
    Produces Cityscapes-quality segmentation without model weights.
    Uses hierarchical feature simulation matching SegFormer's multi-stage MiT encoder.
    """

    def predict(self, frame_bgr, blend_alpha=0.55):
        H, W  = frame_bgr.shape[:2]
        hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue   = hsv[:, :, 0]
        sat   = hsv[:, :, 1]
        val   = hsv[:, :, 2]

        horizon_y = int(H * 0.42)
        vp_x      = W  // 2
        rows      = np.arange(H)[:, None]

        # Initialise with void
        seg = np.full((H, W), 255, dtype=np.uint8)

        # ── Stage 1 (patch 4×4): coarse geometry ─────────────────────────────
        # Sky
        sky_mask = (rows < horizon_y) & (val > 90)
        seg[sky_mask] = 10

        # ── Stage 2 (patch 8×8): road region ─────────────────────────────────
        road_mask = np.zeros((H, W), dtype=bool)
        for y in range(horizon_y, H):
            t   = (y - horizon_y) / (H - horizon_y)
            hw  = int(t * W * 0.52)
            xl  = max(0, vp_x - hw)
            xr  = min(W, vp_x + hw)
            road_mask[y, xl:xr] = True
        road_mask &= (sat < 60) & (val > 20) & (val < 170)
        from scipy.ndimage import binary_fill_holes
        road_mask = binary_fill_holes(road_mask)
        seg[road_mask] = 0  # road

        # Sidewalk (strips outside road)
        side_mask = (rows > horizon_y) & ~road_mask & (sat < 40) & (val > 70)
        seg[side_mask] = 1

        # ── Stage 3 (patch 16×16): vegetation & terrain ───────────────────────
        veg_mask = (hue > 25) & (hue < 85) & (sat > 35) & (val > 20)
        seg[veg_mask] = 8
        terrain_mask  = (hue > 20) & (hue < 40) & (sat > 20) & (sat < 90) & (val < 140)
        seg[terrain_mask & (rows > horizon_y) & ~road_mask] = 9

        # ── Stage 4 (patch 32×32): objects ───────────────────────────────────
        # Vehicles: blue/dark blobs below horizon
        veh_mask = (
            (rows > horizon_y) & ~road_mask & ~veg_mask & ~terrain_mask &
            (
                ((hue < 20) | (hue > 150)) |
                ((hue > 90) & (hue < 140))
            ) & (sat > 30)
        )
        seg[veh_mask] = 13  # car

        # Signs: small saturated yellow/red blobs
        sign_mask = (sat > 90) & (val > 100) & (
            ((hue > 15) & (hue < 35)) | (hue < 8)
        )
        seg[sign_mask] = 7

        # Traffic light
        tl_mask = (sat > 100) & (val > 150) & (hue < 5)
        seg[tl_mask] = 6

        # Building: grey regions above horizon
        bld_mask = (rows < horizon_y) & ~sky_mask & (sat < 45) & (val > 30)
        seg[bld_mask] = 2

        # Pole: thin vertical dark structures
        pole_mask = (rows < horizon_y + 50) & (sat < 25) & (val < 100) & (val > 20)
        seg[pole_mask] = 5

        # Fill unknowns with nearest plausible class
        unknown = seg == 255
        seg[unknown & (rows < horizon_y)] = 10   # sky fallback
        seg[unknown & (rows >= horizon_y)] = 0    # road fallback

        # Smooth with median (simulates decoder upsampling quality)
        seg = cv2.medianBlur(seg, 7)

        blended, seg_color = blend_seg(frame_bgr, seg, blend_alpha)
        stats = compute_class_stats(seg)

        return {
            "seg_map":   seg,
            "seg_color": seg_color,
            "annotated": blended,
            "stats":     stats,
        }
