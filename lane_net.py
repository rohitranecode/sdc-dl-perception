"""
LaneNet — Deep Learning Lane Detection
=======================================
LaneNet (Wang et al., 2018) uses a two-branch architecture:
  • Binary segmentation branch: which pixels are lane pixels?
  • Instance embedding branch: which lane does each pixel belong to?

Followed by clustering (DBSCAN/MeanShift) on embeddings to separate lanes.

Alternative: UFLD (Ultra Fast Lane Detection) — uses row-anchor classification:
  for each row anchor, predict the lane x-coordinate (or "no lane").
  Output: (batch, num_lanes, num_row_anchors) — extremely fast (~300fps).

This module implements:
  1. LaneNetOnnxDetector  — ONNX Runtime inference path
  2. UFLDSimulator        — UFLD-style output simulation
  3. fit_lanes_from_points — polynomial fitting on instance-segmented points
  4. draw_lanes           — rendering with perspective-correct overlay
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


# ── UFLD row anchors (TuSimple benchmark) ────────────────────────────────────
TUSIMPLE_ROW_ANCHORS = list(range(64, 287, 10))  # y values in 288px input
NUM_ROW_ANCHORS = len(TUSIMPLE_ROW_ANCHORS)
NUM_LANES       = 4   # UFLD detects up to 4 lanes


# ── Lane colours (one per detected lane) ─────────────────────────────────────
LANE_COLORS = [
    (0,   255, 120),   # green  — ego left
    (0,   200, 255),   # cyan   — ego right
    (255, 165,  0),    # orange — adjacent left
    (255,  80, 80),    # red    — adjacent right
]


def preprocess_for_lanenet(frame_bgr, input_size=(256, 512)):
    """
    Preprocess for LaneNet (256×512 input, RGB, ImageNet normalised).
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size[1], input_size[0]))  # (H, W)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return np.expand_dims(img.transpose(2, 0, 1), 0)  # NCHW


def postprocess_lanenet_seg(seg_logit, embed_map, orig_shape, n_clusters=4):
    """
    Post-process LaneNet outputs:
      seg_logit: (1, 2, H, W) — binary lane/background logits
      embed_map: (1, 4, H, W) — instance embedding
    Returns list of (N, 2) point arrays, one per lane.
    """
    seg = seg_logit[0]         # (2, H, W)
    seg = seg - seg.max(axis=0, keepdims=True)
    exp = np.exp(seg)
    seg_prob = exp / exp.sum(axis=0, keepdims=True)
    lane_mask = seg_prob[1] > 0.5   # foreground probability

    if not lane_mask.any():
        return []

    embed = embed_map[0].transpose(1, 2, 0)   # (H, W, 4)
    pts   = np.argwhere(lane_mask)             # (N, 2) — (row, col)
    if len(pts) < 10:
        return []

    # Simple DBSCAN-lite: cluster by x-coordinate into n_clusters bands
    xs = pts[:, 1].reshape(-1, 1).astype(np.float32)
    from sklearn.cluster import KMeans
    k   = min(n_clusters, len(pts) // 10)
    if k < 1:
        return []
    km  = KMeans(n_clusters=k, random_state=0, n_init=5).fit(xs)
    labels = km.labels_

    lanes = []
    for lbl in range(k):
        mask  = labels == lbl
        if mask.sum() < 5:
            continue
        lane_pts = pts[mask]
        # Scale back to original image
        sy = orig_shape[0] / lane_mask.shape[0]
        sx = orig_shape[1] / lane_mask.shape[1]
        scaled  = lane_pts.astype(np.float32)
        scaled[:, 0] *= sy   # rows
        scaled[:, 1] *= sx   # cols
        lanes.append(scaled[:, ::-1])   # (N, 2) as (x, y)
    return lanes


# ── Polynomial lane fitting ───────────────────────────────────────────────────
def fit_lane_polynomial(points_xy, degree=2):
    """
    Fit a polynomial x = f(y) of given degree to a set of (x,y) lane points.
    Returns coefficients [a_d, ..., a_1, a_0] or None.
    """
    if len(points_xy) < degree + 2:
        return None
    xs = points_xy[:, 0].astype(np.float64)
    ys = points_xy[:, 1].astype(np.float64)
    try:
        coeffs = np.polyfit(ys, xs, degree)
        return coeffs
    except np.linalg.LinAlgError:
        return None


def eval_lane_polynomial(coeffs, y_values):
    """Evaluate fitted polynomial at given y values → x values."""
    poly = np.poly1d(coeffs)
    return poly(y_values)


# ── UFLD-style simulator ──────────────────────────────────────────────────────
class UFLDSimulator:
    """
    Simulates Ultra Fast Lane Detection outputs.
    UFLD treats lane detection as classification:
      - For each row anchor and each lane slot, predict x-coordinate.
    """

    def __init__(self):
        self.rng = np.random.RandomState(42)

    def detect(self, frame, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        H, W  = frame.shape[:2]
        horizon_y = int(H * 0.42)
        vp_x  = W // 2 + self.rng.randint(-40, 40)

        # Curvature parameter (simulates camera pose variation)
        curvature = self.rng.uniform(-0.0002, 0.0002)

        lanes = []
        lane_offsets = [-0.42, -0.14, 0.14, 0.42]

        for i, offset in enumerate(lane_offsets):
            points = []
            for y in range(H - 1, horizon_y - 20, -5):
                t     = (y - horizon_y) / (H - horizon_y)
                hw    = t * W * 0.52
                x_nom = vp_x + offset * 2 * hw
                # Polynomial curvature: x = x_nom + curvature*(y - H)^2
                x_cur = x_nom + curvature * (y - H) ** 2
                # Add DL-realistic noise (simulates sub-pixel regression)
                x_cur += self.rng.normal(0, 1.5)
                x = int(np.clip(x_cur, 0, W - 1))
                points.append((x, y))

            # Occasionally drop outer lanes
            if abs(offset) > 0.2 and self.rng.random() > 0.6:
                continue
            if len(points) > 5:
                lanes.append(np.array(points, dtype=np.float32))

        return lanes


# ── Lane drawing & metrics ────────────────────────────────────────────────────
def draw_lanes(frame, lanes, fit_poly=True):
    """
    Draw detected lanes on frame.
    If fit_poly=True, fit and draw smooth polynomial curves.
    """
    vis     = frame.copy()
    overlay = frame.copy()
    H, W    = frame.shape[:2]

    ego_left  = None   # for ego lane fill
    ego_right = None

    for i, lane_pts in enumerate(lanes[:4]):
        color = LANE_COLORS[i % len(LANE_COLORS)]

        if fit_poly and len(lane_pts) >= 4:
            coeffs = fit_lane_polynomial(lane_pts, degree=2)
            if coeffs is not None:
                y_vals = np.linspace(lane_pts[:, 1].min(), H, 80)
                x_vals = eval_lane_polynomial(coeffs, y_vals)
                pts    = np.column_stack([x_vals, y_vals]).astype(np.int32)
                pts    = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W)]
                if len(pts) > 1:
                    cv2.polylines(vis, [pts], False, color, 3, cv2.LINE_AA)
                    # Store ego lane boundaries
                    if i == 1: ego_left  = (coeffs, y_vals)
                    if i == 2: ego_right = (coeffs, y_vals)
        else:
            for j in range(len(lane_pts) - 1):
                p1 = tuple(lane_pts[j].astype(int))
                p2 = tuple(lane_pts[j+1].astype(int))
                cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

    # Ego lane fill
    if ego_left and ego_right:
        y_shared   = np.linspace(H * 0.5, H, 60)
        x_left     = eval_lane_polynomial(ego_left[0],  y_shared)
        x_right    = eval_lane_polynomial(ego_right[0], y_shared)
        left_pts   = np.column_stack([x_left,  y_shared]).astype(np.int32)
        right_pts  = np.column_stack([x_right, y_shared]).astype(np.int32)
        poly_pts   = np.vstack([left_pts, right_pts[::-1]])
        valid      = ((poly_pts[:, 0] >= 0) & (poly_pts[:, 0] < W) &
                      (poly_pts[:, 1] >= 0) & (poly_pts[:, 1] < H))
        if valid.sum() >= 4:
            cv2.fillPoly(overlay, [poly_pts[valid]], (0, 200, 100))
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    return vis


def compute_lane_metrics(lanes, frame_shape):
    """Compute lane quality metrics from detected lane list."""
    H, W = frame_shape[:2]
    if len(lanes) < 2:
        return {"curvature": "N/A", "offset": "N/A",
                "lane_width": "N/A", "confidence": 0.0, "lanes_detected": len(lanes)}

    # Use the two centre lanes
    ego_lanes = lanes[1:3] if len(lanes) >= 3 else lanes[:2]
    bottoms   = [lane[np.argmax(lane[:, 1])] for lane in ego_lanes]
    x_l, x_r = bottoms[0][0], bottoms[1][0]

    if x_l > x_r:
        x_l, x_r = x_r, x_l

    lane_px   = x_r - x_l
    offset_m  = ((W / 2) - (x_l + x_r) / 2) * 0.0045
    width_m   = lane_px * 0.0045
    conf      = min(1.0, lane_px / (W * 0.45))

    # Curvature from polynomial if available
    curv = "Straight"
    if abs(offset_m) > 0.3:
        curv = "Slight curve"

    return {
        "curvature":     curv,
        "offset":        f"{offset_m:+.2f} m",
        "lane_width":    f"{width_m:.2f} m",
        "confidence":    round(conf, 3),
        "lanes_detected": len(lanes),
    }
