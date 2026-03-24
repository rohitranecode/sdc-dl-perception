"""
╔══════════════════════════════════════════════════════════════╗
║    SELF-DRIVING CAR — DEEP LEARNING PERCEPTION  v3.0         ║
║    YOLOv8 · MiDaS · SegFormer · LaneNet (UFLD)               ║
║    ONNX Runtime · CPU inference · Cityscapes + COCO           ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time, sys, os, io

sys.path.insert(0, os.path.dirname(__file__))

from utils.scene_generator  import generate_road_scene
from utils.visualization    import (hud_overlay, pipeline_grid, warp_bev,
                                    project_lanes_bev, occupancy_map)
from inference.model_runner import (YOLOv8Detector, MiDaSDepthEstimator,
                                    SegFormerSegmentor, LaneDetector)
from architectures.midas_arch import COLORMAPS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDC DL Perception",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;500;700&display=swap');
  html,body,[class*="css"]{ font-family:'Syne',sans-serif; }
  .stApp{ background:#080b12; color:#c4d0e0; }
  [data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b0f1a 0%,#0f1623 100%); border-right:1px solid #1a2840; }
  [data-testid="stMetric"]{ background:#0d1424; border:1px solid #162038; border-radius:8px; padding:10px 14px !important; }
  [data-testid="stMetricLabel"]{ color:#4a6280 !important; font-size:0.70rem !important; font-family:'JetBrains Mono',monospace !important; }
  [data-testid="stMetricValue"]{ color:#00c8ff !important; font-family:'JetBrains Mono',monospace !important; }
  .stButton>button{ background:linear-gradient(135deg,#0ea5e9,#0055aa); color:#fff; border:none; border-radius:6px; font-family:'JetBrains Mono',monospace; letter-spacing:.06em; box-shadow:0 0 10px rgba(14,165,233,.28); transition:all .2s; }
  .stButton>button:hover{ transform:translateY(-1px); box-shadow:0 0 20px rgba(14,165,233,.5); }
  .stTabs [data-baseweb="tab-list"]{ background:#0d1424; border-bottom:1px solid #162038; }
  .stTabs [data-baseweb="tab"]{ color:#4a6280; font-family:'JetBrains Mono',monospace; font-size:.78rem; }
  .stTabs [aria-selected="true"]{ background:#162038 !important; color:#00c8ff !important; border-bottom:2px solid #00c8ff; }
  h1{ color:#00c8ff !important; font-family:'JetBrains Mono',monospace !important; font-size:1.3rem !important; letter-spacing:.08em; }
  h2{ color:#38bdf8 !important; font-size:1rem !important; }
  h3{ color:#7dd3fc !important; font-size:.9rem !important; }
  label{ color:#4a6280 !important; font-size:.80rem !important; }
  .stProgress>div>div{ background:linear-gradient(90deg,#0ea5e9,#00c8ff); }
  hr{ border-color:#162038; }
  .model-badge-on  { display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;background:#0d2b0d;color:#4ade80;border:1px solid #1a5c1a;font-family:monospace; }
  .model-badge-sim { display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;background:#1a1a08;color:#facc15;border:1px solid #4a4000;font-family:monospace; }
</style>
""", unsafe_allow_html=True)


# ── Model initialisation (cached — load once per session) ─────────────────────
@st.cache_resource(show_spinner="Loading deep learning models…")
def load_models():
    det   = YOLOv8Detector(conf_thresh=0.25)
    depth = MiDaSDepthEstimator(max_range_m=120)
    seg   = SegFormerSegmentor(blend_alpha=0.55)
    lane  = LaneDetector()
    return det, depth, seg, lane


@st.cache_data(show_spinner=False)
def _gen_scene(tod, weather, seed):
    return generate_road_scene(time_of_day=tod, weather=weather, seed=seed)


def bgr2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DL PERCEPTION")
    st.markdown("---")

    st.markdown("### 📷 Input")
    mode = st.radio("", ["Synthetic Scene", "Upload Image"], label_visibility="collapsed")

    if mode == "Synthetic Scene":
        st.markdown("### 🌍 Scene")
        tod     = st.select_slider("Time of Day", ["day","dusk","night"], "day")
        weather = st.selectbox("Weather", ["clear","rain","fog","snow"])
        seed    = st.slider("Variation Seed", 0, 99, 42)
        auto    = st.toggle("▶ Auto-cycle", value=False)
    else:
        uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp"])

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    conf_thresh   = st.slider("Detection Confidence",  0.10, 0.90, 0.25, 0.05)
    depth_cmap    = st.selectbox("Depth Colormap", list(COLORMAPS.keys()), 0)
    seg_alpha     = st.slider("Seg Blend Alpha", 0.20, 0.85, 0.55, 0.05)
    max_depth     = st.slider("Max Depth Range (m)", 30, 200, 120, 10)




# ── Load models ───────────────────────────────────────────────────────────────
detector, depth_model, segmentor, lane_detector = load_models()
depth_model.max_range = max_depth

# Update conf threshold dynamically
detector.conf_thresh = conf_thresh
segmentor.alpha      = seg_alpha



# ── Get frame ─────────────────────────────────────────────────────────────────
frame_bgr = None
auto = False  # default — only True in Synthetic Scene mode with toggle on

if mode == "Upload Image":
    if 'uploaded' in dir() and uploaded:
        pil = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        st.info("⬅  Upload a dashcam image to run inference.")
        st.stop()
else:
    if auto:
        if "aseed" not in st.session_state:
            st.session_state.aseed = 0
        seed = st.session_state.aseed % 100
        st.session_state.aseed += 1
        time.sleep(0.05)
    with st.spinner("Generating synthetic scene…"):
        frame_bgr, meta = _gen_scene(tod, weather, seed)


# ── Run full DL pipeline ──────────────────────────────────────────────────────
t0 = time.perf_counter()

with st.spinner("Running DL inference…"):
    _seed = seed if mode == "Synthetic Scene" else 42
    lane_res  = lane_detector.detect(frame_bgr, seed=_seed)
    obj_res   = detector.detect(frame_bgr, seed=_seed)
    depth_res = depth_model.predict(frame_bgr, colormap=depth_cmap)
    seg_res   = segmentor.predict(frame_bgr)

    bev_img, M   = warp_bev(frame_bgr)
    bev_lanes    = project_lanes_bev(bev_img, lane_res["lanes"], M)
    occ          = occupancy_map(obj_res["detections"], frame_bgr.shape)

    model_info = {
        "YOLOv8":    detector.using_model,
        "MiDaS":     depth_model.using_model,
        "SegFormer": segmentor.using_model,
        "LaneNet":   lane_detector.using_model,
    }
    hud_frame = hud_overlay(
        obj_res["annotated"].copy(),
        lane_res["metrics"],
        obj_res["detections"],
        depth_res["stats"],
        model_info=model_info,
        speed=65,
    )
    grid = pipeline_grid(
        frame_bgr, lane_res["annotated"], obj_res["annotated"],
        depth_res["depth_color"], seg_res["annotated"], bev_lanes, occ
    )

elapsed_ms = (time.perf_counter() - t0) * 1000


# ── Top metrics ───────────────────────────────────────────────────────────────
st.markdown("# 🧠 SELF-DRIVING CAR — DL PERCEPTION SYSTEM")
st.markdown("*YOLOv8 · MiDaS · SegFormer-B0 · LaneNet/UFLD — ONNX Runtime inference*")
st.markdown("---")

m = st.columns(7)
lm   = lane_res["metrics"]
dets = obj_res["detections"]
ds   = depth_res["stats"]
conf = float(lm.get("confidence",0))

m[0].metric("⚡ Pipeline ms",  f"{elapsed_ms:.0f}")
m[1].metric("🛣️ Lanes Found",  str(lm.get("lanes_detected",0)))
m[2].metric("📐 Lane Conf",    f"{conf:.0%}")
m[3].metric("📦 Detections",   str(len(dets)))
m[4].metric("📏 Nearest",
            f"{dets[0]['distance']:.0f}m" if dets else "—",
            delta="CAUTION" if dets and dets[0]["distance"]<30 else None,
            delta_color="inverse")
m[5].metric("🔭 Center Depth", ds.get("center_dist","N/A"))
m[6].metric("🎨 Road Coverage",
            f"{seg_res['stats'].get('road',{}).get('percent',0):.1f}%")

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Full Pipeline",
    "🛣️ LaneNet",
    "📦 YOLOv8",
    "🔭 MiDaS Depth",
    "🎨 SegFormer",
    "🗺️ BEV + Occupancy",
    "📊 Analytics",
    "🏗️ Architectures",
])


# ─────────── TAB 0: FULL PIPELINE ────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### HUD — All modules fused")
    st.image(bgr2pil(hud_frame), use_container_width=True)
    st.markdown("#### 2×3 DL Perception Grid")
    st.image(bgr2pil(grid), use_container_width=True,
             caption="Raw · LaneNet · YOLOv8 · MiDaS Depth · SegFormer · Bird's Eye View")
    if auto:
        st.rerun()


# ─────────── TAB 1: LANENET ──────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### LaneNet / UFLD Deep Learning Lane Detection")
    st.image(bgr2pil(lane_res["annotated"]), use_container_width=True,
             caption=f"Detected {lm.get('lanes_detected',0)} lanes (polynomial fit, degree=2)")

    st.markdown("#### Lane Metrics")
    lc1,lc2,lc3,lc4 = st.columns(4)
    lc1.metric("Curvature",     lm.get("curvature","N/A"))
    lc2.metric("Lateral Offset",lm.get("offset","N/A"))
    lc3.metric("Lane Width",    lm.get("lane_width","N/A"))
    lc4.metric("Confidence",    f"{conf:.1%}")

    with st.expander("🏗️ Architecture: Ultra Fast Lane Detection (UFLD)"):
        st.markdown("""
**Input:** RGB frame → 288×800 (resized) → ImageNet normalised NCHW tensor

**Backbone:** ResNet-18/34 pretrained on ImageNet

**UFLD approach — row-anchor classification:**
Instead of dense binary segmentation, UFLD reformulates lane detection as classification:
- Divide the image into `num_row_anchors` horizontal strips (e.g. 56 strips)
- For each strip and each lane slot (up to 4 lanes), classify which column the lane falls in
- Output: `(batch, num_lanes, num_row_anchors, num_grid_cols + 1)` — the +1 is the "no lane" class

**Why this is faster:**
The row-anchor approach avoids pixel-level prediction. Instead of predicting 288×800=230,400 pixel labels, 
UFLD predicts only `4 × 56 × 101 = 22,624` class probabilities — 10× fewer predictions, ~300fps on GPU.

**Post-processing:**
1. Softmax over grid columns → select argmax x-coordinate per row
2. Filter rows where "no lane" class wins
3. Fit degree-2 polynomial x=f(y) to surviving (x,y) pairs
4. Draw smooth curve via polynomial evaluation
        """)


# ─────────── TAB 2: YOLOV8 ───────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### YOLOv8 Object Detection")
    st.image(bgr2pil(obj_res["annotated"]), use_container_width=True,
             caption=f"YOLOv8n — {len(dets)} detections (conf ≥ {conf_thresh:.0%}, IoU NMS 0.45)")

    st.markdown("#### Detection Results")
    if dets:
        ICONS = {"car":"🚗","truck":"🚛","bus":"🚌","person":"🚶","motorcycle":"🏍️",
                 "bicycle":"🚴","traffic light":"🚦","stop sign":"🛑"}
        hdr = st.columns([1.8,1.6,1,1,1.2])
        for t,c in zip(["CLASS","CONFIDENCE","DISTANCE","SPEED","BOUNDING BOX"],hdr):
            c.markdown(f"**{t}**")
        for det in dets:
            row = st.columns([1.8,1.6,1,1,1.2])
            x1,y1,x2,y2 = det["bbox"]
            row[0].markdown(f"{ICONS.get(det['class'],'📦')} **{det['class'].upper()}**")
            row[1].progress(det["score"])
            row[2].markdown(f"`{det['distance']:.0f} m`")
            row[3].markdown(f"`{det['speed']:.0f} km/h`")
            row[4].markdown(f"`[{x1},{y1},{x2},{y2}]`")
    else:
        st.success("✅ No objects detected above confidence threshold.")

    with st.expander("🏗️ Architecture: YOLOv8 Anchor-Free Detector"):
        st.markdown("""
**Input:** 640×640 RGB NCHW float32 (letterboxed + normalised [0,1])

**Backbone:** CSPDarknet with C2f blocks (Cross-Stage Partial with 2 bottlenecks)

**Neck:** PAN-FPN (Path Aggregation Network) at strides 8, 16, 32

**Head:** Decoupled anchor-free head — two separate branches:
- **Classification branch:** `(batch, num_classes, H_feat, W_feat)`
- **Regression branch:** `(batch, 4 × reg_max, H_feat, W_feat)` — DFL encoding

**Anchor-free design:** Predicts `(cx, cy, w, h)` directly at each grid cell. No anchor priors.

**DFL (Distribution Focal Loss):** Bounding box regression modelled as a discrete distribution 
over `reg_max=16` bins. Instead of predicting a single distance value, predicts 16 probabilities 
per side. The final distance = `Σ(prob_i × i)` — a learned expectation.

**Output tensor:** `(1, 84, 8400)` — 84 = 4 coords + 80 COCO classes, 8400 = 80²/64 + 40²/16 + 20²/4 anchor points

**NMS:** Per-class IoU-based NMS at 0.45 threshold
        """)


# ─────────── TAB 3: MiDaS ────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### MiDaS Monocular Depth Estimation")
    dc1,dc2 = st.columns(2)
    with dc1:
        st.image(bgr2pil(depth_res["annotated"]), caption="Depth blend (65% depth + 35% RGB)", use_container_width=True)
    with dc2:
        st.image(bgr2pil(depth_res["depth_color"]), caption=f"Pure depth map ({depth_cmap} colormap)", use_container_width=True)

    st.markdown("#### Depth Statistics")
    dd = depth_res["stats"]
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Min Distance", dd["min_dist"])
    d2.metric("Max Distance", dd["max_dist"])
    d3.metric("Center Distance", dd["center_dist"])
    d4.metric("Mean Distance", dd["mean_dist"])

    # Histogram
    st.markdown("#### Depth Distribution")
    import matplotlib.pyplot as plt
    depth_flat = depth_res["metric_depth"].flatten()
    hist, bins = np.histogram(depth_flat, bins=60, range=(0, max_depth))
    fig, ax = plt.subplots(figsize=(10,2.5),facecolor="#080b12")
    ax.set_facecolor("#0d1424")
    ax.fill_between(bins[:-1],hist,alpha=0.75,color="#0ea5e9",step="post")
    ax.set_xlabel("Distance (m)",color="#4a6280"); ax.set_ylabel("Pixels",color="#4a6280")
    ax.set_title("Scene depth distribution (MiDaS output)", color="#00c8ff")
    ax.tick_params(colors="#4a6280")
    for s in ax.spines.values(): s.set_edgecolor("#162038")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("🏗️ Architecture: MiDaS / DPT"):
        st.markdown(f"""
**Input:** `(1, 3, 256, 256)` NCHW float32 — ImageNet normalised RGB

**Encoder (MiDaS v2.1 Small):** EfficientNet-Lite3 pretrained on ~1.4M images  
(Mix of ReDWeb, DIML, MegaDepth, WSVD, 3D Movies, MiDaS-3D)

**Decoder:** Multi-scale feature fusion with 4 skip connections:
```
Encoder features: 1/4, 1/8, 1/16, 1/32 resolution
Decoder: reassemble + fusion blocks → dense upsampling
```

**Loss: Scale-shift invariant:**  
```
L = ||d̂ - (s·d_gt + t)||₁   where s,t = least-squares scale/shift per image
```
This handles the metric ambiguity in monocular depth — the model learns relative depth.

**Output:** `(1, H_in, W_in)` float32 — **inverse relative depth** (higher = closer)

**Post-processing:**
1. Resize to original frame via bilinear interpolation
2. Invert (MiDaS outputs closer=higher, we want closer=lower for metric projection)
3. Normalise to [0,1] → project to [0, {max_depth}m]
        """)


# ─────────── TAB 4: SEGFORMER ────────────────────────────────────────────────
with tabs[4]:
    st.markdown("#### SegFormer-B0 Semantic Segmentation (Cityscapes, 19 classes)")
    sc1,sc2 = st.columns(2)
    with sc1:
        st.image(bgr2pil(seg_res["annotated"]), caption="Segmentation blend", use_container_width=True)
    with sc2:
        st.image(bgr2pil(seg_res["seg_color"]), caption="Pure class map", use_container_width=True)

    st.markdown("#### Class Coverage")
    from architectures.segformer_arch import CITYSCAPES_CLASSES
    stats = seg_res["stats"]
    sorted_stats = sorted(stats.items(), key=lambda x: -x[1]["percent"])
    for cls_name, cls_data in sorted_stats:
        if cls_data["percent"] < 0.5: continue
        r,g,b = cls_data["color"]
        col_hex = f"#{r:02x}{g:02x}{b:02x}"
        pc,ic = st.columns([5,1])
        pc.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
            f'<span style="width:11px;height:11px;border-radius:2px;background:{col_hex};display:inline-block"></span>'
            f'<span style="color:#c4d0e0;font-size:.80rem;width:130px">{cls_data["label"]}</span>'
            f'</div>', unsafe_allow_html=True)
        pc.progress(cls_data["percent"]/100)
        ic.markdown(f"`{cls_data['percent']:.1f}%`")

    with st.expander("🏗️ Architecture: SegFormer"):
        st.markdown("""
**Input:** `(1, 3, 512, 512)` NCHW — ImageNet normalised

**Encoder — Mix Transformer (MiT-B0):**
4 hierarchical stages with overlapping patch merging:

| Stage | Patch | Stride | Channels | Heads | Seq Reduction |
|-------|-------|--------|----------|-------|---------------|
| 1     | 7×7   | 4      | 32       | 1     | R=8           |
| 2     | 3×3   | 2      | 64       | 2     | R=4           |
| 3     | 3×3   | 2      | 160      | 5     | R=2           |
| 4     | 3×3   | 2      | 256      | 8     | R=1           |

**Efficient self-attention:** Reduces key/value sequence by factor R before attention:
`K' = Reshape(K, R²)·W` — reduces complexity from O(N²) to O(N²/R²)

**Decoder — All-MLP:**
No positional embeddings. Simply upsample all 4 feature maps to 1/4 resolution, 
concatenate, and apply two MLP layers → `(1, 19, H/4, W/4)` logits.
Final resize to input size via bilinear interpolation.

**Output:** `(1, 19, 128, 128)` → argmax → `(512, 512)` class map → resize to original
        """)


# ─────────── TAB 5: BEV + OCCUPANCY ─────────────────────────────────────────
with tabs[5]:
    st.markdown("#### Bird's Eye View + Occupancy Grid")
    bc1,bc2 = st.columns(2)
    with bc1:
        st.image(bgr2pil(bev_lanes), caption="IPM + Lane projection", use_container_width=True)
    with bc2:
        st.image(bgr2pil(occ), caption="2D occupancy grid", use_container_width=True)


# ─────────── TAB 6: ANALYTICS ────────────────────────────────────────────────
with tabs[6]:
    st.markdown("#### DL Pipeline Analytics")
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(1,3,figsize=(14,4),facecolor="#080b12")
    for ax in axes: ax.set_facecolor("#0d1424")

    # Detection class pie
    ax0 = axes[0]
    if dets:
        cc = Counter(d["class"] for d in dets)
        CPIE = ["#0ea5e9","#38bdf8","#7dd3fc","#bae6fd","#e0f2fe","#93c5fd"]
        w,t,at = ax0.pie(cc.values(),labels=cc.keys(),autopct="%1.0f%%",
                         colors=CPIE[:len(cc)],textprops={"color":"#c4d0e0","fontsize":8})
        for at_ in at: at_.set_color("#080b12")
    else:
        ax0.text(0.5,0.5,"No detections",ha="center",va="center",color="#4a6280")
    ax0.set_title("Object classes",color="#00c8ff",fontsize=10)

    # Distance bars
    ax1 = axes[1]
    if dets:
        names = [f"{d['class'][:4]}\n{d['distance']:.0f}m" for d in dets[:8]]
        dd_   = [d["distance"] for d in dets[:8]]
        bcols = ["#ef4444" if d<20 else "#f59e0b" if d<40 else "#22c55e" for d in dd_]
        ax1.barh(names,dd_,color=bcols)
        ax1.set_xlabel("Distance (m)",color="#4a6280")
        ax1.tick_params(colors="#4a6280",labelsize=7)
    ax1.set_title("Object distances",color="#00c8ff",fontsize=10)
    for s in ax1.spines.values(): s.set_edgecolor("#162038")

    # Segmentation coverage
    ax2 = axes[2]
    valid = {k:v for k,v in stats.items() if v["percent"]>1.0}
    xlabs = [v["label"] for v in valid.values()]
    xvals = [v["percent"] for v in valid.values()]
    xcols = [f"#{v['color'][0]:02x}{v['color'][1]:02x}{v['color'][2]:02x}" for v in valid.values()]
    ax2.bar(xlabs,xvals,color=xcols)
    ax2.set_ylabel("Coverage %",color="#4a6280")
    ax2.tick_params(axis='x',rotation=35,labelsize=7,colors="#4a6280")
    ax2.tick_params(axis='y',colors="#4a6280")
    ax2.set_title("Segmentation classes",color="#00c8ff",fontsize=10)
    for s in ax2.spines.values(): s.set_edgecolor("#162038")

    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confidence histogram
    if dets:
        st.markdown("#### YOLOv8 Confidence Distribution")
        confs = [d["score"] for d in dets]
        fig2,ax3 = plt.subplots(figsize=(10,2.5),facecolor="#080b12")
        ax3.set_facecolor("#0d1424")
        ax3.hist(confs,bins=12,range=(0,1),color="#0ea5e9",alpha=0.8,edgecolor="#162038")
        ax3.axvline(conf_thresh,color="#f59e0b",linestyle="--",
                    label=f"Threshold {conf_thresh:.0%}")
        ax3.set_xlabel("Confidence",color="#4a6280")
        ax3.set_ylabel("Count",color="#4a6280")
        ax3.tick_params(colors="#4a6280")
        ax3.legend(facecolor="#080b12",labelcolor="#c4d0e0",fontsize=8)
        for s in ax3.spines.values(): s.set_edgecolor("#162038")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    import onnxruntime as _ort
    _ort_ver = _ort.__version__
    st.markdown(f"<small style='color:#2a3a50'>Pipeline: {elapsed_ms:.1f}ms · "
                f"Frame: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}px · "
                f"ONNX Runtime {_ort_ver}</small>",
                unsafe_allow_html=True)


# ─────────── TAB 7: ARCHITECTURES ────────────────────────────────────────────
with tabs[7]:
    st.markdown("#### Deep Learning Model Architectures")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("##### 🏗️ YOLOv8 Data Flow")
        st.code("""
Input: (1, 3, 640, 640) float32
    ↓ CSPDarknet + C2f blocks
    ↓ Stride 8,16,32 features
    ↓ PAN-FPN neck
    ↓ Decoupled head
Output: (1, 84, 8400)
    ↓ Slice [:4] → box coords (cx,cy,w,h)
    ↓ Slice [4:] → class logits (80 COCO)
    ↓ Sigmoid → class probs
    ↓ Threshold @ conf_thresh
    ↓ Per-class NMS @ IoU 0.45
→ List[{bbox, score, class, distance}]
""", language="text")

        st.markdown("##### 🏗️ MiDaS Data Flow")
        st.code("""
Input: (1, 3, 256, 256) float32 (ImageNet norm)
    ↓ EfficientNet-Lite3 encoder
    ↓ Multi-scale features: /4, /8, /16, /32
    ↓ Reassemble blocks (reshape + project)
    ↓ Fusion blocks (bilinear up + conv)
Output: (1, 256, 256) float32 (inverse depth)
    ↓ Resize to (720, 1280) bilinear
    ↓ Invert (MiDaS: high=close → low=close)
    ↓ Normalise [0,1]
    ↓ Project → [0, 120m]
→ depth_norm (H,W) + metric_depth (H,W)
""", language="text")

    with c2:
        st.markdown("##### 🏗️ SegFormer Data Flow")
        st.code("""
Input: (1, 3, 512, 512) float32 (ImageNet norm)
    ↓ Overlapping patch embed 7×7 s=4 → 128×128
    ↓ MiT Stage 1: SR-attn R=8, 32ch
    ↓ Patch merge 3×3 s=2 → 64×64
    ↓ MiT Stage 2: SR-attn R=4, 64ch
    ↓ Patch merge 3×3 s=2 → 32×32
    ↓ MiT Stage 3: SR-attn R=2, 160ch
    ↓ Patch merge 3×3 s=2 → 16×16
    ↓ MiT Stage 4: SR-attn R=1, 256ch
    ↓ All-MLP decoder: upsample all→32×32
    ↓ Concat + MLP → (1,19,128,128)
    ↓ Softmax → argmax → resize NEAREST
Output: (H,W) uint8 — Cityscapes class IDs
""", language="text")

        st.markdown("##### 🏗️ UFLD (LaneNet) Data Flow")
        st.code("""
Input: (1, 3, 288, 800) float32 (ImageNet norm)
    ↓ ResNet-18/34 backbone
    ↓ Features at 1/4, 1/8, 1/16, 1/32
    ↓ Row-anchor classification head
    ↓ For each of 56 row anchors:
       classify x-pos into 100 columns + 1 "no lane"
Output: (1, 4, 56, 101) — 4 lanes × 56 rows × 101 classes
    ↓ Softmax over column dim
    ↓ Argmax → x-coordinate per row
    ↓ Filter "no lane" class
    ↓ np.polyfit(ys, xs, deg=2) per lane
→ 4 lane polynomials + existence probs
""", language="text")

    st.markdown("---")
    st.markdown("##### Model size comparison")
    import matplotlib.pyplot as plt
    models_info = {
        "YOLOv8n\n(detector)":       {"params_m": 3.2,  "size_mb": 6.2,  "fps_gpu": 1200, "fps_cpu": 35},
        "MiDaS-Small\n(depth)":      {"params_m": 21,   "size_mb": 82,   "fps_gpu": 120,  "fps_cpu": 8},
        "SegFormer-B0\n(segmentor)":  {"params_m": 3.7,  "size_mb": 15,   "fps_gpu": 480,  "fps_cpu": 22},
        "UFLD\n(lanes)":              {"params_m": 11.5, "size_mb": 75,   "fps_gpu": 300,  "fps_cpu": 18},
    }
    fig3, axes3 = plt.subplots(1,2,figsize=(12,3),facecolor="#080b12")
    for ax in axes3: ax.set_facecolor("#0d1424")

    names = list(models_info.keys())
    params = [v["params_m"] for v in models_info.values()]
    fps_gpu= [v["fps_gpu"]  for v in models_info.values()]
    fps_cpu= [v["fps_cpu"]  for v in models_info.values()]
    cols3  = ["#0ea5e9","#38bdf8","#7dd3fc","#bae6fd"]

    axes3[0].bar(names,params,color=cols3); axes3[0].set_ylabel("Parameters (M)",color="#4a6280")
    axes3[0].set_title("Model size (parameters)",color="#00c8ff",fontsize=10)
    axes3[0].tick_params(colors="#4a6280",labelsize=7)

    x = np.arange(len(names)); w2=0.35
    axes3[1].bar(x-w2/2,fps_gpu,w2,label="GPU (RTX3090)",color="#0ea5e9")
    axes3[1].bar(x+w2/2,fps_cpu,w2,label="CPU (i7-12th)",color="#38bdf8",alpha=0.7)
    axes3[1].set_xticks(x); axes3[1].set_xticklabels(names,fontsize=7)
    axes3[1].set_ylabel("FPS",color="#4a6280"); axes3[1].tick_params(colors="#4a6280",labelsize=7)
    axes3[1].set_title("Inference speed",color="#00c8ff",fontsize=10)
    axes3[1].legend(facecolor="#080b12",labelcolor="#c4d0e0",fontsize=8)

    for ax in axes3:
        for s in ax.spines.values(): s.set_edgecolor("#162038")
    plt.tight_layout(); st.pyplot(fig3); plt.close()
