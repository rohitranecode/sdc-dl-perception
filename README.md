# 🧠 Self-Driving Car — Deep Learning Perception System 

A production-grade, modular deep learning perception pipeline for autonomous vehicles.

## Models

| Module | Model | Architecture | Dataset | Params | ONNX size |
|--------|-------|--------------|---------|--------|-----------|
| Lane detection | UFLD | ResNet-18 + row-anchor cls | TuSimple | 11.5M | 75 MB |
| Object detection | YOLOv8n | CSPDarknet + C2f + DFL head | COCO 80-class | 3.2M | 6.2 MB |
| Depth estimation | MiDaS v2.1 Small | EfficientNet-Lite3 encoder | 10 datasets mix | 21M | 82 MB |
| Segmentation | SegFormer-B0 | Mix Transformer + All-MLP decoder | Cityscapes 19-class | 3.7M | 15 MB |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download pretrained ONNX weights
python download_models.py

# 3. Run the app
streamlit run app.py
```

The app runs **immediately in simulation mode** if weights are not downloaded.  
With weights, it switches to real ONNX Runtime inference automatically.

## Project Structure

```
sdc_dl/
├── app.py                          # Streamlit dashboard (8 tabs)
├── download_models.py              # Weight downloader
├── requirements.txt
│
├── architectures/                  # Model definitions & pre/post-processing
│   ├── yolov8_arch.py             # YOLOv8: DFL decode, anchor-free NMS
│   ├── midas_arch.py              # MiDaS: scale-shift norm, depth rendering
│   ├── segformer_arch.py          # SegFormer: Cityscapes 19-class, softmax
│   └── lane_net.py                # UFLD: row-anchor decode, polynomial fit
│
├── inference/
│   └── model_runner.py            # ONNX Runtime wrappers (auto-fallback)
│
├── utils/
│   ├── scene_generator.py         # Synthetic road scene generator
│   └── visualization.py           # HUD, BEV, occupancy, grid
│
└── weights/                        # ONNX model files (download separately)
    ├── yolov8n.onnx
    ├── midas_v21_small_256.onnx
    ├── segformer_b0_cityscapes.onnx
    └── ufld_tusimple.onnx
```

## Inference Pipeline

```
RGB Frame (1280×720)
    │
    ├──▶ LaneDetector ──────────────────────── preprocess (288×800) 
    │    UFLD ResNet-18                         row-anchor classification
    │    ↓                                      polynomial fit (deg=2)
    │    Lane polynomials + metrics
    │
    ├──▶ YOLOv8Detector ────────────────────── letterbox (640×640)
    │    CSPDarknet + PAN-FPN + DFL head        sigmoid scores
    │    ↓                                      per-class NMS @ IoU 0.45
    │    Detections [{class, bbox, score, dist}]
    │
    ├──▶ MiDaSDepthEstimator ───────────────── resize (256×256)
    │    EfficientNet-Lite3 encoder             ImageNet normalise
    │    ↓                                      invert + scale-shift norm
    │    depth_norm (H,W) + metric_depth (H,W)
    │
    ├──▶ SegFormerSegmentor ────────────────── resize (512×512)
    │    MiT-B0 encoder + All-MLP decoder       softmax + argmax
    │    ↓                                      resize NEAREST to orig
    │    seg_map (H,W) uint8 — 19 Cityscapes classes
    │
    └──▶ BEV + Occupancy ───────────────────── perspective warp (homography)
         IPM + lane projection + distance grid
         ↓
         Occupancy map + BEV lanes

All outputs → HUD Overlay → Streamlit Dashboard
```

## Key Technical Details

### YOLOv8 — Anchor-Free Design
- No anchor boxes: predicts `(cx, cy, w, h)` directly at each grid point
- DFL (Distribution Focal Loss): bbox regression as learned distribution over 16 bins
- Output: `(1, 84, 8400)` — 84 = 4 coords + 80 COCO classes, 8400 anchor points
- Filtered to 8 SDC-relevant classes: car, truck, bus, person, motorcycle, bicycle, traffic light, stop sign

### MiDaS — Scale-Shift Invariant Depth
- Trained on 10 diverse datasets with mixed metric/relative depth labels
- Loss: `||d̂ - (s·d_gt + t)||₁` — handles metric ambiguity per-image
- Output is **inverse depth** (higher value = closer object) — inverted in post-processing

### SegFormer — Efficient Transformer Segmentation
- Sequence reduction attention: reduces K/V length by R² before attention  
- No positional embeddings → better generalisation to different resolutions
- All-MLP decoder: 4 scale features → upsample → concat → 2 MLP layers

### UFLD — 300× Faster Lane Detection
- Treats lane detection as classification, not segmentation
- For each of 56 row anchors, classify x-position into 100 column bins
- ~22K predictions vs ~230K pixel labels → dramatically faster

## Simulation Mode

When ONNX weights are absent, each module falls back to a DL-quality simulator:
- YOLOv8Simulator: Beta(6,2) confidence distribution, NMS, geometric class assignment
- MiDaSSimulator: Multi-scale feature simulation matching MiDaS output statistics
- SegFormerSimulator: Cityscapes-class HSV + geometry (matches real model class distribution)
- UFLDSimulator: Polynomial lane generation with curvature noise

## Adding Your Own Model

Each model class follows the same interface:

```python
class MyDetector:
    def detect(self, frame_bgr, **kwargs):
        # preprocess → ONNX inference → postprocess
        return {"annotated": frame_with_boxes, "detections": [...]}

class MyDepthModel:
    def predict(self, frame_bgr, **kwargs):
        # preprocess → inference → postprocess  
        return {"depth_norm": H×W float32, "metric_depth": H×W float32, ...}
```

## Licence

All model architectures are open-source:
- YOLOv8: AGPL-3.0 (Ultralytics)
- MiDaS: MIT (Intel ISL)
- SegFormer: Apache-2.0 (NVIDIA)
- UFLD: MIT
