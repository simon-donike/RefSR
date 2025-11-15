# RefSR overview

RefSR is a compact PyTorch Lightning sandbox for reference-guided super-resolution that fuses Sentinel-2 (10 m) spectral inputs with high-resolution SPOT panchromatic guidance. The goal is to quickly validate fusion ideas before scaling the model or data pipeline.

## Key concepts
- **Dual-branch encoding** – dedicated encoders extract high-frequency PAN cues and aligned 10 m features for Sentinel-2 and SPOT inputs.
- **Cross-attention fusion** – a transformer-style attention map links LR S2 context with reference texture to steer the SR head.
- **Modular building blocks** – deformable convolutions, residual blocks, and attention modules are factored into `model/blocks.py` for reuse.

## Data flow
1. Sentinel-2 LR query, SPOT PAN HR reference, and an optional 10 m PAN key are prepared by the data module defined in `data/example_data.py`.
2. `PanS2FusionSR` encodes LR/HR features, builds a spatial attention map, upsamples it to the HR grid, and fuses it with PAN features.
3. The SR head reconstructs super-resolved Sentinel-2 predictions that match the PAN grid resolution.

Use the navigation on the left to jump to setup instructions, architectural details, or the API reference.