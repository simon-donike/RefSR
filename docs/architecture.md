# Architecture

## PanS2FusionSR pipeline
`model/model.py` defines `PanS2FusionSR`, a dual-branch fusion network:

1. **PAN HR encoder (`PanFeatureEncoder`)** – extracts high-frequency cues from the SPOT 2.5 m panchromatic image via deformable convolution, residual refinement, and spatial attention.
2. **Shared 10 m encoders (`S2Spot10mEncoder`)** – process Sentinel-2 LR inputs and downsampled PAN references into a common feature space enriched with channel attention.
3. **Cross-attention (`CrossAttentionMap`)** – treats Sentinel-2 features as queries and PAN features as keys/values to yield a spatial attention map at the LR grid.
4. **SR head (`SRHead`)** – upsamples the attention map, concatenates it with PAN HR features, and reconstructs the super-resolved Sentinel-2 output.

## Data handling
`data/example_data.py` implements a dummy Lightning `DataModule` that emits random tensors with the correct Sentinel-2/SPOT shapes. Replace it with domain loaders when integrating real scenes.

## Training loop
`train.py` initializes the configuration, data module, `PanS2System`, and PyTorch Lightning trainer. The system wraps the model, computes simple reconstruction losses, and exposes optimizer/scheduler hooks for experimentation.