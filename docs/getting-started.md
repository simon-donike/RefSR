# Getting started

## Install dependencies
Use Python 3.9+ with PyTorch, torchvision, and PyTorch Lightning. The repo ships with a lightweight requirements file:

```bash
pip install -r requirements.txt
```

## Configure an experiment
`config/example_config.yaml` captures data, model, loss, and trainer hyperparameters in a single file. Update the paths or channel counts to match your Sentinel-2/SPOT assets.

Key knobs:
- `data` – batch size, number of workers, and dummy tensor shapes for smoke tests.
- `model` – number of input/output channels plus the base feature width.
- `trainer` – epochs, accelerator, logging cadence.

## Run the trainer
Execute the Lightning entry point to launch a quick 2-epoch synthetic run:

```bash
python train.py
```

The script instantiates the OmegaConf config, builds the `ExampleRefSRDataModule`, and wires it to `PanS2FusionSR` along with a simple L1+L2 loss for debugging.