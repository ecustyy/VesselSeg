# Direction-Aware Hybrid Architecture for Retinal Vessel Segmentation

This is the official code for the paper **"Direction-Aware Hybrid Architecture for Retinal Vessel Segmentation"**.

The proposed method integrates a Vision Transformer (ViT) encoder with
Transformer Direction Attention (DA_tr), a Direction Mamba bottleneck
with four-direction cross-scan SSM and Direction-Specific Merge (DSM),
and a CNN decoder with Convolution Direction Attention (DA_conv). A
dynamic direction consistency loss supervises training, and a
three-stage direction-aware post-processing algorithm reconnects
fragmented vessels during inference.

## Directory structure

```
.
├── train_cross_scan.py      # main training script (Cross-Scan Mamba + DSM)
├── test.py                  # test/evaluation (DRIVE)
├── test_crossscan.py        # Cross-Scan model test wrapper
├── config_updated.py        # command-line configuration
├── config.py                # legacy config (used by test.py)
├── function.py              # data loading / train / validation helpers
│
├── networks/                # network modules
│   ├── vit_seg_modeling.py            # ViT encoder (R50-ViT-B_16)
│   ├── vit_seg_configs.py             # ViT configuration presets
│   ├── vit_seg_modeling_resnet_skip.py# ResNet stem + skip variants
│   ├── mamba_block.py                 # SSM (S6) basic block
│   ├── cross_scan_mamba.py            # Cross-Scan + S6 + CrossMerge + DSM
│   ├── direction_aware_attention.py   # DA_tr / DA_conv modules
│   └── direction_aware_hybrid.py      # hybrid bottleneck wrapper
│
├── lib/                     # losses, data, post-processing, metrics
│   ├── losses/
│   │   ├── loss.py                      # cross-entropy loss
│   │   ├── direction_loss.py            # direction consistency loss
│   │   └── improved_direction_loss.py   # dynamic-weight combined loss
│   ├── postprocess.py        # direction-aware post-processing (Stage 1-3)
│   ├── dataset.py / datasetV2.py       # dataset classes
│   ├── extract_patches.py    # patch extraction / overlap reconstruction
│   ├── metrics.py            # AUC / F1 / Acc / SE / SP / Precision
│   ├── pre_processing.py     # image normalization
│   ├── visualize.py          # result visualization
│   ├── common.py             # utilities (seed, schedulers, AverageMeter)
│   └── logger.py             # log / tensorboard writers
│
├── models/                  # baseline models (U-Net, LadderNet)
│
└── prepare_dataset/         # data path lists (DRIVE)
    ├── data_path_list/DRIVE/{train,test}.txt
    └── drive.py             # DRIVE preprocessing
```

## Core algorithm components

| Component | File | Description |
|-----------|------|-------------|
| ViT encoder + DA_tr | `networks/vit_seg_modeling.py`, `networks/direction_aware_attention.py` | 12-layer ViT with channel-wise attention gating |
| Direction Mamba bottleneck | `networks/cross_scan_mamba.py`, `networks/mamba_block.py` | four-direction cross-scan SSM + Direction-Specific Merge |
| CNN decoder + DA_conv | `networks/direction_aware_attention.py` | Direction Convolutional layers with oriented kernels |
| Direction consistency loss | `lib/losses/improved_direction_loss.py` | dynamic-weight (0.1→1.0) combined with CE |
| Direction-aware post-processing | `lib/postprocess.py` | connected-component → skeleton → endpoints → direction/width → reconnection |

## Requirements

- Python 3.7+, PyTorch (CUDA), einops, numpy, scipy, scikit-learn, tqdm, opencv-python, PIL

## Training

```bash
python train_cross_scan.py \
    --save ablation_Phase1/ours_full_crossscan \
    --N_epochs 50 --batch_size 4 \
    --use_hybrid --use_direction_attention
```

## Testing

```bash
python test_crossscan.py          # Cross-Scan model (adjust SAVE path inside)
```
