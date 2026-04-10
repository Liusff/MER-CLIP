# MER-CLIP: AU-Guided Vision-Language Alignment for Micro-Expression Recognition

This is the official implementation of:

> **MER-CLIP: AU-Guided Vision-Language Alignment for Micro-Expression Recognition**
>
> Shifeng Liu, Xinglong Mao, Sirui Zhao*, Peiming Li, Tong Xu, Enhong Chen*
>
> *IEEE Transactions on Affective Computing, 2025*


## Introduction

Micro-expressions (MEs) are fleeting and subtle facial movements revealing genuine emotions, with important applications in criminal investigation and psychological diagnosis. MER-CLIP integrates the CLIP model's powerful cross-modal semantic alignment capability into micro-expression recognition (MER). Specifically, we convert AU (Action Unit) labels into detailed textual descriptions of facial muscle movements, guiding fine-grained spatiotemporal ME learning by aligning visual dynamics with AU-based textual representations. We also introduce an Emotion Reasoning Module to capture the nuanced relationship between ME patterns and emotions. To mitigate overfitting caused by scarce ME data, we propose LocalStaticFaceMix, an effective data augmentation strategy that enhances facial diversity while preserving key ME features.

## Project Structure

```
MER_CLIP_code/
├── projects/merclip/          # Core MER-CLIP implementation
│   ├── models/                   #   Model definitions
│   │   ├── uf2_clip_clshead_dfme.py   # Main model: ActionClip_MA_WithCls_ME
│   │   ├── clip/                 #   CLIP encoders (text & vision)
│   │   └── ...
│   ├── configs/                  #   Experiment configs (DFME/CASME/SAMM)
│   └── utils/                    #   Training utilities
├── data/                         # Dataset annotations (DFME/CASME2/CASME3/SAMM)
├── tools/                        # Training & testing scripts
├── configs/                      # Baseline configs (UniFormerV2 w/o CLIP)
├── mmaction/                     # Modified mmaction2 framework
│   ├── datasets/                 #   Custom ME datasets & transforms
│   ├── models/                   #   Custom backbones, heads, losses, data_preprocessors
│   ├── evaluation/               #   MEMetric (UF1, UAR)
│   └── engine/                   #   WithEpochBasedTrainLoop
├── resources/                    # Pretrained backbone weights
├── requirements_merclip.txt      # Pinned dependency versions
└── setup.py
```

## Environment Setup

### Prerequisites

- Linux (tested on Ubuntu)
- Python 3.8
- NVIDIA GPU with CUDA 11.6
- Conda (Miniconda or Anaconda)

### Installation

```shell
# Step 1: Create conda environment
conda create --name merclip python=3.8 -y
conda activate merclip

# Step 2: Install PyTorch (CUDA 11.6)
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Step 3: Install OpenMMLab dependencies
pip install openmim==0.3.9
mim install mmengine==0.10.4
mim install mmcv==2.0.0

# Step 4: Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Step 5: Install remaining dependencies
pip install ftfy regex pytorchvideo decord timm einops \
    scikit-learn pandas seaborn opencv-python numpy pillow scipy matplotlib

# Step 6: Install this project (editable mode)
cd MER_CLIP_code
pip install -v -e .
```

> For other CUDA versions, refer to [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/). The full pinned dependency versions are available in `requirements_merclip.txt`.

### Quick Verification

```shell
python -c "
import torch, mmcv, mmengine, clip
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'mmcv: {mmcv.__version__}, mmengine: {mmengine.__version__}')
print('CLIP: OK')
import mmaction; print(f'mmaction2: {mmaction.__version__} ({mmaction.__file__})')
"
```

Expected output should show all versions and `mmaction2` pointing to your local `MER_CLIP_code/mmaction/`.

## Pretrained Weights

Place the following pretrained weights under `resources/`:

| File                                  | Description                          |
|---------------------------------------|--------------------------------------|
| `sthv2_uniformerv2_b16_32x224.pyth`  | UniFormerV2 backbone (SthV2 pretrain)|
| `ViCLIP-B_InternVid-FLT-10M.pth`    | ViCLIP text encoder weights          |

## Supported Datasets

| Dataset   | Classes | Validation | Config Example |
|-----------|---------|------------|----------------|
| DFME      | 7       | Hold-out   | `uf2_progres_clip+transformer_clshead_dfme_ccac.py` |
| CAS(ME)²  | 3/5     | LOSO       | `uf2_progres_clip+transformer_aug_clshead_casme2_dfme-pre.py` |
| CAS(ME)³  | 3/4/7   | LOSO       | `uf2_progres_clip+transformer_aug_clshead_casme3.py` |
| SAMM      | 3/5     | LOSO       | `uf2_progres_clip+transformer_aug_clshead_samm_dfme-pre.py` |

### Dataset Preparation

Each dataset should be organized as raw frames under `data_root` specified in the config file. Annotation files are provided under `data/`.

## Training

**For datasets without LOSO** (e.g., DFME):

```shell
cd tools
bash dist_train.sh ../projects/merclip/configs/<CONFIG>.py <NUM_GPUS>
```

**For datasets with LOSO** (e.g., CASME series, SAMM):

```shell
cd tools
bash dist_loso_train.sh ../projects/merclip/configs/<CONFIG>.py <NUM_GPUS> <DATASET>
# <DATASET>: casme2, casme3, samm
```

### Config Examples

```shell
# DFME (CCAC competition), 2 GPUs
bash dist_train.sh ../projects/merclip/configs/uf2_progres_clip+transformer_clshead_dfme_ccac.py 2

# CAS(ME)² with DFME pre-training, 2 GPUs
bash dist_loso_train.sh ../projects/merclip/configs/uf2_progres_clip+transformer_aug_clshead_casme2_dfme-pre.py 2 casme2

# SAMM with DFME pre-training, 2 GPUs
bash dist_loso_train.sh ../projects/merclip/configs/uf2_progres_clip+transformer_aug_clshead_samm_dfme-pre.py 2 samm
```

### Single GPU Training

```shell
cd tools
CUDA_VISIBLE_DEVICES=0 python train.py ../projects/merclip/configs/<CONFIG>.py
```

## Testing

```shell
# Standard test
bash dist_test.sh ../projects/merclip/configs/<CONFIG>.py <NUM_GPUS>

# LOSO test
bash dist_loso_test.sh ../projects/merclip/configs/<CONFIG>.py <NUM_GPUS> <DATASET>
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025merclip,
  title={MER-CLIP: AU-Guided Vision-Language Alignment for Micro-Expression Recognition},
  author={Liu, Shifeng and Mao, Xinglong and Zhao, Sirui and Li, Peiming and Xu, Tong and Chen, Enhong},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  doi={10.1109/TAFFC.2025.3572931}
}
```

## Acknowledgement

This project is built upon [MMAction2](https://github.com/open-mmlab/mmaction2) and [ActionCLIP](https://github.com/sallymmx/ActionCLIP). We use pretrained weights from [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2) and [ViCLIP](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid). We thank the authors for their excellent work.
