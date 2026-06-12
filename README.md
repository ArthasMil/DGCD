# Relative Depth Guided Cross Domain Semantic Segmentation for High Resolution Satellite Remote Sensing Images

## Introduction

Official codebase of the paper **[Relative Depth Guided Cross Domain Semantic Segmentation for High Resolution Satellite Remote Sensing Images](https://ieeexplore.ieee.org/document/11556424)** by ***Yongqi Sun, Yujun Quan, Anzhu Yu, Yu Su, Xin Li, Chenguang Dai, Xuanguang Liu and Yanfei Zhong***.


## Getting Started

### Step 1. Environment Setup

Follow the installation procedure of [Open-CD](https://github.com/likyoo/open-cd) to configure [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and this codebase.

```bash
# 1) Create conda environment
conda create -n dgcd python=3.8 -y
conda activate dgcd

# 2) Install PyTorch (choose a version compatible with mmcv)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 3) Install OpenMMLab toolkits
pip install -U openmim
mim install mmengine==0.10.4
mim install mmcv==2.1.0
mim install mmpretrain==1.2.0
pip install mmsegmentation==1.2.2
pip install mmdet==3.3.0

# 4) Install DGCD (this repo)
git clone https://github.com/ArthasMil/DGCD.git
cd DGCD
pip install -v -e .

# 5) Install other dependencies
pip install ftfy regex
```

For more details, please refer to [docs/install.md](docs/install.md).

### Step 2. Prepare LoveDA Dataset

Download the officially released [LoveDA](https://github.com/Junjue-Wang/LoveDA) dataset from [Zenodo](https://doi.org/10.5281/zenodo.5706578) or [Baidu Drive](https://github.com/Junjue-Wang/LoveDA) (code: `27vc`). **All experiments in this paper use the official split.**

Following the official LoveDA domain-adaptive semantic segmentation protocol:

- **Rural → Urban (R2U):** train on Rural (`Train` + `Val`), evaluate on Urban (`Test`)
- **Urban → Rural (U2R):** train on Urban (`Train` + `Val`), evaluate on Rural (`Test`)

Organize the data under `Loveda_rural_Train+rural_Val_urban_Test/` (set as `data_root` in configs) as follows:

```
Loveda_rural_Train+rural_Val_urban_Test/
├── train/
│   ├── A/               # RGB optical images (source-domain train set)
│   ├── B/               # relative depth maps (Step 3)
│   └── labels/          # semantic labels
├── test/
│   ├── A/               # RGB optical images (target-domain evaluation set)
│   ├── B/               # relative depth maps (Step 3)
│   └── labels/          # semantic labels
└── test_upload/
    ├── Urban/A          # RGB optical images for Urban test submission
    ├── Urban/B          # relative depth maps for Urban test submission
    ├── Rural/A          # RGB optical images for Rural test submission
    └── Rural/B          # relative depth maps for Rural test submission
```

- `A`: optical RGB images from LoveDA
- `B`: relative depth maps inferred by Depth Anything (Step 3)

Utility scripts in the repo root can help with preprocessing:

- `convert_png.py`: convert `.tif` images to `.png`
- `crop_loveda.py`: crop 1024×1024 images into 512×512 patches

### Step 3. Generate Relative Depth with Depth Anything

Use [Depth Anything](https://github.com/LiheYoung/Depth-Anything) to infer relative depth for **all LoveDA RGB images** (including `Train`, `Val`, and `Test`, both Rural and Urban), then place the outputs into the corresponding `B` folders in Step 2.

```bash
git clone https://github.com/LiheYoung/Depth-Anything.git
cd Depth-Anything
pip install -r requirements.txt

# Example: infer relative depth for one image folder
python run.py --encoder vitl --img-path /path/to/LoveDA/RGB/images --outdir /path/to/depth/output --pred-only --grayscale
```

Recommended settings:

- `--encoder`: `vitl` (default in our experiments), or `vitb` / `vits` for faster inference
- `--pred-only`: save depth maps only
- `--grayscale`: save single-channel relative depth maps

**Important:** keep filenames in `B` identical to the corresponding RGB images in `A`. Run the command above for every RGB folder listed in Step 2.

### Step 4. Train and Inference

Use `myconfigs/U2R/U2R.py` for **Urban → Rural (U2R)** training and inference:

```bash
# Train
python tools/train.py myconfigs/U2R/U2R.py --work-dir=myconfigs/logs/u2r --amp

# Inference (replace the checkpoint path with your trained weights)
python tools/test.py myconfigs/U2R/U2R.py myconfigs/logs/u2r/best_mIoU_iter_XXXX.pth
```

- `--amp`: mixed-precision training
- Checkpoints and logs are saved under `--work-dir`
- Inference results for the Rural test set are exported to `results/` (as configured in `myconfigs/U2R/U2R.py`). Prediction labels are saved in the benchmark format with class ids **`0`–`6`** (consistent with `reduce_zero_label=True` during training).

### Step 5. Online Submission

Submit prediction maps to the [LoveDA Unsupervised Domain Adaptation Challenge](https://www.codabench.org/competitions/13037/) on CodaBench. Please follow the submission format described on the competition page.

**Submission format (LoveDA UDA):**

- Pack all prediction maps into a **`.zip`** archive.
- Each file should be a single-channel **`.png`** grayscale mask.
- **Filename** must match the corresponding test image name (e.g., `4191.png`).
- **Pixel values** use class ids **`0`–`6`** (the benchmark submission format used in this repo):
  - `0`: background
  - `1`: building
  - `2`: road
  - `3`: water
  - `4`: barren
  - `5`: forest
  - `6`: agricultural
- For **U2R**, submit predictions on the **Rural** test set (`test_upload/Rural/`).
- For **R2U**, submit predictions on the **Urban** test set (`test_upload/Urban/`).

> **Note:** The LoveDA organizers updated the evaluation platform in **January 2026**. The current benchmark is hosted at [CodaBench #13037](https://www.codabench.org/competitions/13037/). Results submitted to the earlier platform before this migration **cannot be viewed or downloaded** on the new site.


## Acknowledgement

This codebase is built upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [Open-CD](https://github.com/likyoo/open-cd) (for multi-modal support). We thank the authors for their open-source contributions.

## Citation

If you find this project useful in your research, please cite:

```bibtex
@ARTICLE{11556424,
  author={Sun, Yongqi and Quan, Yujun and Yu, Anzhu and Su, Yu and Li, Xin and Dai, Chenguang and Liu, Xuanguang and Zhong, Yanfei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Relative Depth Guided Cross Domain Semantic Segmentation for High Resolution Satellite Remote Sensing Images},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Remote sensing;Semantic segmentation;Training;Dams;Conferences;Educational institutions;Computers;Computer vision;Buildings;Semantic segmentation;cross domain;relative depth;satellite remote sensing images;high-resolution},
  doi={10.1109/TGRS.2026.3701909}
}
```
