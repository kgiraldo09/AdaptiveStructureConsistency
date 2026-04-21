# Adaptive Structure Consistency for Faithful Unpaired Medical Image Synthesis

Official PyTorch implementation of **"Adaptive structure consistency for faithful unpaired medical image synthesis"** published in IEEE Transactions on Medical Imaging.

> **Authors:** Kevin Estiven Giraldo Paniagua, Pierre-Henri Conze, Vincent Jaouen, Elsa Angelini  
> **Affiliations:** LTCI, Télécom Paris, Institut Polytechnique de Paris | IMT Atlantique, LaTIM UMR 1101, Inserm


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

Adaptive Structure Consistency for Faithful Unpaired Medical Image Synthesis introduces a novel framework for unpaired medical image translation that enforces anatomically consistent translations while adapting to modality-specific structure characteristics. Unlike traditional GAN-based approaches that may hallucinate or distort anatomical structures, Adaptive Structure Consistency for Faithful Unpaired Medical Image Synthesis preserves anatomical fidelity through spatially-adaptive structural constraints.


<img title="a title" alt="Alt text" src="/git_images/general_pipeline_GAN_NGFAI_journal.jpg">



### Key Features

- **Adaptive Structure Consistency**: Dynamically modulates source image structural amplitude to match target distribution
- **Normalized Gradient Fields (NGF)**: Modality-invariant edge representation for anatomical boundary preservation
- **State-of-the-art Performance**: Superior results on cross-modality (MRI→CT, CBCT→CT) and intra-modality (T1↔T2) translation tasks
- **Clinical Applications**: Suitable for radiotherapy planning, data harmonization, and artifact reduction

### Results Highlights
#### Adaptive structure consistency with NGF (AdaNGF) 

# Quantitative Results 
## Quality metrics

Mean ± standard deviation for **PSNR**, **SSIM**, and **HAARPSI**.

## IXI — MRI T2 → T1

| Model | PSNR | SSIM | HAARPSI |
|------|------|------|------|
| CUT | 15.30 ± 1.44 | 58.05 ± 4.98 | 43.90 ± 4.86 |
| CycleGAN | **17.08 ± 1.27** | 69.73 ± 6.98 | 48.19 ± 5.55 |
| UNIT | 16.76 ± 1.16 | 70.05 ± 7.24 | 46.06 ± 6.52 |
| SynDiff | 16.74 ± 1.26 | 68.66 ± 7.42 | 46.56 ± 6.36 |
| NGF NoAdapt | 16.61 ± 1.28 | 71.69 ± 7.69 | 49.20 ± 5.47 |
| **AdaNGF (ours)** | 16.99 ± 1.42 | **72.73 ± 7.91** | **49.21 ± 5.62** |


## SynthRAD 2023 — CBCT → CT

| Model | PSNR | SSIM | HAARPSI |
|------|------|------|------|
| CUT | 24.18 ± 1.74 | 81.88 ± 4.18 | 63.37 ± 7.31 |
| CycleGAN | 25.48 ± 1.98 | 83.65 ± 4.17 | 67.41 ± 6.77 |
| UNIT | 25.69 ± 1.69 | **83.91 ± 3.69** | 68.66 ± 5.63 |
| SynDiff | 25.34 ± 1.93 | 83.70 ± 4.45 | 69.11 ± 6.40 |
| NGF NoAdapt | 25.86 ± 2.14 | 82.67 ± 5.31 | 69.14 ± 6.90 |
| **AdaNGF (ours)** | **26.05 ± 2.07** | 83.81 ± 4.43 | **70.29 ± 6.63** |


## Gold Atlas — MR → CT

| Model | PSNR | SSIM | HAARPSI |
|------|------|------|------|
| CUT | 21.42 ± 0.95 | 73.74 ± 3.37 | 47.64 ± 5.06 |
| CycleGAN | 23.55 ± 1.22 | 84.23 ± 2.20 | 59.43 ± 5.23 |
| UNIT | 22.92 ± 0.99 | 83.98 ± 2.23 | 56.09 ± 4.14 |
| SynDiff | 20.20 ± 1.03 | 76.75 ± 2.53 | 46.64 ± 6.79 |
| NGF NoAdapt | 21.40 ± 0.33 | 65.44 ± 2.22 | 51.32 ± 1.86 |
| **AdaNGF (ours)** | **24.30 ± 1.35** | **87.86 ± 1.90** | **64.26 ± 4.95** |


## Anatomical Consistency — Gold Atlas

| Model | Femur DS (%) | Femur HD (mm) | Femur ASD (mm) | Sacrum DS (%) | Sacrum HD (mm) | Sacrum ASD (mm) | Hip DS (%) | Hip HD (mm) | Hip ASD (mm) |
|------|------|------|------|------|------|------|------|------|------|
| MRI* | 84.8 | 5.3 | 1.7 | 68.9 | 6.9 | 1.9 | 79.8 | 3.5 | 1.6 |
| CUT | 90.4 | 3.3 | 1.3 | 75.0 | 4.8 | 1.8 | 84.0 | 3.7 | 1.6 |
| CycleGAN | 93.1 | 2.4 | 1.0 | 80.9 | 3.5 | 1.2 | 90.0 | 2.6 | 1.2 |
| UNIT | 90.5 | 3.4 | 1.4 | 51.0 | 9.0 | 2.0 | 86.2 | 4.3 | 1.4 |
| SynDiff | 73.3 | 8.6 | 2.7 | 58.8 | 7.2 | 2.2 | 70.2 | 6.6 | 2.4 |
| NGF NoAdapt | 85.8 | 8.7 | 4.7 | 42.0 | 18.8 | 3.3 | 71.9 | 24.6 | 4.5 |
| **AdaNGF (ours)** | **94.4** | **2.0** | **0.9** | **86.0** | **2.6** | **0.9** | **92.6** | **2.0** | **0.8** |

## Segmentation Consistency — IXI Dataset
Evaluated structures include thalamus, caudate, putamen, pallidum, hippocampus, amygdala, and accumbens.

| Model | Dice | HD95 (mm) | ASD (mm) |
|------|------|------|------|
| CUT | 0.620 ± 0.128 | 3.16 ± 0.53 | 1.37 ± 0.16 |
| CycleGAN | 0.805 ± 0.064 | 1.91 ± 0.26 | 0.91 ± 0.07 |
| SynDiff | 0.824 ± 0.044 | 1.80 ± 0.17 | 0.84 ± 0.07 |
| UNIT | 0.887 ± 0.042 | 1.14 ± 0.15 | 0.41 ± 0.06 |
| NGF NoAdapt | 0.925 ± 0.021 | 1.09 ± 0.13 | 0.36 ± 0.04 |
| **AdaNGF (ours)** | **0.931 ± 0.021** | **1.03 ± 0.07** | **0.33 ± 0.05** |



## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)

### Dependencies

This implementation is based on the [CycleGAN implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and uses [MONAI](https://monai.io/) for medical image processing.

## Dataset Organization

Organize your dataset in the following structure:

```
datasets/
├── Gold_Atlas/
│   ├── patient_1_CT.nii.gz
│   ├── patient_1_T2.nii.gz
│   ├── patient_n_CY.nii.gz
│   └── patient_n_T2.nii.gz
```
---

## Training

### Basic Training Command 2D and 3D

```bash
python train.py --cfg /configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name AdaNGF --model AdaNGF --lambda_edge 10 --netG resnet_9blocks2 --epsilonT 0.000025 --multi_parameter 8.0 --epsilon_multi_scale
```
```bash
python python train.py --cfg configs/Gold_Atlas/3d_GA.yaml --checkpoints_dir results/ --name AdaNGF_3D --model AdaNGF --lambda_edge 10 --netG resnet_9blocks2 --epsilonT 0.0000025 --multi_parameter 80.0 --epsilon_multi_scale --dim 3
```

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `--cfg` | Path to dataset configuration file, update it with your own paths |
| `--checkpoints_dir` | Path where the output folder containing model and experiment results will be created |
| `--name` | Output file name |
| `--model` | Model that will be used |
| `--lambda_ngf` | Weight for NGF loss |
| `--epsilonT` and `--multi_parameter`| Fixed target domain tolerance parameter (ratio): `--epsilonT` * `--multi_parameter`|
| `--epsilon_multi_scale` | use of multi scale discriminator for edge domain only |

**Do not forget to update your config file!**

---

## Testing

### Basic Testing Command 2D and 3D

```bash
python train.py --cfg configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name AdaNGF --model AdaNGF --lambda_edge 10 --netG resnet_9blocks2 --epsilonT 0.000025 --multi_parameter 8.0 --epsilon_multi_scale --test True
```
```bash
python python train.py --cfg configs/Gold_Atlas/3d_GA.yaml --checkpoints_dir results/ --name AdaNGF_3D --model AdaNGF --lambda_edge 10 --netG resnet_9blocks2 --epsilonT 0.0000025 --multi_parameter 80.0 --epsilon_multi_scale --dim 3 --test True
```

### Evaluation Metrics

The framework automatically computes **Image Quality** metrics: PSNR, SSIM, HaarPSI, MS-SSIM. 

Those are store in `--checkpoints_dir` / `--name` / inf / metrics_{`--name`}.csv:

---

## Other model architectures training:
#### Cycle GAN
```bash
python train.py --cfg configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name CycleGAN --model cycle_gan --lambda_A 10 --lambda_B 10
```
#### CUT
```bash
python train.py --cfg configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name cut --model cut
```

#### UNIT
```bash
python train.py --cfg configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name UNIT --model UNIT --disc True
```
Where `--disc` allow to use the same discriminator architecture as in AdaNGF, CycleGAN or CUT. 

#### NGF
```bash
python train.py --cfg configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name NGF --model NGF --lambda_NGF 10.0 --epsilonT 0.0002
```



---
## Model Architecture

### Generator Architecture
- **Encoder**: ResNet-based with shared weights between image and adaptive tolerance parameter generator
- **Decoder**: Separate decoders for image translation and adaptive tolerance parameter
- **Output**: Translated image + spatially-varying epsilon map

### Discriminator Architecture
- **Modality Discriminator**: PatchGAN discriminator for realistic image generation
- **Structure Discriminator**: Multi-scale discriminator for edge maps.

---

## 📖 Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{giraldo2024adasc,
  title={Adaptive structure consistency for faithful unpaired medical image synthesis},
  author={Giraldo Paniagua, Kevin Estiven and Conze, Pierre-Henri and Jaouen, Vincent and Angelini, Elsa},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  volume={XX},
  number={XX},
  pages={1-15},
  doi={10.1109/TIP.2024.XXXXXXX}
}
@inproceedings{AdaNGF_ISBI,
  TITLE = {Adaptive gradient domain normalization for one-sided unsupervised medical image synthesis},
  AUTHOR = {Giraldo Paniagua, Kevin and Conze, Pierre-Henri and Jaouen, Vincent and Angelini, Elsa},
  BOOKTITLE = {{IEEE International Symposium on Biomedical Imaging}},
  YEAR = {2026},
}
```

---

## 🤝 Acknowledgments

This work was partially supported by IMT Future & Ruptures.

This implementation is based on:
- [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu
- [CUT](https://github.com/taesungp/contrastive-unpaired-translation) by Taesung Park
- [UNIT](https://github.com/mingyuliutw/UNIT) by Ming-Yu Liu
- [MONAI](https://monai.io/) - Medical Open Network for AI

We thank the authors for making their code publicly available.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub

---

## 🔗 Related Projects

- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [CUT: Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation)
- [UNIT](https://github.com/mingyuliutw/UNIT) by Ming-Yu Liu
- [SynDiff](https://github.com/icon-lab/SynDiff)


---

## 📝 Updates
- **[2026-01-13]**: AdaNGF Paper accepted at ISBI 2026
- **[2026-XX-XX]**: Initial release
- **[2026-XX-XX]**: Paper accepted at IEEE TIP
