# Adaptive Structure Consistency for Faithful Unpaired Medical Image Synthesis

Official PyTorch implementation of **"Adaptive structure consistency for faithful unpaired medical image synthesis"** published in IEEE Transactions on Medical Imaging.

> **Authors:** Kevin Estiven Giraldo Paniagua, Pierre-Henri Conze, Vincent Jaouen, Elsa Angelini  
> **Affiliations:** LTCI, Télécom Paris, Institut Polytechnique de Paris | IMT Atlantique, LaTIM UMR 1101, Inserm

[![arXiv](https://img.shields.io/badge/arXiv-2XXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
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

| Dataset | Task | PSNR (dB) | SSIM (%) |
|---------|------|-----------|----------|
| Gold Atlas | MRI T2 → CT | **24.30** | **87.86** |
| SynthRAD 2023 | CBCT → CT | **26.05** | **83.81** | 
| IXI | T1 → T2 | **16.58** | **66.86** | 

---

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

### Basic Training Command

```bash
python train.py --cfg /configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name AdaNGF --model AdaNGF --lambda_edge 10 --netG resnet_9blocks2 --epsilonT 0.000025 --multi_parameter 8.0 --epsilon_multi_scale
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

### Basic Testing Command

```bash
python train.py --cfg /configs/Gold_Atlas/2d_GA.yaml --checkpoints_dir results/ --name AdaNGF --model AdaNGF --lambda_edge 10 --netG resnet_9blocks2 --epsilonT 0.000025 --multi_parameter 8.0 --epsilon_multi_scale --test True
```

### Evaluation Metrics

The framework automatically computes **Image Quality** metrics: PSNR, SSIM, HaarPSI, MS-SSIM. 

Those are store in `--checkpoints_dir` / `--name` / inf / metrics_{`--name`}.csv:

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
@article{giraldo2024adangf,
  title={Adaptive structure consistency for faithful unpaired medical image synthesis},
  author={Giraldo Paniagua, Kevin Estiven and Conze, Pierre-Henri and Jaouen, Vincent and Angelini, Elsa},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  volume={XX},
  number={XX},
  pages={1-15},
  doi={10.1109/TMI.2024.XXXXXXX}
}
```

---

## 🤝 Acknowledgments

This work was partially supported by IMT Future & Ruptures.

This implementation is based on:
- [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu
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

- [NEC: Normalized Edge Consistency](https://github.com/yourusername/NEC)
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [CUT: Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation)
- [CUT: Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation)
- [SynDiff](https://github.com/icon-lab/SynDiff)
- [UNIT](https://github.com/mingyuliutw/UNIT)


---

## 📝 Updates

- **[2026-XX-XX]**: Initial release
- **[2026-XX-XX]**: Paper accepted at IEEE TMI
