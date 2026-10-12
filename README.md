<p align="center">
  <h2 align="center">Endo3R: Unified Online Reconstruction from Dynamic Monocular Endoscopic Video</h2>
  <h3 align="center">MICCAI 2025 (Oral)</h3>  
  <h3 align="center"><a href="https://arxiv.org/abs/2504.03198">Paper</a> | <a href="https://wrld.github.io/Endo3R/">Project Page</a> </h3>
  <div align="center"></div>
  <p align="center">
    <a href="https://wrld.github.io/"><strong>Jiaxin Guo</strong></a><sup>1</sup>&nbsp;&nbsp;&nbsp;
    <strong>Wenzhen Dong</strong></a><sup>2</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=nhbSplwAAAAJ&hl=en"><strong>Tianyu Huang</strong></a><sup>1,2</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=NIP-G-cAAAAJ&hl=en"><strong>Hao Ding</strong></a><sup>3</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://hk.linkedin.com/in/ziyi-wang-5ba98a167"><strong>Ziyi Wang</strong></a><sup>1</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=ra-BCykAAAAJ&hl=zh-CN"><strong>Haomin Kuang</strong></a><sup>4</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://www.cse.cuhk.edu.hk/~qdou/"><strong>Qi Dou</strong></a><sup>1,2</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://www4.mae.cuhk.edu.hk/peoples/liu-yun-hui/"><strong>Yun-Hui Liu</strong></a><sup>1,2</sup>&nbsp;&nbsp;&nbsp;
    <br />
    <sup>1</sup><strong>The Chinese University of Hong Kong</strong>    <sup>2</sup><strong>Hong Kong Centre For Logistics Robotics</strong><br>
<sup>3</sup><strong>Johns Hopkins University</strong>    <sup>4</sup><strong>Shanghai Jiao Tong University</strong>
  </p>
</p>


The repository contains the official implementation for the paper [Endo3R: Unified Online Reconstruction from Dynamic Monocular Endoscopic Video](https://arxiv.org/abs/2504.03198).

## TODO
- [x] Release model weights, inference and evaluation code
- [ ] Release training code

## Overview
 In this paper, we present <b>Endo3R</b>, a unified 3D surgical foundation model for online scale-consistent reconstruction from monocular endoscopic video <b>without any prior information or extra optimization</b>, predicting globally aligned pointmaps, scale-consistent video depth, camera poses and intrinsics.

![unified reconstruction](./static/images/input.jpg)
The core contribution of our method is expanding the capability of the recent pairwise reconstruction model to long-term incremental dynamic reconstruction by an uncertainty-aware dual memory mechanism. 
![unified reconstruction](./static/images/pipeline.jpg)

## Getting Started
### Installation
1. Clone Endo3R.
``` bash
git clone https://github.com/wrld/Endo3R.git
```
2. Create the environment, following the below command.
``` bash
conda create -n endo3r python=3.11 cmake=3.14.0
conda activate endo3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
```
3. Optional, compile the cuda kernels for RoPE:
``` bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```
### Prepare Datasets
- We train our method on four datasets containing GT/Stereo depth and pose (datasets 1-7 of [SCARED](https://endovissub2019-scared.grand-challenge.org/Home/), [StereoMIS](https://zenodo.org/records/7727692), [C3VD](https://durrlab.github.io/C3VD/), [EndoMapper](https://www.synapse.org/Synapse:syn26707219)), four datasets without GT data ([Cholec80](https://opendatalab.com/OpenDataLab/Cholec80), [AutoLaparo](https://autolaparo.github.io/), [EndoVis17](https://opencas.dkfz.de/endovis/datasetspublications/), [EndoVis18](https://opencas.dkfz.de/endovis/datasetspublications/)).
- We evaluate our method on datasets 8-9 of [SCARED](https://endovissub2019-scared.grand-challenge.org/Home/) and all scenes of [Hamlyn](https://www.kaggle.com/datasets/mcocoz/hamlyn).
### Download Checkpoints
Please download the pretrained models:
``` bash
mkdir checkpoints
cd checkpoints
gdown https://drive.google.com/uc?id=11hbBHEqBWes4oK2e8OeNi2RM-QtzhKE0
```
Also download the DUSt3R checkpoint:
``` bash
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```
### Inference
Please follow the command:
``` bash
python demo.py  --demo_path SEQ_PATH --kf_every IMG_INTERVAL   --save_path SAVE_PATH  --ckpt_path ./checkpoints/endo3r.pth --save_result
# example:
# python demo.py  --demo_path examples/hamlyn_23/ --kf_every 1   --save_path outputs/hamlyn_23/  --ckpt_path ./checkpoints/endo3r.pth
```
To visualize the 3D reconstruction result, please follow:
``` bash
python vis.py --recon_path SAVE_PATH
# example:
# python vis.py --recon_path outputs/hamlyn_23/
```
### Evaluation
To validate our method, please run:
``` bash
# SCARED 
python eval.py --data_root EVAL_DATA_ROOT --data_type scared --ckpt_path ./checkpoints/endo3r.pth --resolution 320

# Hamlyn
python eval.py --data_root EVAL_DATA_ROOT --data_type hamlyn --ckpt_path ./checkpoints/endo3r.pth --resolution 320
```
### Flash/KDA, Sampson and PPO experiment workflow

The new local-to-server pipeline preserves the released legacy path and compares
`legacy_official`, `hybrid_sampson` and `hybrid_ppo` on a verified SCARED manifest.
See [SCARED experiment protocol](docs/SCARED_EXPERIMENT_PROTOCOL.md) for data
contracts, training, metrics, server requirements and reproducible commands.
The [benchmark preview](docs/benchmark_preview/benchmark.md) contains pending
N/A rows and separately identified paper references, not measured results.

```bash
# Read-only command/config preview; no jobs, writes, installations or downloads.
python tools/run_scared_study.py --stage all
python -B -m unittest discover -s tests -v
```

Training entrypoints require explicit `--run-training`; the study executor
requires `--server-run`. SCARED data and existing checkpoints stay on the server.

# Acknowledgement
We would like to thank the authors of [MonST3R](https://monst3r-project.github.io/files/monst3r_paper.pdf), [Spann3R](https://arxiv.org/abs/2408.16061), and [CUT3R](http://arxiv.org/abs/2501.12387) for their excellent work!

# Citation
``` bibtex
@inproceedings{guo2025endo3r,
  title={Endo3R: Unified Online Reconstruction from Dynamic Monocular Endoscopic Video}, 
  author={Jiaxin Guo and Wenzhen Dong and Tianyu Huang and Hao Ding and Ziyi Wang and Haomin Kuang and Qi Dou and Yun-Hui Liu},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025},
  }
```
