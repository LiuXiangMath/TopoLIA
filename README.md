# TopoLIA

<p align="center">
  <b>TopoLIA: Topology-Inspired Latent Interaction Attention for Drug Discovery</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#notes">Notes</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**TopoLIA** is a topology-inspired deep learning framework for learning protein--ligand interactions from three-dimensional molecular structures. It combines multiscale persistent spectral representations with a new **latent interaction attention (LIA)** mechanism and self-supervised masked pretraining.
Persistent Laplacian spectra provide structural tokens that encode complementary topological and geometric information across multiple spatial scales. These tokens are processed by **latent interaction Transformer (LIT)** layers, in which LIA extends standard attention by learning not only **how much** information is exchanged between molecular components, but also **how** that information is transformed through a compact latent interaction space.
Given a source--target pair, query--key similarity determines the attention weight, while source- and target-dependent projections mediate information transformation through a compact latent interaction space. TopoLIA is pretrained through masked reconstruction of persistent spectral tokens and subsequently transferred to downstream scoring and docking tasks.

---

## Architecture

<p align="center">
  <img src="figures/architecture.png" width="850">
</p>

<p align="center">
  <b>Figure 1.</b> Overview of the TopoLIA framework. Multiscale persistent spectral embeddings are used as structural tokens for masked self-supervised pretraining. The encoder is composed of latent interaction Transformer (LIT) layers equipped with latent interaction attention (LIA), and the pretrained representations are transferred to downstream molecular prediction tasks.
</p>


---

## Installation

### 1. Create a Conda Environment

```bash
conda create -n topolia python=3.10 -y
conda activate topolia
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Usage

TopoLIA experiments consist of three main stages:

1. Persistent spectral feature preparation using `prepare_feature.py`
2. Self-supervised masked pretraining using `pretrain.py`
3. Fine-tuning on downstream molecular tasks using `finetune.py`

### Step 1: Prepare Persistent Spectral Features

Before training or evaluation, generate the persistent spectral representations for the target dataset:

```bash
python prepare_feature.py --dataname DATASET_NAME
```

For example:

```bash
python prepare_feature.py --dataname 2016
```

The generated persistent spectral features serve as structural tokens for TopoLIA.

### Step 2: Pretrain TopoLIA

Run masked self-supervised pretraining:

```bash
python pretrain.py
```

During pretraining, a subset of persistent spectral tokens is masked and reconstructed from the remaining molecular context using an encoder--decoder architecture composed of LIT layers.

Alternatively, a pretrained TopoLIA checkpoint can be downloaded from:

[Pretrained model](https://drive.google.com/file/d/1E0FvYK_swVVMRRPklxGF9rGsrCrw3EDa/view?usp=sharing)

Place the downloaded checkpoint under:

```bash
./checkpoint/
```

### Step 3: Fine-tune on Downstream Tasks

After pretraining, fine-tune TopoLIA on a downstream benchmark:

```bash
python finetune.py --dataname DATASET_NAME
```

For example:

```bash
python finetune.py --dataname 2016
```

The pretrained encoder is transferred to the downstream task, where the learned representations are aggregated and mapped to task-specific predictions through a multilayer perceptron.

---

## Repository Structure

```bash
TopoLIA/
├── checkpoint/              # Pretrained TopoLIA checkpoints
├── configs/                 # Configuration files and hyperparameters
├── data/                    # Dataset files
├── figures/                 # Figures used in the README and paper
├── src/                     # Source code
├── prepare_feature.py       # Persistent spectral feature preparation
├── pretrain.py              # Self-supervised masked pretraining
├── finetune.py              # Downstream fine-tuning
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Notes

- Place pretrained model checkpoints under `./checkpoint/`.
- Place datasets under `./data/`.
- Persistent spectral features must be generated before pretraining, fine-tuning, or evaluation.
- The dataset name specified by `--dataname` should match the corresponding dataset folder or configuration file.
- The latent interaction rank \(r\) can be specified in the model configuration.

---

## Citation

If you find TopoLIA useful in your research, please consider citing our work:

```bibtex
@article{liu2026topolia,
  title   = {TopoLIA: Topology-Inspired Latent Interaction Attention for Drug Discovery},
  author  = {Xiang Liu and Yiming Ren and Mustafa Hajij and Pietro Li{\`o} and Guo-Wei Wei},
  journal = {Submitted},
  year    = {2026}
}
```
