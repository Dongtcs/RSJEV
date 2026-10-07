<div align="center">
  <h2><strong>RSJEV: Discriminative Remote Sensing Scene Classification with Multimodal Large Language Models</strong></h2>
  <p>
    <strong>Dongchen Si</strong><sup>1</sup>, 
    <strong>Di Wang</strong><sup>1</sup>, 
    <strong>Mingzhen Xu</strong><sup>1</sup>, 
    <strong>Jing Zhang</strong><sup>1</sup>, 
    <strong>Bo Du</strong><sup>1</sup>, 
    <strong>Liangpei Zhang</strong><sup>1</sup>
  </p>
  <p>
    <sup>1</sup>Wuhan University, Wuhan
  </p>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2610.08539">
    <img src="https://img.shields.io/badge/ArXiv-2610.08539-brown?logo=arxiv" alt="paper">
  </a>
</div>

## 🔥 News
🚀 Code release coming soon!

## 📚Contents

- [📚Contents](#contents)
- [🔍Introduction](#overview)
- [🛠️Methodology](#️methodology)
- [🚀Evaluation](#evaluation)
  - [Inference efficiency](#inference-efficiency)
- [🔗Citation](#citation)

## 🔍Introduction

Remote sensing scene classification is a fundamental task in Earth observation and geospatial analysis. Existing approaches mainly follow three paradigms: task-specific visual classification, vision-language similarity matching, and autoregressive multimodal generation. However, visual classifiers rely on predefined label spaces, CLIP-based methods perform recognition through static image-text alignment, and multimodal large language models (MLLMs) introduce unnecessary token-level generation for classification tasks with explicit candidate categories. To address these limitations, we propose RSJEV, a one-pass multimodal decision framework for remote sensing scene classification. Unlike conventional MLLMs that formulate classification as autoregressive text generation, RSJEV reformulates scene classification as a candidate-conditioned multimodal discriminative decision process, where visual representations, task instructions, and candidate category semantics are jointly modeled. Specifically, we introduce a OnePass Decider that extracts multimodal decision states and directly estimates category probabilities within the candidate category space, eliminating autoregressive decoding while preserving vision-language interactions. Extensive experiments on three widely used remote sensing scene classification benchmarks, including UC Merced, AID, and NWPU-RESISC45, demonstrate that RSJEV achieves superior classification performance compared with representative CNN-, Transformer-, Mamba-, CLIP-, and MLLM-based methods. Moreover, RSJEV significantly reduces inference costs and achieves a better accuracy-efficiency trade-off with only a compact 0.8B-parameter model. These results demonstrate the effectiveness of state-conditioned multimodal decision making for efficient remote sensing image understanding. 

## 🛠️Methodology

![RSJEV architecture: image and task prompt are processed by Qwen3.5-0.8B; OnePass Decider scores candidate categories from the RSSC answer slot.](figure/RSJEV_model.png)

<p align="center"><strong>Figure 1. RSJEV and the OnePass Decider.</strong></p>

1. A remote sensing image and a prompt containing the classification question and candidate categories are processed by the multimodal backbone.
2. OPD reads the final-layer hidden state at the designated `[RSSC]` answer slot.
3. The pretrained language-model head scores the candidate option tokens. A softmax over those scores produces category probabilities and the highest-scoring option is selected.

The experiments use **Qwen3.5-0.8B** as the main backbone.

## 🚀Evaluation

The paper reports the following performance on three remote sensing scene classification benchmarks:

| Dataset | Classes | Training split | Overall accuracy | F1 score |
| :---: | :---: | :---: | :---: | :---: |
| UC Merced Land Use (UCM) | 21 | 50% | **97.71%** | **97.72%** |
| Aerial Image Dataset (AID) | 30 | 20% | **96.56%** | **96.30%** |
| NWPU-RESISC45 (NWPU) | 45 | 20% | **94.59%** | **94.58%** |

These are the RSJEV results reported in Table I of the paper. The remaining images in each split are used for testing.

### Inference efficiency

On one NVIDIA A40 GPU, with batch size 1 and bfloat16 inference, the paper reports **102.18 ms per image**, **9.79 FPS**, and **1.68 GB GPU memory** for RSJEV. Measurements use 30 test images after a five-image warm-up.

<p align="center">
  <img src="figure/efficiency_memory_latency.png" alt="GPU memory and inference latency for RSJEV and six MLLM baselines" width="47%">
  <img src="figure/efficiency_fps.png" alt="Inference throughput in FPS for RSJEV and six MLLM baselines" width="47%">
</p>

<p align="center"><strong>Figure 2. GPU memory and latency (left), Inference throughput (right).</strong></p>

<!-- See the [paper](paper/RSJEV.pdf) for the full methodology, baseline comparisons, and ablation studies. -->

## 🔗Citation

If you use RSJEV in your research, please cite:

```bibtex
@article{si_rsjev,
  title        = {RSJEV: Discriminative Remote Sensing Scene Classification with Multimodal Large Language Models},
  author       = {Si, Dongchen and Wang, Di and Xu, Mingzhen and Zhang, Jing and Du, Bo and Zhang, Liangpei},
  journal      = {arXiv preprint arXiv:2610.08539},
  year         = {2026},
  eprint       = {2610.08539},
  archivePrefix = {arXiv},
  url          = {https://arxiv.org/abs/2610.08539}
}
```
