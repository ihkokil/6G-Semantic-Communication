# Improving Information Quality in 6G Networks Using Semantic Auto-Encoders

This repository explores **Semantic Auto-Encoders** for robust communication over noisy channels with **LDPC coding** and **semantic interference cancellation (SemantIC)**.  

It provides implementations of:
- A semantic encoder–decoder (RED-CNN style) trained on **CIFAR-10**
- A GoogLeNet-inspired classifier to maintain task accuracy after compression
- A SemantIC pipeline that enhances LDPC decoding on **AWGN channels** using semantic extrinsic information

**Author:** Md. Iqbal Haider Khan (@ihkokil)
**LinkedIn:** Md. Iqbal Haider Khan ([@ihkokil](https://www.linkedin.com/in/ihkokil/))
**License:** MIT  

---

## 📑 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Quickstart](#quickstart)
- [Training](#training)
- [SemantIC Testing](#semantic-testing-awgn--ldpc--extrinsic-info)
- [Project Structure](#project-structure)
- [Configuration & Tips](#configuration--tips)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project demonstrates how semantic auto-encoders can improve **information quality in 6G-like networks** by:

- Compressing images into compact latent representations optimized for a **downstream classifier (GoogLeNet)**  
- Transmitting over a **noisy AWGN channel** with LDPC coding  
- Using **semantic interference cancellation (SemantIC)** to inject semantic-level extrinsic information into the LDPC decoder  
- Improving **bit error rate (BER)** and **reconstruction quality** compared to conventional pipelines  

### Core Scripts
- `googlenet_train.py` — Train CIFAR-10 classifier  
- `ENC_DEC_train.py` — Train semantic encoder/decoder with perceptual + task loss  
- `SemantIC.py` — Perform LDPC coding/decoding over AWGN with semantic extrinsic information  

---

## Results

Below: **Original vs Distorted vs SemantIC Reconstruction**  

| Original | Distorted | SemantIC (Reconstructed) |
|:--:|:--:|:--:|
| ![](images/result/original.png) | ![](images/result/distorted.png) | ![](images/result/semantic.png) |

---

## Quickstart

### Tested With
- Python **3.13**
- PyTorch **1.13.0**

### Installation
```bash
# (Recommended) create a new environment
# conda create -n semantic-6g python=3.10 -y && conda activate semantic-6g

pip install -r requirements.txt
````

**requirements.txt** example:

```txt
torch==1.13.0
torchvision==0.14.0
numpy
scipy
numba
pillow
imageio
pandas
```

### Dataset

* CIFAR-10 is automatically downloaded to `./data` on first run.

### GPU Usage

* Scripts use **CUDA if available**.
* You can set a specific GPU via:

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  ```

---

## Training

### 1. Train the Classifier

```bash
python googlenet_train.py
```

* Saves weights as `google_net.pkl` after each epoch.

### 2. Train the Semantic Encoder/Decoder

```bash
python ENC_DEC_train.py --alpha 0.8 --pretrain_epoch 0
```

* `--alpha`: weight for task loss (default **0.8**)
* `--pretrain_epoch`: warmup with MSE-only loss (default **0**)
* `compression_rate`: defined inside script (default **0.3**)

**Outputs:**

* `semantic_coder.pkl`
* Classifier checkpoints (e.g., `google_net_final-lambda-XX.pkl`)
* Sample reconstructions → `images/`
* Training logs (accuracy, PSNR) → `./CIFAR/MLP_sem_CIFAR/*.csv`

---

## SemantIC Testing (AWGN + LDPC + Extrinsic Info)

Run:

```bash
python SemantIC.py
```

**Pipeline:**

1. Load CIFAR-10 & semantic coder (`semantic_coder.pkl`)
2. Quantize latent → binary stream (8-bit)
3. LDPC encode (n=900, d\_v=2, d\_c=3)
4. Transmit via AWGN at SNR ∈ {-5, …, 9} dB
5. Iterative LDPC decoding + semantic extrinsic LLR injection
6. Save outputs & metrics

**Outputs:**

* Per-SNR images: `images/snr{snr}/origin-semantic-{epoch}-{iter}.png`
* Logs: `images/snr{snr}.csv` (BER + Energy Distance metrics)

**Customizable Parameters:**

* Epoch length, batch size → inside `SemantIC.py`
* SNR range → `range(-5, 10)`
* LDPC settings → `n_code, d_v, d_c`

---

## Project Structure

```plaintext
.
├── ENC_DEC_train.py       # Train semantic encoder/decoder
├── googlenet_train.py     # Train CIFAR-10 classifier (GoogLeNet-like)
├── SemantIC.py            # LDPC + AWGN + semantic interference cancellation
├── LDPC/
│   ├── code.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── ldpc_audio.py
│   ├── ldpc_images.py
│   ├── utils.py
│   ├── utils_audio.py
│   └── utils_img.py
├── images/                # Outputs (generated automatically)
│   └── result/            # Example figures for README
│       ├── original.png
│       ├── distorted.png
│       └── semantic.png
├── data/                  # CIFAR-10 (auto-downloaded)
├── requirements.txt
└── readme.md
```

---

## Configuration & Tips

* **Training options (`ENC_DEC_train.py`):**

  * `--alpha`: task loss weight (default 0.8)
  * `--pretrain_epoch`: warmup epochs with MSE loss (default 0)
  * `compression_rate`: inside script (default 0.3)

* **Image preprocessing:**

  * CIFAR-10 resized to **96×96**, normalized to **\[-1, 1]**

* **Checkpoints:**

  * Classifier → `google_net.pkl` / `google_net_final-lambda-XX.pkl`
  * Semantic coder → `semantic_coder.pkl`

* **Outputs:**

  * Training → `images/` (side-by-side reconstructions)
  * Testing → `images/snr{snr}/` (per-SNR results + CSV logs)

* **Adapting to other backbones:**

  * Replace encoder/decoder in `class SemanticNN` (inside `SemantIC.py`)
  * Ensure `enc()` outputs a quantized bitstream & `dec()` reconstructs inputs

---

## Dependencies

* Python **3.10–3.13**
* PyTorch **1.13.0+**
* torchvision
* numpy, scipy, numba
* pillow, imageio
* pandas

Install:

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

This project builds upon the following works and resources:

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (used for training and evaluation)  
- **LDPC Implementation:** adapted from [pyldpc](https://github.com/hichamjanati/pyldpc)  
- **Semantic Training References:**  
  - [SJTU Semantic Communication Systems](https://github.com/SJTU-mxtao/Semantic-Communication-Systems)  
  - [Semantic Communication via Deep Learning (arXiv:2205.00271)](https://arxiv.org/abs/2205.00271)  

This framework can be extended to other semantic backbones by modifying the `SemanticNN` class in `SemantIC.py`.

---

## Citation

If you use this code, please cite:

```bibtex
@software{Khan_Semantic6G_2025,
  author  = {Md. Iqbal Haider Khan},
  title   = {Improving Information Quality in 6G Networks Using Semantic Auto-Encoders},
  year    = {2025},
  url     = {https://github.com/ihkokil/6G-Semantic-Communication},
  note    = {MIT License}
}
```

---

## License

MIT License © Md. Iqbal Haider Khan ([@ihkokil](https://www.linkedin.com/in/ihkokil/))

See [LICENSE](LICENSE) for details.