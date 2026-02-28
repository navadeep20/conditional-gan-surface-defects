# 🛠️ Conditional GAN for Manufacturing Surface Defect Generation

## 📌 Project Overview

This project implements a **Conditional Generative Adversarial Network (cGAN)** to generate synthetic manufacturing surface defect images conditioned on defect type. The system addresses **data scarcity and class imbalance** in industrial visual inspection pipelines.

The generated synthetic defects can be used to **augment training datasets** for automated quality inspection systems in industries such as automotive, steel, electronics, and ceramics.

---

## 🚀 Key Highlights

* Conditional image generation based on defect type
* End-to-end GAN training pipeline
* Synthetic data augmentation workflow
* Downstream classifier validation
* Interactive Streamlit web application
* Modular and production-ready code structure

---

## 🎯 Problem Statement

In real manufacturing environments:

* Rare defect types have very few samples
* Data collection and labeling are expensive
* Class imbalance hurts defect detection models
* Manual inspection is subjective and inconsistent

This project demonstrates how **conditional GANs** can generate realistic synthetic defects to mitigate these challenges.

---

## 🧠 Proposed Solution

We design a conditional GAN system that learns from available defect images and generates new samples conditioned on defect class.

### Pipeline

Raw Images
→ Preprocessing & Label Mapping
→ Conditional GAN Training
→ Synthetic Defect Generation
→ Classifier Training (Baseline vs Augmented)
→ Web Application Deployment

---

## 🏗️ System Architecture

### Generator (G)

**Input:**

* Random noise vector (z ∈ ℝ¹⁰⁰)
* Defect class embedding

**Output:**

* Synthetic defect image (64×64 RGB)

**Architecture:**

* Label embedding + noise concatenation
* Fully connected projection
* DCGAN-style ConvTranspose upsampling
* Tanh output in [-1, 1]

---

### Discriminator (D)

**Input:**

* Image
* Defect class embedding

**Output:**

* Real/Fake probability

**Architecture:**

* Label spatial embedding
* Convolutional downsampling blocks
* LeakyReLU activations
* Sigmoid output

---

## ⚙️ Training Strategy

To stabilize GAN training, the following techniques were applied:

* Label smoothing (real label = 0.9)
* Gaussian noise injection to discriminator input
* Adam optimizer (β₁ = 0.5, β₂ = 0.999)
* Balanced generator/discriminator learning rates
* Conditional label embedding

---

## 📊 Experimental Validation

### Baseline

A ResNet-18 classifier was trained on **real data only**.

### Augmented

A second classifier was trained on **real + GAN synthetic data**.

### Observations

* Synthetic data showed early-stage texture realism
* Augmented training achieved comparable overall accuracy
* Improved recall observed in texture-dominant classes (e.g., rust)
* Fine-grained defects require longer GAN training for best gains

**Conclusion:** Conditional GAN–based augmentation is feasible and beneficial with sufficient training.

---

## 🖥️ Streamlit Web Application

An interactive UI allows non-ML users to generate synthetic defects on demand.

### Features

* Defect type dropdown
* Image count slider
* One-click sample generation
* Image grid visualization
* ZIP download of generated samples

### Run the App

```bash
streamlit run src/app_surface_cgan.py
```

---

## 📂 Project Structure

```
conditional-gan-surface-defects/
│
├── src/
│   ├── generator_cgan_surface.py
│   ├── discriminator_cgan_surface.py
│   ├── train_cgan_surface.py
│   ├── inference_surface_cgan.py
│   ├── app_surface_cgan.py
│   └── ...
│
├── checkpoints/        # (ignored in git)
├── data/               # (ignored in git)
├── samples/            # (ignored in git)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### 1️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Usage

### Train GAN

```bash
python -m src.train_cgan_surface
```

### Generate Samples

```bash
python -m src.inference_surface_cgan
```

### Train Baseline Classifier

```bash
python -m src.defect_classifier_train
```

### Train Augmented Classifier

```bash
python -m src.defect_classifier_train_augmented
```

### Run Web App

```bash
streamlit run src/app_surface_cgan.py
```

---

## 🏭 Industrial Use Cases

* Automotive body panel inspection
* Steel/aluminum surface monitoring
* PCB defect detection
* Ceramic and glass quality control
* Textile surface inspection

---

## 🔮 Future Improvements

* Train GAN for 100+ epochs for sharper defects
* Integrate real industrial datasets
* Implement ACGAN variant
* Add FID / IS quantitative metrics
* Dockerize deployment
* Real-time camera integration

---

## 👨‍💻 Author

**Batch-8 GAN for Images**
B.Tech — Generative AI & Machine Learning
Academic Year: 2025–26

---

## ⭐ If you found this useful

Consider giving the repository a star — it helps visibility and motivates further improvements.
