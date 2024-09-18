# 🚶 Completed Interaction Networks for Pedestrian Trajectory Prediction

![CINet](https://via.placeholder.com/1000x300.png) <!-- 你可以用自己的图片链接替换这张图片 -->

This repository contains the official implementation of **CINet**, a novel approach for pedestrian trajectory prediction that considers social interactions in all moments, environmental interactions, and short-term goals in a unified framework. Thanks to my tutor for his great support.

## 📑 Table of Contents
- [Introduction](#introduction)
- [🚀 Features](#features)
- [⚙️ Installation](#installation)
- [📝 Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [🔍 Ablation Studies](#ablation-studies)
- [📊 Results](#results)
- [📚 Citation](#citation)
- [📞 Contact](#contact)

## Introduction

CINet integrates both social and environmental interactions for accurate pedestrian trajectory prediction, outperforming previous methods in various complex scenarios such as sharp turns or obstacle avoidance. The network utilizes a Spatio-Temporal Transformer Layer (STTL) to mine the spatio-temporal information of pedestrian trajectories, and a Gradual Goal Module (GGM) to capture environmental interactions under short-term goals.

**Key contributions of CINet include:**
- 🧑‍🤝‍🧑 Simultaneously modeling social interactions in all moments and environmental interactions.
- 🎯 Utilizing short-term goals to guide predictions.
- 🏆 Outperforming state-of-the-art methods on multiple public pedestrian trajectory datasets.

## 🚀 Features
- 📁 Supports **ETH/UCY**, **Stanford Drone Dataset (SDD)**, and **Intersection Drone Dataset (inD)**.
- ✨ Implements **Spatio-Temporal Transformer Layer (STTL)** and **Gradual Goal Module (GGM)** for advanced trajectory prediction.
- 🔧 Easily configurable for **ablation studies**.

## ⚙️ Installation

To get started, you will need **Python 3.7** and the dependencies listed in `requirement.txt`.

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/cinet-pedestrian-prediction.git
cd cinet-pedestrian-prediction

### ⚙️ Step 2: Install dependencies

To install the necessary dependencies for the project, run the following command in your terminal:

```bash
pip install -r requirement.txt
