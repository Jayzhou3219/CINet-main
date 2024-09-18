# 🚶 Completed Interaction Networks for Pedestrian Trajectory Prediction

![图片1](https://github.com/user-attachments/assets/74f695cf-7e9f-4f2f-9ca5-41551291af67) <!-- 你可以用自己的图片链接替换这张图片 -->

This repository contains the official implementation of **CINet**, a novel approach for pedestrian trajectory prediction that considers social interactions in all moments, environmental interactions, and short-term goals in a unified framework. Thanks to my tutor for his great support.

## 📑 Table of Contents
- [Introduction](#introduction)
- [🚀 Features](#features)
- [⚙️ Installation](#installation)
- [📝 Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)

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
```bash``` 
git clone https://github.com/Jayzhou3219/CINet-main.git
<br />cd CInet-pedestrian-prediction

### Step 2: Install dependencies

pip install -r requirement.txt

## 📝 Usage

### -Training
To train the CINet model, use the following command:

<br />python main.py --train --dataset <dataset_name>

<br />Replace <dataset_name> with the dataset you want to train on (e.g., ETH, UCY, SDD).

### -Testing
You can evaluate the model using the following command:
 
<br />python main.py --test --dataset <dataset_name> --checkpoint <path_to_model_checkpoint>

<br />Replace <path_to_model_checkpoint> with the path to your saved model.





