# CINet: Completed Interaction Network for Pedestrian Trajectory Prediction

This repository contains the implementation of the CINet model, which is designed for pedestrian trajectory prediction by integrating social interactions, environmental interactions, and short-term goals into a unified framework.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Running the Code](#running-the-code)
- [Ablation Studies](#ablation-studies)
- [Results and Evaluation](#results-and-evaluation)
- [Citation](#citation)
- [License](#license)

## Introduction

CINet is a novel network architecture for pedestrian trajectory prediction. It simultaneously considers:
- Social interactions across all time steps
- Environmental interactions from scene semantic maps
- Short-term goals for precise and reasonable trajectory predictions

The model is evaluated on several benchmark datasets, including ETH/UCY, Stanford Drone Dataset (SDD), and inD, achieving state-of-the-art results.

## Installation

### Requirements

- Python 3.7
- PyTorch 1.13.1 with CUDA 11.7 support

To install the required dependencies, use the following commands:

```bash
# Clone the repository
git clone https://github.com/YourRepo/CINet.git
cd CINet

# Install Python dependencies
pip install -r requirement.txt
