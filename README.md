# commonroad-trajectory-transformer

# Introduction

**commonroad-geometric** is a Python framework that facilitates deep-learning based research projects in the autonomous driving domain, e.g. related to behavior planning and state representation learning. 

At its core, it provides a standardized interface for heterogeneous graph representations of traffic scenes using the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) framework.


---

# Installation
This repo is based on commonroad-geometric. To test it, install commonroad geometric and put this repo in tutorial folder to match the path.

The installation script [`scripts/create-dev-environment.sh`](scripts/create-dev-environment.sh) installs the commonroad-geometric package and all its dependencies into a conda environment:  

Execute the script inside the directory which you want to use for your development environment.

Note: make sure that the CUDA versions are compatible with your setup.

---
A simple demo of graph representation:
![sumo_sim_1](https://user-images.githubusercontent.com/60959779/179300095-f9c22942-8e8f-4e83-b21e-95b22de86758.gif )

---
