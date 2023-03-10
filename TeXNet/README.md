# TEXNet
This page contains the architecture, implementation details, and the code (in PyTorch) for "TeXNet", a DNN to decompose thermal images into Temperature, emissivity, and Texture maps.

## Contents

1. [Introduction](#introduction)
2. [Implementation Details](#Implementation-Details)
3. [Results](#Results)
4. [Contacts](#contacts)

## Introduction

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/edit/main/TeXNet/TeXNet.png" alt="Sublime's custom image"/>
</p>

a) Architecture of our TeX-Net for inverse TeX decomposition. TeX-Net is physics-inspired for three aspects. Firstly, TeX decomposition of heat cubes relies on both spatial patterns and spectral thermal signatures. This inspires the adoption of spectral and pyramid (spatial) attention layers in the UNet model. Secondly, due to TeX degeneracy, the mathematical structure, ![equation](https://latex.codecogs.com/svg.image?X_{\alpha\nu}=\sum_\beta&space;V_{\alpha\beta}S_{\beta\nu}), has to be specified to ensure the uniqueness of inverse mapping, and hence it is essential to learn thermal lighting factors $V$ instead of texture <em>X</em>. That is, TeX-Net cannot be trained end-to-end. Here, α,β and γ are indices of objects, and ν is the wavenumber. X<sub>α</sub> is constructed with V and $S_{\beta\nu}$ indirectly, where $S_{\beta\nu}$ is the down-sampled $S_{\alpha\nu}$ to approximate $k$ most significant environmental objects. 
