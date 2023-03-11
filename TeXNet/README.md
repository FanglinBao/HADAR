# TEXNet
This page contains the architecture, implementation details, and the code (in PyTorch) for "TeXNet", a DNN to decompose thermal images into Temperature, emissivity, and Texture maps.

## Contents

1. [Introduction](#introduction)
2. [Implementation Details](#Implementation-Details)
3. [Results](#Results)
4. [Contacts](#contacts)

## Introduction

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/TeXNet/TeXNet.png" />
</p>

Architecture of our TeX-Net for inverse TeX decomposition. TeX-Net is physics-inspired for three aspects. Firstly, TeX decomposition of heat cubes relies on both spatial patterns and spectral thermal signatures. This inspires the adoption of spectral and pyramid (spatial) attention layers in the UNet model. Secondly, due to TeX degeneracy, the mathematical structure has to be specified to ensure the uniqueness of inverse mapping, and hence it is essential to learn thermal lighting factors $V$ instead of texture <em>X</em>. That is, TeX-Net cannot be trained end-to-end. Here, α,β and γ are indices of objects, and ν is the wavenumber. X<sub>α</sub> is constructed with <em>V</em> and S<sub>βν</sub> indirectly, where S<sub>βν</sub> is the down-sampled S<sub>αν</sub> to approximate <em>k</em> most significant environmental objects. Thirdly, the material library <em>M</em> and its dimension are key to the network. TeX-Net can either be trained with ground truth <em>T</em>, <em>m</em>, and <em>V</em> in supervised learning, or alternatively, with material library <em>M</em>, Planck's law B<sub>ν</sub>(T<sub>α</sub>), and the mathematical structure of X<sub>αν</sub> in unsupervised learning. In supervised learning, the loss function is a combination of individual losses with regularization hyper-parameters. In unsupervised learning, the loss function defined on the re-constructed heat cube is based on physics models of the heat signal. In practice, a hybrid loss function with T, e, V contributions (50%) in addition to the physics-based loss (50%) is used.

## Implementation Details

### Dependencies
All the dependencies are included in a .yml file that can be downloaded from the following link:<br />
[OneDrive](https://purdue0-my.sharepoint.com/personal/baof_purdue_edu/_layouts/15/onedrive.aspx?ga=1&noAuthRedirect=1&id=%2Fpersonal%2Fbaof%5Fpurdue%5Fedu%2FDocuments%2FHADAR%2FTeX%2DNet%2Ftexnet%2Eyml&parent=%2Fpersonal%2Fbaof%5Fpurdue%5Fedu%2FDocuments%2FHADAR%2FTeX%2DNet)

```
conda env create -f texnet.yml
```

### Train
The following command can be treated as an example to train TeXNet on the HADAR databse

```
python main.py --ngpus 1 --backbone resnet50 --data_dir ../ --workers 8 --epochs 40000 --checkpoint_dir supervised_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42
```

### Validation
The following command can be treated as an example to validate the trained model

```
python main.py --ngpus 1 --backbone resnet50 --data_dir ../ --workers 8 --epochs 40000 --checkpoint_dir supervised_check_crop --lr 1e-3 --weight-decay 1e-3 --train_T --train_v --no_log_images --eval_every 500 --res full --batch-size 10 --seed 42 --resume (trained model checkpoint) --eval
```

## Results

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/TeXNet/TeXNet_Results.png" />
</p>
