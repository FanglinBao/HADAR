# HADAR
This is an LWIR stereo-hyperspectral database to develop HADAR algorithms for thermal navigation. Based on this database, one can develop algorithms for TeX decomposition to generate TeX vision. One can also develop algorithms about object detection, semantic or scene segmentation, optical or scene flow, stereo depth etc. based on TeX vision instead of traditional RGB or thermal vision.

## Contents

1. [Introduction](#introduction)
2. [HADAR Database](#HADAR-Database)
3. [TeX Code Package](#TeX-Code-Package)
4. [TeX-Net](#TeX-Net)
5. [Contacts](#contacts)

## Introduction

Machine perception uses advanced sensors to collect information of the surrounding scene for situational awareness. State-of-the-art machine perception  utilizing active sonar, radar and LiDAR to enhance camera vision faces difficulties when the number of intelligent agents scales up. Exploiting omnipresent heat signal could be a new frontier for scalable perception. However, objects and their environment constantly emit and scatter thermal radiation leading to textureless images famously known as the ‘ghosting effect’. Thermal vision thus has no specificity limited by information loss while thermal ranging, crucial for navigation, has been elusive even when combined with artificial intelligence (AI). Here we propose and experimentally demonstrate heat-assisted detection and ranging (HADAR) overcoming this open challenge of ghosting and benchmark it against AI-enhanced thermal sensing. HADAR not only sees texture and depth through the darkness as if it were day, but also perceives decluttered physical attributes beyond RGB or thermal vision, paving the way to fully-passive and physics-aware machine perception (see Fig.1). We develop HADAR estimation theory and address its photonic shot-noise limits depicting information-theoretical bounds to HADAR-based AI performance. HADAR ranging at night beats thermal ranging and shows an accuracy comparable with RGB stereovision in daylight (see Fig.2). Our automated HADAR thermography reaches the Cramer-Rao bound on temperature accuracy, beating existing thermography techniques. Our work leads to a disruptive technology that can accelerate the Fourth Industrial Revolution (Industry 4.0) with HADAR-based autonomous navigation and human-robot social interactions.

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/Fig1.png" alt="Sublime's custom image"/><br />
  <em>Fig.1 HADAR as a paradigm shift in machine perception.</em>
</p>

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/Fig2.png" alt="Sublime's custom image"/><br />
  <em>Fig.2 HADAR ranging at night beats thermal ranging and matches RGB stereovision in daylight.</em>
</p>

## HADAR Database

- The current HADAR database consists of 11 datasets, with scenes ranging from Crowded street, Highway, to Suburb, Countryside, to Indoor, and to Forest and Desert, covering most common road conditions that HADAR may find applications in. The first 10 scenes are synthetic datasets mimicing self driving situations, with two HADAR sensors, left (L) and right (R), either mounted at the positions of headlights, or on the top of the automated vehicles, or on robot helpers. Each scene has two views from two stereo sensors. Each view has 5 frames. Each frame is a Height-Width-Channel = 1080-1920-54 heat cube. The 11th dataset is a real-world experimental scene with only one view and 4 frames. Each frame is a Height-Width-Channel = 260-1500-49 heat cube.
- The database and the video demonstration of TeX vision can be downloaded from the following links:

| HADAR Database |  Real-world and Numeric TeX videos| Overall description of the database |
|---|---|---|
|[OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/sjape_purdue_edu/Enhca6JZUlNOm5BEIDnj7N0B_jNzZB0eE7ha_fCJtkDPgA?e=GIBcFL)|[OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/sjape_purdue_edu/EgOSZhDdEe5Bo-fSPP7b4X4BII5v5X0iJfB6TJoegddrJA?e=gnMmi4)|[OneDrive](https://purdue0-my.sharepoint.com/:t:/g/personal/sjape_purdue_edu/EV1KOivhzqJMpd4v7dIQkA0B8Z8ciFN2iUQCkbKhc1lrgg?e=hlQg0u)|

- The dataset-12 with non-uniform temperature can be downloaded from the following link:<br />
[OneDrive](https://purdue0-my.sharepoint.com/:u:/g/personal/baof_purdue_edu/EQezQml4xfBIuD8ItRGV1qIBWGaYezu_M2YJu9msXDWbow)


## TeX Code Package

- The TeX code package contains the matlab codes for TeX-SGD (Semi-Global-Decomposition) and TeX vision visualization.
- mainTeX.m gives line-by-line sample commands with comments to use the TeX code package. The code package can be downloaded from the following link:<br />
[OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/sjape_purdue_edu/EikkN6bcJJNLoAP-ks_DqVQBIzIrE_LnN8bGEjtlKz6jlA?e=UOAlvE)

## TeX-Net

Please see the folder TeXNet.

## Contacts

Dr. Fanglin Bao (baof[at]purdue[dot]edu)
