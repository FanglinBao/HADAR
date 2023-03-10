# HADAR
This is an LWIR stereo-hyperspectral database to develop HADAR algorithms for thermal navigation. Based on this database, one can develop algorithms for TeX decomposition to generate TeX vision. One can also develop algorithms about object detection, semantic or scene segmentation, optical or scene flow, stereo depth etc. based on TeX vision instead of traditional RGB or thermal vision.

## Contents

1. [Introduction](#introduction)
2. [Database](#Database)
3. [Semi-Global Decomposition and TeX Vision Matlab Code](#Semi-Global-Decomposition-and-TeX-Vision-Matlab-Code)
4. [Contacts](#contacts)

## Introduction

Machine perception uses advanced sensors to collect information of the surrounding scene for situational awareness. State-of-the-art machine perception  utilizing active sonar, radar and LiDAR to enhance camera vision faces difficulties when the number of intelligent agents scales up. Exploiting omnipresent heat signal could be a new frontier for scalable perception. However, objects and their environment constantly emit and scatter thermal radiation leading to textureless images famously known as the ‘ghosting effect’. Thermal vision thus has no specificity limited by information loss while thermal ranging, crucial for navigation, has been elusive even when combined with artificial intelligence (AI). Here we propose and experimentally demonstrate heat-assisted detection and ranging (HADAR) overcoming this open challenge of ghosting and benchmark it against AI-enhanced thermal sensing. HADAR not only sees texture and depth through the darkness as if it were day, but also perceives decluttered physical attributes beyond RGB or thermal vision, paving the way to fully-passive and physics-aware machine perception. We develop HADAR estimation theory and address its photonic shot-noise limits depicting information-theoretical bounds to HADAR-based AI performance. HADAR ranging at night beats thermal ranging and shows an accuracy comparable with RGB stereovision in daylight. Our automated HADAR thermography reaches the Cram ́er-Rao bound on temperature accuracy, beating existing thermography techniques. Our work leads to a disruptive technology that can accelerate the Fourth Industrial Revolution (Industry 4.0) with HADAR-based autonomous navigation and human-robot social interactions.

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/Fig1.png" alt="Sublime's custom image"/>
  Fig.1 HADAR as a paradigm shift in machine perception.
</p>

<p align="center">
  <img src="https://github.com/FanglinBao/HADAR/blob/main/Fig2.png" alt="Sublime's custom image"/>
</p>
Fig.2 HADAR ranging at night beats thermal ranging and matches RGB stereovision in daylight.

## Database

- The dataset consists of 10 scenes that are simulated. The thermal images contained within each scene have a spatial resolution of 1080 x 1920 and a spectral resolution of 54. These 54 spectral channels are within the heat spectrum 715 - 1250 cm<sup>-1</sup>. 
- Scene 11 contains real-world experimental data.
- The dataset and the video demonstration of TeX vision can be downloaded from the following links:

| HADAR Dataset |  Real-world and Numeric TeX videos|
|---|---|
|[OneDrive](https://purdue0-my.sharepoint.com/personal/baof_purdue_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fbaof%5Fpurdue%5Fedu%2FDocuments%2FHADAR%2FHADAR%20database)|[OneDrive](https://purdue0-my.sharepoint.com/personal/baof_purdue_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fbaof%5Fpurdue%5Fedu%2FDocuments%2FHADAR%2FReal%2Dworld%20and%20numeric%20TeX%20vision%20video%20demonstrations%20at%20night)|[OneDrive]

## Semi-Global Decomposition and TeX Vision Matlab Code
[OneDrive](https://purdue0-my.sharepoint.com/personal/baof_purdue_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fbaof%5Fpurdue%5Fedu%2FDocuments%2FHADAR%2FSGD%5Fand%5FTeX%5Fvision%5Fmatlab%5Fcode%5Fpackage)


## Contacts

