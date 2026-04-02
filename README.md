<div align="center">
  
# Frequency-Modulated Visual Restoration for Matryoshka Large Multimodal Models

[Qingtao Pan](https://qingtaopan.github.io/), [Zhihao Dou](https://scholar.google.com/citations?user=JiBGiB8AAAAJ&hl=zh-CN&oi=ao), and [Shuo Li](https://case.edu/engineering/about/faculty-and-staff-directory/shuo-li)

**CVPR 2026 Findings, 2026**

<a href='https://arxiv.org/abs/2603.11220'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Overview
FMVR disentangles the visual representation of fewer visual tokens into low- and high-frequency components through AvgPool and MaxPool. The high-frequency from AvgPool acts as a saliency filter to enhance saliency visual semantics, while the low-frequency from MaxPool acts as an anti-saliency filter to strengthen weak visual semantics. Additionally, we inject FMVR into Matryoshka Representation Learning to learn coarse-to-fine visual token sets, thus enabling to elastically adjust the number of visual tokens during inference while maintaining comparable performance.
<p align="center">
  <img src="assets/Matry_Architecture.png" width="70%"></a> <br>
</p>
