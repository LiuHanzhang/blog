---
title: test
date: 2020-03-20 21:16:33
tags:
categories: test
---

1. $1_i^{\rm obj}$值为1或0，表示训练样本的网格$i$中是否存在物体。$1_{ij}^{\rm obj}$值为1或0，表示网格$i$的Bounding Box $j$是否对预测“负责”(responsible)，即是否是与Ground Truth IoU最大的那一个.
2. $\hat{C}_i$是预测的置信度。当训练样本的网格$i$不含有物体时，$C_i = 0$；当含有物体时，$C_i$为"负责的"预测框与Ground Truth的IoU.
3. $\hat{p}_i(c)$是预测的每个类别的概率。当$c$是训练样本网格$i$的真实类别时，$p_i(c) = 1$; 对其他$c$，$p_i(c) = 0$.