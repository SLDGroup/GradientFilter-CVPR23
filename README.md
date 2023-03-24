# Efficient On-device Training via Gradient Filtering

This is the official repo for paper `Efficient On-device Training via Gradient Filtering` accepted by CVPR 2023.

[arxiv](https://arxiv.org/abs/2301.00330)

## Abstract
Despite its importance for federated learning, continuous learning and many other applications,
on-device training remains an open problem for EdgeAI.
The problem stems from the large number of operations (*e.g.*, floating point multiplications and additions) and memory consumption required during training by the back-propagation algorithm.
Consequently, in this paper, we propose a new gradient filtering approach which enables on-device CNN model training. More precisely, our approach creates a special structure with fewer unique elements in the gradient map, thus significantly reducing the computational complexity and memory consumption of back propagation during training.
Extensive experiments on image classification and semantic segmentation with multiple CNN models (*e.g.*, MobileNet, DeepLabV3, UPerNet) and devices (*e.g.*, Raspberry Pi and Jetson Nano) demonstrate the effectiveness and wide applicability of our approach. For example, compared to SOTA, we achieve up to 19 $\times$ speedup and 77.1\% memory savings on ImageNet classification with only 0.1\% accuracy loss. Finally, our method is easy to implement and deploy; over 20 $\times$ speedup and 90\% energy savings have been observed compared to highly optimized baselines in MKLDNN and CUDNN on NVIDIA Jetson Nano. Consequently, our approach opens up a new direction of research with a huge potential for on-device training.

## Code will be released soon!
