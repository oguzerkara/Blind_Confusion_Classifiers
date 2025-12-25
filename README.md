# Blind Confusion of Classification Networks

This repository provides the implementation used to generate the experimental results reported in the paper 
“Blind Confusion of Classification Networks”. The work evaluates the robustness of image classification models 
under common and structured image corruptions in a black-box setting.

# Abstract 
  
Real world image classifiers frequently operate under unknown corruptions that
degrade both accuracy and confidence in unpredictable ways. (1) This study evaluates
the robustness of 37 neural network models for image classification under diverse
corruptions through black box attacks. The tested models include conventional and
modern CNNs such as AlexNet ResNet and Inception. They also include noise robust
variants such as Noisy Student and AugMix ResNet and vision transformers such as
ViT, DeiT and Swin. (2) Fifteen corruption types are applied on the ImageNet
ILSVRC2012 validation set and cover common corruptions such as Gaussian Speckle
and Salt and Pepper, as well as structured corruptions including Random Lines
Random Crosses and Confusion Blocks. (3) To distinguish accuracy degradation from
shifts in model confidence, this work complements the Corruption Error metric CE with
the proposed Accuracy Confidence Divergence ACD which summarizes the directional
gap between accuracy and predicted confidence across corruption severities. Our
results show that structured perturbations in particular Random Lines with CE above
0.63 Random Crosses with CE above 0.59 and Salt and Pepper noise with CE above
0.62 degrade performance more severely than common corruptions whose lower CE
bound lies roughly between 0.51 and 0.32. These findings highlight distinct
vulnerabilities in modern architectures and demonstrate the importance of extended
robustn

<img width="1024" height="1536" alt="BlindConfusionofNeuralNetworks_abstract" src="https://github.com/user-attachments/assets/d1f38963-3a45-4a41-bf0a-2277b2aa0c1e" />


