# CVPR2020-Code

CVPR 2020 Paper Open source project

- [CNN](#CNN)
- [Image Classification](#Image-Classification)
- [Object Detection](#Object-Detection)
- [3D Object Detection](#3D-Object-Detection)
- [Video Object Detection](#Video-Object-Detection)
- [Object Tracking](#Object-Tracking)
- [Semantic Segmentation](#Semantic-Segmentation)
- [Instance Segmentation](#Instance-Segmentation)
- [Panoptic Segmentation](#Panoptic-Segmentation)
- [Video Object Segmentation](#VOS)
- [Superpixel Segmentation](#Superpixel)
- [NAS](#NAS)
- [GAN](#GAN)
- [Re-ID](#Re-ID)
- [3D Point cloud (classification/segmentation/registration, etc.)](#3D-PointCloud)
- [Face (recognition/detection/reconstruction, etc.)](#Face)
- [Human Pose Estimation(2D/3D)](#Human-Pose-Estimation)
- [Human Parsing](#Human-Parsing)
- [Scene Text Detection](#Scene-Text-Detection)
- [Scene Text Recognition](#Scene-Text-Recognition)
- [Super Resolution](#Super-Resolution)
- [Model Compression](#Model-Compression)
- [Action Recognition](#Action-Recognition)
- [Crowd Counting](#Crowd-Counting)
- [Depth Estimation](#Depth-Estimation)
- [6D Target pose estimation](#6DOF)
- [Hand Pose](#Hand-Pose)
- [Saliency Object Detection](#Saliency)
- [Denoising](#Denoising)
- [Deblurring](#Deblurring)
- [Dehazing](#Dehazing)
- [Feature point detection and description](#Feature)
- [Visual Q&A](#VQA)
- [VideoQA](#VideoQA)
- [Visual language navigation](#VLN)
- [Video compression](#Video-Compression)
- [Video interpolation](#Video-Frame-Interpolation)
- [Style Transfer](#Style-Transfer)
- [Lane Detection](#Lane-Detection)
- [Human-Object Interaction](#HOI)
- [Human Trajectory Prediction](#HTP)
- [Motion Prediction](#Motion-Predication)
- [Virtual Try-On](#Virtual-Try-On)
- [HDR](#HDR)
- [Adversarial](#AE)
- [Semantic Scene Completion](#SSC)
- [Dataset](#Datasets)
- [Others](#Others)
- [Not Sure](#Not-Sure)

<a name="CNN"></a>

# CNN

**Exploring Self-attention for Image Recognition**

- Paper：https://hszhao.github.io/papers/cvpr20_san.pdf

- Code：https://github.com/hszhao/SAN

**Improving Convolutional Networks with Self-Calibrated Convolutions**

- Homepage：https://mmcheng.net/scconv/

- Paper：http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
- Code：https://github.com/backseason/SCNet

**Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets**

- Paper：https://arxiv.org/abs/2003.13549
- Code：https://github.com/zeiss-microscopy/BSConv

<a name="Image-Classification"></a>

# Image Classification

**Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion**

- Paper：https://arxiv.org/abs/2003.04490

- Code：https://github.com/AdamKortylewski/CompositionalNets

**Spatially Attentive Output Layer for Image Classification**

- Paper：https://arxiv.org/abs/2004.07570 

- Code（Seems to be deleted by the original author）：https://github.com/ildoonet/spatially-attentive-output-layer 

<a name="Object-Detection"></a>

# Object Detection

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**

- Paper：https://arxiv.org/abs/2005.09973

- Code with Dataset：https://github.com/Anymake/DRN_CVPR2020

**Scale-Equalizing Pyramid Convolution for Object Detection**

Paper：https://arxiv.org/abs/2005.03101

Code：https://github.com/jshilong/SEPC

**Revisiting the Sibling Head in Object Detector**

- Paper：https://arxiv.org/abs/2003.07540

- Code：https://github.com/Sense-X/TSD 

**Detection in Crowded Scenes: One Proposal, Multiple Predictions**

- Paper：https://arxiv.org/abs/2003.09163
- Code：https://github.com/megvii-model/CrowdDetection

**Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection**

- Paper：https://arxiv.org/abs/2004.04725
- Code：https://github.com/NVlabs/wetectron

**Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection**

- Paper：https://arxiv.org/abs/1912.02424 
- Code：https://github.com/sfzhang15/ATSS

**BiDet: An Efficient Binarized Object Detector**

- Paper：https://arxiv.org/abs/2003.03961 
- Code：https://github.com/ZiweiWangTHU/BiDet

**Harmonizing Transferability and Discriminability for Adapting Object Detectors**

- Paper：https://arxiv.org/abs/2003.06297
- Code：https://github.com/chaoqichen/HTCN

**CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection**

- Paper：https://arxiv.org/abs/2003.09119
- Code：https://github.com/KiveeDong/CentripetalNet

**Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection**

- Paper：https://arxiv.org/abs/2003.11818
- Code：https://github.com/ggjy/HitDet.pytorch

**EfficientDet: Scalable and Efficient Object Detection**

- Paper：https://arxiv.org/abs/1911.09070
- Code：https://github.com/google/automl/tree/master/efficientdet 

<a name="3D-Object-Detection"></a>

# 3D Object Detection

**Train in Germany, Test in The USA: Making 3D Object Detectors Generalize**

- Paper：https://arxiv.org/abs/2005.08139

- Code：https://github.com/cxy1997/3D_adapt_auto_driving

**MLCVNet: Multi-Level Context VoteNet for 3D Object Detection**

- Paper：https://arxiv.org/abs/2004.05679
- Code：https://github.com/NUAAXQ/MLCVNet

**3DSSD: Point-based 3D Single Stage Object Detector**

- CVPR 2020 Oral

- Paper：https://arxiv.org/abs/2002.10187

- Code：https://github.com/tomztyang/3DSSD

**Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation**

- Paper：https://arxiv.org/abs/2004.03572

- Code：https://github.com/zju3dv/disprcn

**End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection**

- Paper：https://arxiv.org/abs/2004.03080

- Code：https://github.com/mileyan/pseudo-LiDAR_e2e

**DSGN: Deep Stereo Geometry Network for 3D Object Detection**

- Paper：https://arxiv.org/abs/2001.03398
- Code：https://github.com/chenyilun95/DSGN

**LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention**

- Paper：https://arxiv.org/abs/2004.01389
- Code：https://github.com/yinjunbo/3DVID

**PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection**

- Paper：https://arxiv.org/abs/1912.13192

- Code：https://github.com/sshaoshuai/PV-RCNN

**Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud**

- Paper：https://arxiv.org/abs/2003.01251 
- Code：https://github.com/WeijingShi/Point-GNN 

<a name="Video-Object-Detection"></a>

# Video Object Detection

**Memory Enhanced Global-Local Aggregation for Video Object Detection**

Paper：https://arxiv.org/abs/2003.12063

Code：https://github.com/Scalsol/mega.pytorch

<a name="Object-Tracking"></a>

# Object Tracking

**D3S -- A Discriminative Single Shot Segmentation Tracker**

- Paper：https://arxiv.org/abs/1911.08862
- Code：https://github.com/alanlukezic/d3s

**ROAM: Recurrently Optimizing Tracking Model**

- Paper：https://arxiv.org/abs/1907.12006

- Code：https://github.com/skyoung/ROAM

**Siam R-CNN: Visual Tracking by Re-Detection**

- Homepage：https://www.vision.rwth-aachen.de/page/siamrcnn
- Paper：https://arxiv.org/abs/1911.12836
- Paper2：https://www.vision.rwth-aachen.de/media/papers/192/siamrcnn.pdf
- Code：https://github.com/VisualComputingInstitute/SiamR-CNN

**Cooling-Shrinking Attack: Blinding the Tracker with Imperceptible Noises**

- Paper：https://arxiv.org/abs/2003.09595 
- Code：https://github.com/MasterBin-IIAU/CSA 

**High-Performance Long-Term Tracking with Meta-Updater**

- Paper：https://arxiv.org/abs/2004.00305

- Code：https://github.com/Daikenan/LTMU

**AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization**

- Paper：https://arxiv.org/abs/2003.12949

- Code：https://github.com/vision4robotics/AutoTrack

**Probabilistic Regression for Visual Tracking**

- Paper：https://arxiv.org/abs/2003.12565
- Code：https://github.com/visionml/pytracking

**MAST: A Memory-Augmented Self-supervised Tracker**

- Paper：https://arxiv.org/abs/2002.07793
- Code：https://github.com/zlai0/MAST

**Siamese Box Adaptive Network for Visual Tracking**

- Paper：https://arxiv.org/abs/2003.06761

- Code：https://github.com/hqucv/siamban

<a name="Semantic-Segmentation"></a>

# Semantic Segmentation

**Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation**

- Paper：No

- Code：https://github.com/JianqiangWan/Super-BPD

**Single-Stage Semantic Segmentation from Image Labels**

- Paper：https://arxiv.org/abs/2005.08104

- Code：https://github.com/visinf/1-stage-wseg

**Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation**

- Paper：https://arxiv.org/abs/2003.00867
- Code：https://github.com/MyeongJin-Kim/Learning-Texture-Invariant-Representation

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**

- Paper：http://vladlen.info/papers/MSeg.pdf
- Code：https://github.com/mseg-dataset/mseg-api

**CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement**

- Paper：https://arxiv.org/abs/2005.02551
- Code：https://github.com/hkchengrex/CascadePSP

**Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision**

- Oral
- Paper：https://arxiv.org/abs/2004.07703
- Code：https://github.com/feipan664/IntraDA

**Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**

- Paper：https://arxiv.org/abs/2004.04581
- Code：https://github.com/YudeWang/SEAM

**Temporally Distributed Networks for Fast Video Segmentation**

- Paper：https://arxiv.org/abs/2004.01800

- Code：https://github.com/feinanshan/TDNet

**Context Prior for Scene Segmentation**

- Paper：https://arxiv.org/abs/2004.01547

- Code：https://git.io/ContextPrior

**Strip Pooling: Rethinking Spatial Pooling for Scene Parsing**

- Paper：https://arxiv.org/abs/2003.13328

- Code：https://github.com/Andrew-Qibin/SPNet

**Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**

- Paper：https://arxiv.org/abs/2003.05128
- Code：https://github.com/shachoi/HANet

**Learning Dynamic Routing for Semantic Segmentation**

- Paper：https://arxiv.org/abs/2003.10401

- Code：https://github.com/yanwei-li/DynamicRouting

<a name="Instance-Segmentation"></a>

# Instance Segmentation

**PolarMask: Single Shot Instance Segmentation with Polar Representation**

- Paper：https://arxiv.org/abs/1909.13226 
- Code：https://github.com/xieenze/PolarMask 
- Blog：https://zhuanlan.zhihu.com/p/84890413 

**CenterMask : Real-Time Anchor-Free Instance Segmentation**

- Paper：https://arxiv.org/abs/1911.06667 
- Code：https://github.com/youngwanLEE/CenterMask 

**BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation**

- Paper：https://arxiv.org/abs/2001.00309
- Code：https://github.com/aim-uofa/AdelaiDet

**Deep Snake for Real-Time Instance Segmentation**

- Paper：https://arxiv.org/abs/2001.01629
- Code：https://github.com/zju3dv/snake

**Mask Encoding for Single Shot Instance Segmentation**

- Paper：https://arxiv.org/abs/2003.11712

- Code：https://github.com/aim-uofa/AdelaiDet

<a name="Panoptic-Segmentation"></a>

# Panoptic Segmentation

**Pixel Consensus Voting for Panoptic Segmentation**

- Paper：https://arxiv.org/abs/2004.01849
- Code：还未公布

**BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation**

Paper：https://arxiv.org/abs/2003.14031

Code：https://github.com/Mooonside/BANet

<a name="VOS"></a>

# Video Objection Segmentation

**A Transductive Approach for Video Object Segmentation**

- Paper：https://arxiv.org/abs/2004.07193

- Code：https://github.com/microsoft/transductive-vos.pytorch

**State-Aware Tracker for Real-Time Video Object Segmentation**

- Paper：https://arxiv.org/abs/2003.00482

- Code：https://github.com/MegviiDetection/video_analyst

**Learning Fast and Robust Target Models for Video Object Segmentation**

- Paper：https://arxiv.org/abs/2003.00908 
- Code：https://github.com/andr345/frtm-vos

**Learning Video Object Segmentation from Unlabeled Videos**

- Paper：https://arxiv.org/abs/2003.05020
- Code：https://github.com/carrierlxk/MuG

<a name="Superpixel"></a>

# Superpixel

**Superpixel Segmentation with Fully Convolutional Networks**

- Paper：https://arxiv.org/abs/2003.12929
- Code：https://github.com/fuy34/superpixel_fcn

<a name="NAS"></a>

# NAS

**AOWS: Adaptive and optimal network width search with latency constraints**

- Paper：https://arxiv.org/abs/2005.10481
- Code：https://github.com/bermanmaxim/AOWS

**Densely Connected Search Space for More Flexible Neural Architecture Search**

- Paper：https://arxiv.org/abs/1906.09607

- Code：https://github.com/JaminFong/DenseNAS

**MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning**

- Paper：https://arxiv.org/abs/2003.14058

- Code：https://github.com/bhpfelix/MTLNAS

**FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions**

- Paper：https://arxiv.org/abs/2004.05565

- Code：https://github.com/facebookresearch/mobile-vision

**Neural Architecture Search for Lightweight Non-Local Networks**

- Paper：https://arxiv.org/abs/2004.01961
- Code：https://github.com/LiYingwei/AutoNL

**Rethinking Performance Estimation in Neural Architecture Search**

- Paper：https://arxiv.org/abs/2005.09917
- Code：https://github.com/zhengxiawu/rethinking_performance_estimation_in_NAS
- Blog1：https://www.zhihu.com/question/372070853/answer/1035234510
- Blog2：https://zhuanlan.zhihu.com/p/111167409

**CARS: Continuous Evolution for Efficient Neural Architecture Search**

- Paper：https://arxiv.org/abs/1909.04977 
- Code（Open source soon）：https://github.com/huawei-noah/CARS 

<a name="GAN"></a>

# GAN

**Semantically Mutil-modal Image Synthesis**

- Homepage：http://seanseattle.github.io/SMIS
- Paper：https://arxiv.org/abs/2003.12697
- Code：https://github.com/Seanseattle/SMIS

**Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping**

- Paper：https://yiranran.github.io/files/CVPR2020_Unpaired%20Portrait%20Drawing%20Generation%20via%20Asymmetric%20Cycle%20Mapping.pdf
- Code：https://github.com/yiranran/Unpaired-Portrait-Drawing

**Learning to Cartoonize Using White-box Cartoon Representations**

- Paper：https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf

- Homepage：https://systemerrorwang.github.io/White-box-Cartoonization/
- Code：https://github.com/SystemErrorWang/White-box-Cartoonization
- Blog：https://zhuanlan.zhihu.com/p/117422157
- Demo video：https://www.bilibili.com/video/av56708333

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- Paper：https://arxiv.org/abs/2003.08936

- Code：https://github.com/mit-han-lab/gan-compression

**Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions**

- Paper：https://arxiv.org/abs/2003.01826 
- Code：https://github.com/cc-hpc-itwm/UpConv 

<a name="Re-ID"></a>

# Re-ID

**COCAS: A Large-Scale Clothes Changing Person Dataset for Re-identification**

- Paper：https://arxiv.org/abs/2005.07862

- Dataset：No

**Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking**

- Paper：https://arxiv.org/abs/2004.04199

- Code：https://github.com/whj363636/Adversarial-attack-on-Person-ReID-With-Deep-Mis-Ranking

**Pose-guided Visible Part Matching for Occluded Person ReID**

- Paper：https://arxiv.org/abs/2004.00230
- Code：https://github.com/hh23333/PVPM

**Weakly supervised discriminative feature learning with state information for person identification**

- Paper：https://arxiv.org/abs/2002.11939 
- Code：https://github.com/KovenYu/state-information 

<a name="3D-PointCloud"></a>

# 3D Point cloud (classification/segmentation/registration, etc.)

## 3D Point cloud convolution

**Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds**

- Paper：https://arxiv.org/abs/2003.12971

- Code：https://github.com/raoyongming/PointGLR

**Grid-GCN for Fast and Scalable Point Cloud Learning**

- Paper：https://arxiv.org/abs/1912.02984

- Code：https://github.com/Xharlie/Grid-GCN

**FPConv: Learning Local Flattening for Point Convolution**

- Paper：https://arxiv.org/abs/2002.10701
- Code：https://github.com/lyqun/FPConv

## 3D Point cloud classification

**PointAugment: an Auto-Augmentation Framework for Point Cloud Classification**

- Paper：https://arxiv.org/abs/2002.10876 
- Code（Open source soon）： https://github.com/liruihui/PointAugment/ 

## 3D Point cloud semantic segmentation

**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**

- Paper：https://arxiv.org/abs/1911.11236
- Code：https://github.com/QingyongHu/RandLA-Net

- Blog：https://zhuanlan.zhihu.com/p/105433460

**Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels**

- Paper：https://arxiv.org/abs/2004.0409

- Code：https://github.com/alex-xun-xu/WeakSupPointCloudSeg

**PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation**

- Paper：https://arxiv.org/abs/2003.14032
- Code：https://github.com/edwardzhou130/PolarSeg

**Learning to Segment 3D Point Clouds in 2D Image Space**

- Paper：https://arxiv.org/abs/2003.05593

- Code：https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space

## 3D Point cloud instance segmentation

PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

- Paper：https://arxiv.org/abs/2004.01658
- Code：https://github.com/Jia-Research-Lab/PointGroup

## 3D Point cloud registration

**D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features**

- Paper：https://arxiv.org/abs/2003.03164
- Code：https://github.com/XuyangBai/D3Feat

**RPM-Net: Robust Point Matching using Learned Features**

- Paper：https://arxiv.org/abs/2003.13479
- Code：https://github.com/yewzijian/RPMNet 

## 3D Point Cloud Completion

**Cascaded Refinement Network for Point Cloud Completion**

- Paper：https://arxiv.org/abs/2004.03327
- Code：https://github.com/xiaogangw/cascaded-point-completion

## 3D Point cloud target tracking

**P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds**

- Paper：https://arxiv.org/abs/2005.13888
- Code：https://github.com/HaozheQi/P2B

<a name="Face"></a>

# Human face

## Face recognition

**CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition**

- Paper：https://arxiv.org/abs/2004.00288

- Code：https://github.com/HuangYG123/CurricularFace

**Learning Meta Face Recognition in Unseen Domains**

- Paper：https://arxiv.org/abs/2003.07733
- Code：https://github.com/cleardusk/MFR
- Blog：https://mp.weixin.qq.com/s/YZoEnjpnlvb90qSI3xdJqQ 

## Face Detection

## Face live detection

**Searching Central Difference Convolutional Networks for Face Anti-Spoofing**

- Paper：https://arxiv.org/abs/2003.04092

- Code：https://github.com/ZitongYu/CDCN

## Facial expression recognition

**Suppressing Uncertainties for Large-Scale Facial Expression Recognition**

- Paper：https://arxiv.org/abs/2002.10392 

- Code（Open source soon）：https://github.com/kaiwang960112/Self-Cure-Network 

## Face Rotation

**Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images**

- Paper：https://arxiv.org/abs/2003.08124
- Code：https://github.com/Hangz-nju-cuhk/Rotate-and-Render

## Face 3D reconstruction

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**

- Paper：https://arxiv.org/abs/2003.13845
- Dataset：https://github.com/lattas/AvatarMe

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**

- Paper：https://arxiv.org/abs/2003.13989
- Code：https://github.com/zhuhao-nju/facescape

<a name="Human-Pose-Estimation"></a>

# Human pose estimation (2D/3D)

## 2D Human pose estimation

**HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation**

- Paper：https://arxiv.org/abs/1908.10357
- Code：https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

**The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation**

- Paper：https://arxiv.org/abs/1911.07524 
- Code：https://github.com/HuangJunJie2017/UDP-Pose
- Blog：https://zhuanlan.zhihu.com/p/92525039

**Distribution-Aware Coordinate Representation for Human Pose Estimation**

- Homepage：https://ilovepose.github.io/coco/ 

- Paper：https://arxiv.org/abs/1910.06278 

- Code：https://github.com/ilovepose/DarkPose 

## 3D Human pose estimation

**Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: A Geometric Approach**

- Homepage：https://www.zhe-zhang.com/cvpr2020
- Paper：https://arxiv.org/abs/2003.11163

- Code：https://github.com/CHUNYUWANG/imu-human-pose-pytorch

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**

- Paper：https://arxiv.org/abs/2004.01166

- Code：https://github.com/Healthcare-Robotics/bodies-at-rest
- Dataset：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**Self-Supervised 3D Human Pose Estimation via Part Guided Novel Image Synthesis**

- Homepage：http://val.cds.iisc.ac.in/pgp-human/
- Paper：https://arxiv.org/abs/2004.04400

**Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation**

- Paper：https://arxiv.org/abs/2004.00329
- Code：https://github.com/fabbrimatteo/LoCO

**VIBE: Video Inference for Human Body Pose and Shape Estimation**

- Paper：https://arxiv.org/abs/1912.05656 
- Code：https://github.com/mkocabas/VIBE

**Back to the Future: Joint Aware Temporal Deep Learning 3D Human Pose Estimation**

- Paper：https://arxiv.org/abs/2002.11251 
- Code：https://github.com/vnmr/JointVideoPose3D

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- Paper：https://arxiv.org/abs/2003.03972
- Dataset：No

<a name="Human-Parsing"></a>

# Human body analysis

**Correlating Edge, Pose with Parsing**

- Paper：https://arxiv.org/abs/2005.01431

- Code：https://github.com/ziwei-zh/CorrPM

<a name="Scene-Text-Detection"></a>

# Scene text detection

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- Paper：https://arxiv.org/abs/2003.10608
- Code with Dataset：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- Paper：https://arxiv.org/abs/2002.10200 
- Code（Open source soon）：https://github.com/Yuliang-Liu/bezier_curve_text_spotting
- Code（Open source soon）：https://github.com/aim-uofa/adet

**Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection**

- Paper：https://arxiv.org/abs/2003.07493

- Code：https://github.com/GXYM/DRRG

<a name="Scene-Text-Recognition"></a>

# Scene text recognition

**SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition**

- Paper：https://arxiv.org/abs/2005.10977
- Code：https://github.com/Pay20Y/SEED

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- Paper：https://arxiv.org/abs/2003.10608
- Code with Dataset：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- Paper：https://arxiv.org/abs/2002.10200 
- Code（Open source soon）：https://github.com/aim-uofa/adet

**Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition**

- Paper：https://arxiv.org/abs/2003.06606

- Code：https://github.com/Canjie-Luo/Text-Image-Augmentation

<a name="Super-Resolution"></a>

# Super Resolution

## Image Super Resolution

**Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining**

- Paper：https://arxiv.org/abs/2006.01424
- Code：https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention

**Structure-Preserving Super Resolution with Gradient Guidance**

- Paper：https://arxiv.org/abs/2003.13081

- Code：https://github.com/Maclory/SPSR

**Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy**

Paper：https://arxiv.org/abs/2004.00448

Code：https://github.com/clovaai/cutblur

## Video Super Resolution

**Space-Time-Aware Multi-Resolution Video Enhancement**

- Homepage：https://alterzero.github.io/projects/STAR.html
- Paper：http://arxiv.org/abs/2003.13170
- Code：https://github.com/alterzero/STARnet

**Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution**

- Paper：https://arxiv.org/abs/2002.11616 
- Code：https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020 

<a name="Model-Compression"></a>

# Model compression/pruning

**DMCP: Differentiable Markov Channel Pruning for Neural Networks**

- Paper：https://arxiv.org/abs/2005.03354
- Code：https://github.com/zx55/dmcp

**Forward and Backward Information Retention for Accurate Binary Neural Networks**

- Paper：https://arxiv.org/abs/1909.10788

- Code：https://github.com/htqin/IR-Net

**Towards Efficient Model Compression via Learned Global Ranking**

- Paper：https://arxiv.org/abs/1904.12368
- Code：https://github.com/cmu-enyac/LeGR

**HRank: Filter Pruning using High-Rank Feature Map**

- Paper：http://arxiv.org/abs/2002.10179
- Code：https://github.com/lmbxmu/HRank 

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- Paper：https://arxiv.org/abs/2003.08936

- Code：https://github.com/mit-han-lab/gan-compression

**Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression**

- Paper：https://arxiv.org/abs/2003.08935

- Code：https://github.com/ofsoundof/group_sparsity

<a name="Action-Recognition"></a>

# Video understanding/behavior recognition

**Intra- and Inter-Action Understanding via Temporal Action Parsing**

- Paper：https://arxiv.org/abs/2005.10229
- Homepage with Dataset：https://sdolivia.github.io/TAPOS/

**3DV: 3D Dynamic Voxel for Action Recognition in Depth Video**

- Paper：https://arxiv.org/abs/2005.05501
- Code：https://github.com/3huo/3DV-Action

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**

- Homepage：https://sdolivia.github.io/FineGym/
- Paper：https://arxiv.org/abs/2004.06704

**TEA: Temporal Excitation and Aggregation for Action Recognition**

- Paper：https://arxiv.org/abs/2004.01398

- Code：https://github.com/Phoenix1327/tea-action-recognition

**X3D: Expanding Architectures for Efficient Video Recognition**

- Paper：https://arxiv.org/abs/2004.04730

- Code：https://github.com/facebookresearch/SlowFast

**Temporal Pyramid Network for Action Recognition**

- Homepage：https://decisionforce.github.io/TPN

- Paper：https://arxiv.org/abs/2004.03548 
- Code：https://github.com/decisionforce/TPN 

## Skeleton-based motion recognition

**Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition**

- Paper：https://arxiv.org/abs/2003.14111
- Code：https://github.com/kenziyuliu/ms-g3d

<a name="Crowd-Counting"></a>

# Crowd counting

<a name="Depth-Estimation"></a>

# Depth estimation

**Focus on defocus: bridging the synthetic to real domain gap for depth estimation**

- Paper：https://arxiv.org/abs/2005.09623
- Code：https://github.com/dvl-tum/defocus-net

**Bi3D: Stereo Depth Estimation via Binary Classifications**

- Paper：https://arxiv.org/abs/2005.07274

- Code：https://github.com/NVlabs/Bi3D

**AANet: Adaptive Aggregation Network for Efficient Stereo Matching**

- Paper：https://arxiv.org/abs/2004.09548
- Code：https://github.com/haofeixu/aanet

**Towards Better Generalization: Joint Depth-Pose Learning without PoseNet**

- Paper：https://github.com/B1ueber2y/TrianFlow

- Code：https://github.com/B1ueber2y/TrianFlow

## Monocular depth estimation

**On the uncertainty of self-supervised monocular depth estimation**

- Paper：https://arxiv.org/abs/2005.06209
- Code：https://github.com/mattpoggi/mono-uncertainty

**3D Packing for Self-Supervised Monocular Depth Estimation**

- Paper：https://arxiv.org/abs/1905.02693
- Code：https://github.com/TRI-ML/packnet-sfm
- Demo video：https://www.bilibili.com/video/av70562892/

**Domain Decluttering: Simplifying Images to Mitigate Synthetic-Real Domain Shift and Improve Depth Estimation**

- Paper：https://arxiv.org/abs/2002.12114
- Code：https://github.com/yzhao520/ARC

<a name="6DOF"></a>

# 6D Target pose estimation

**MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion**

- Paper：https://arxiv.org/abs/2004.04336
- Code：https://github.com/wkentaro/morefusion

**EPOS: Estimating 6D Pose of Objects with Symmetries**

Homepage：http://cmp.felk.cvut.cz/epos

Paper：https://arxiv.org/abs/2004.00605

**G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features**

- Paper：https://arxiv.org/abs/2003.11089

- Code：https://github.com/DC1991/G2L_Net

<a name="Hand-Pose"></a>

# Gesture estimation

**HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation**

- Paper：https://arxiv.org/abs/2004.00060

- Homepage：http://vision.sice.indiana.edu/projects/hopenet

**Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data**

- Paper：https://arxiv.org/abs/2003.09572

- Code：https://github.com/CalciferZh/minimal-hand

<a name="Saliency"></a>

# Salient Object Detection

**JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection**

- Paper：https://arxiv.org/abs/2004.08515

- Code：https://github.com/kerenfu/JLDCF/

**UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders**

- Homepage：http://dpfan.net/d3netbenchmark/

- Paper：https://arxiv.org/abs/2004.05763
- Code：https://github.com/JingZhang617/UCNet

<a name="Denoising"></a>

# Denoising

**A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising**

- Paper：https://arxiv.org/abs/2003.12751

- Code：https://github.com/Vandermode/NoiseModel

**CycleISP: Real Image Restoration via Improved Data Synthesis**

- Paper：https://arxiv.org/abs/2003.07761

- Code：https://github.com/swz30/CycleISP

<a name="Deraining"></a>

# Deraining

**Multi-Scale Progressive Fusion Network for Single Image Deraining**

- Paper：https://arxiv.org/abs/2003.10985

- Code：https://github.com/kuihua/MSPFN

<a name="Deblurring"></a>

# Deblurring

## Video Deblurring

**Cascaded Deep Video Deblurring Using Temporal Sharpness Prior**

- Homepage：https://csbhr.github.io/projects/cdvd-tsp/index.html 
- Paper：https://arxiv.org/abs/2004.02501 
- Code：https://github.com/csbhr/CDVD-TSP

<a name="Dehazing"></a>

# Dehazing

**Multi-Scale Boosted Dehazing Network with Dense Feature Fusion**

- Paper：https://arxiv.org/abs/2004.13388

- Code：https://github.com/BookerDeWitt/MSBDN-DFF

<a name="Feature"></a>

# Feature point detection and description

**ASLFeat: Learning Local Features of Accurate Shape and Localization**

- Paper：https://arxiv.org/abs/2003.10071

- Code：https://github.com/lzx551402/aslfeat

<a name="VQA"></a>

# Visual Q&A (VQA)

**VC R-CNN：Visual Commonsense R-CNN** 

- Paper：https://arxiv.org/abs/2002.12204
- Code：https://github.com/Wangt-CN/VC-R-CNN

<a name="VideoQA"></a>

# VideoQA

**Hierarchical Conditional Relation Networks for Video Question Answering**

- Paper：https://arxiv.org/abs/2002.10698
- Code：https://github.com/thaolmk54/hcrn-videoqa

<a name="VLN"></a>

# Visual language navigation

**Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training**

- Paper：https://arxiv.org/abs/2002.10638
- Code（Open source soon）：https://github.com/weituo12321/PREVALENT

<a name="Video-Compression"></a>

# Video compression

**Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement**

- Paper：https://arxiv.org/abs/2003.01966 
- Code：https://github.com/RenYang-home/HLVC

<a name="Video-Frame-Interpolation"></a>

# Video interpolation

**Space-Time-Aware Multi-Resolution Video Enhancement**

- Homepage：https://alterzero.github.io/projects/STAR.html
- Paper：http://arxiv.org/abs/2003.13170
- Code：https://github.com/alterzero/STARnet

**Scene-Adaptive Video Frame Interpolation via Meta-Learning**

- Paper：https://arxiv.org/abs/2004.00779
- Code：https://github.com/myungsub/meta-interpolation

**Softmax Splatting for Video Frame Interpolation**

- Homepage：http://sniklaus.com/papers/softsplat
- Paper：https://arxiv.org/abs/2003.05534
- Code：https://github.com/sniklaus/softmax-splatting

<a name="Style-Transfer"></a>

# Style Transfer

**Diversified Arbitrary Style Transfer via Deep Feature Perturbation**

- Paper：https://arxiv.org/abs/1909.08223
- Code：https://github.com/EndyWon/Deep-Feature-Perturbation

**Collaborative Distillation for Ultra-Resolution Universal Style Transfer**

- Paper：https://arxiv.org/abs/2003.08436

- Code：https://github.com/mingsun-tse/collaborative-distillation

<a name="Lane-Detection"></a>

# Lane Detection

**Inter-Region Affinity Distillation for Road Marking Segmentation**

- Paper：https://arxiv.org/abs/2004.05304
- Code：https://github.com/cardwing/Codes-for-IntRA-KD

<a name="HOI"></a>

# Human-Object Interaction

**Detailed 2D-3D Joint Representation for Human-Object Interaction**

- Paper：https://arxiv.org/abs/2004.08154

- Code：https://github.com/DirtyHarryLYL/DJ-RN

**Cascaded Human-Object Interaction Recognition**

- Paper：https://arxiv.org/abs/2003.04262

- Code：https://github.com/tfzhou/C-HOI

**VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions**

- Paper：https://arxiv.org/abs/2003.05541
- Code：https://github.com/ASMIftekhar/VSGNet

<a name="HTP"></a>

# Human Trajectory Prediction

**Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction**

- Paper：https://arxiv.org/abs/2002.11927 
- Code：https://github.com/abduallahmohamed/Social-STGCNN 

<a name="Motion-Predication"></a>

# Motion Prediction

**Collaborative Motion Prediction via Neural Motion Message Passing**

- Paper：https://arxiv.org/abs/2003.06594
- Code：https://github.com/PhyllisH/NMMP

**MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps**

- Paper：https://arxiv.org/abs/2003.06754

- Code：https://github.com/pxiangwu/MotionNet

<a name="Virtual-Try-On"></a>

# Virtual Try-On

**Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content**

- Paper：https://arxiv.org/abs/2003.05863
- Code：https://github.com/switchablenorms/DeepFashion_Try_On

<a name="HDR"></a>

# HDR

**Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline**

- Homepage：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR

- Paper：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR_/00942.pdf

- Code：https://github.com/alex04072000/SingleHDR

<a name="AE"></a>

# Adversarial

**Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance**

- Paper：https://arxiv.org/abs/1911.02466
- Code：https://github.com/ZhengyuZhao/PerC-Adversarial 

<a name="SSC"></a>

# Semantic Scene Completion

**3D Sketch-aware Semantic Scene Completion via Semi-supervised Structure Prior**

- Paper：https://arxiv.org/abs/2003.14052
- Code：https://github.com/charlesCXK/3D-SketchAware-SSC 

<a name="Datasets"></a>

# Dataset

**Intra- and Inter-Action Understanding via Temporal Action Parsing**

- Paper：https://arxiv.org/abs/2005.10229
- Homepage with Dataset：https://sdolivia.github.io/TAPOS/

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**

- Paper：https://arxiv.org/abs/2005.09973

- Code with Dataset：https://github.com/Anymake/DRN_CVPR2020

**COCAS: A Large-Scale Clothes Changing Person Dataset for Re-identification**

- Paper：https://arxiv.org/abs/2005.07862

- Dataset：No

**KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations**

- Paper：https://arxiv.org/abs/2002.12687

- Dataset：https://github.com/qq456cvb/KeypointNet

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**

- Paper：http://vladlen.info/papers/MSeg.pdf
- Code：https://github.com/mseg-dataset/mseg-api

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**

- Paper：https://arxiv.org/abs/2003.13845
- Dataset：https://github.com/lattas/AvatarMe

**Learning to Autofocus**

- Paper：https://arxiv.org/abs/2004.12260
- Dataset：No

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**

- Paper：https://arxiv.org/abs/2003.13989
- Code：https://github.com/zhuhao-nju/facescape

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**

- Paper：https://arxiv.org/abs/2004.01166

- Code：https://github.com/Healthcare-Robotics/bodies-at-rest
- Dataset：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**

- Homepage：https://sdolivia.github.io/FineGym/
- Paper：https://arxiv.org/abs/2004.06704

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**

- Homepage：https://anyirao.com/projects/SceneSeg.html

- Paper：https://arxiv.org/abs/2004.02678

- Code：https://github.com/AnyiRao/SceneSeg

**Deep Homography Estimation for Dynamic Scenes**

- Paper：https://arxiv.org/abs/2004.02132

- Dataset：https://github.com/lcmhoang/hmg-dynamics

**Assessing Image Quality Issues for Real-World Problems**

- Homepage：https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- Paper：https://arxiv.org/abs/2003.12511

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- Paper：https://arxiv.org/abs/2003.10608
- Code with Dataset：https://github.com/Jyouhou/UnrealText/

**PANDA: A Gigapixel-level Human-centric Video Dataset**

- Paper：https://arxiv.org/abs/2003.04852

- Dataset：http://www.panda-dataset.com/

**IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning**

- Paper：https://arxiv.org/abs/2003.02920
- Dataset：https://github.com/intra3d2019/IntrA

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- Paper：https://arxiv.org/abs/2003.03972
- Dataset：No

<a name="Others"></a>

# Others

**Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision**

- Paper：http://www.cvlibs.net/publications/Niemeyer2020CVPR.pdf

- Code：https://github.com/autonomousvision/differentiable_volumetric_rendering

**QEBA: Query-Efficient Boundary-Based Blackbox Attack**

- Paper：https://arxiv.org/abs/2005.14137
- Code：https://github.com/AI-secure/QEBA

**Equalization Loss for Long-Tailed Object Recognition**

- Paper：https://arxiv.org/abs/2003.05176
- Code：https://github.com/tztztztztz/eql.detectron2

**Instance-aware Image Colorization**

- Homepage：https://ericsujw.github.io/InstColorization/
- Paper：https://arxiv.org/abs/2005.10825
- Code：https://github.com/ericsujw/InstColorization

**Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting**

- Paper：https://arxiv.org/abs/2005.09704

- Code：https://github.com/Atlas200dk/sample-imageinpainting-HiFill

**Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching**

- Paper：https://arxiv.org/abs/2005.03860
- Code：https://github.com/shiyujiao/cross_view_localization_DSM

**Epipolar Transformers**

- Paper：https://arxiv.org/abs/2005.04551

- Code：https://github.com/yihui-he/epipolar-transformers 

**Bringing Old Photos Back to Life**

- Homepage：http://raywzy.com/Old_Photo/
- Paper：https://arxiv.org/abs/2004.09484

**MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask**

- Paper：https://arxiv.org/abs/2003.10955 

- Code：https://github.com/microsoft/MaskFlownet 

**Self-Supervised Viewpoint Learning from Image Collections**

- Paper：https://arxiv.org/abs/2004.01793
- Paper2：https://research.nvidia.com/sites/default/files/pubs/2020-03_Self-Supervised-Viewpoint-Learning/SSV-CVPR2020.pdf 
- Code：https://github.com/NVlabs/SSV 

**Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations**

- Oral

- Paper：https://arxiv.org/abs/2003.12237 
- Code：https://github.com/cuishuhao/BNM 

**Towards Learning Structure via Consensus for Face Segmentation and Parsing**

- Paper：https://arxiv.org/abs/1911.00957
- Code：https://github.com/isi-vista/structure_via_consensus

**Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging**

- Oral
- Paper：https://arxiv.org/abs/2003.13654

- Code：https://github.com/liuyang12/PnP-SCI

**Lightweight Photometric Stereo for Facial Details Recovery**

- Paper：https://arxiv.org/abs/2003.12307
- Code：https://github.com/Juyong/FacePSNet

**Footprints and Free Space from a Single Color Image**

- Paper：https://arxiv.org/abs/2004.06376

- Code：https://github.com/nianticlabs/footprints

**Self-Supervised Monocular Scene Flow Estimation**

- Paper：https://arxiv.org/abs/2004.04143
- Code：https://github.com/visinf/self-mono-sf

**Quasi-Newton Solver for Robust Non-Rigid Registration**

- Paper：https://arxiv.org/abs/2004.04322
- Code：https://github.com/Juyong/Fast_RNRR

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**

- Homepage：https://anyirao.com/projects/SceneSeg.html

- Paper：https://arxiv.org/abs/2004.02678

- Code：https://github.com/AnyiRao/SceneSeg

**DeepFLASH: An Efficient Network for Learning-based Medical Image Registration**

- Paper：https://arxiv.org/abs/2004.02097

- Code：https://github.com/jw4hv/deepflash

**Self-Supervised Scene De-occlusion**

- Homepage：https://xiaohangzhan.github.io/projects/deocclusion/
- Paper：https://arxiv.org/abs/2004.02788
- Code：https://github.com/XiaohangZhan/deocclusion

**Polarized Reflection Removal with Perfect Alignment in the Wild** 

- Homepage：https://leichenyang.weebly.com/project-polarized.html
- Code：https://github.com/ChenyangLEI/CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment 

**Background Matting: The World is Your Green Screen**

- Paper：https://arxiv.org/abs/2004.00626
- Code：http://github.com/senguptaumd/Background-Matting

**What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective**

- Paper：https://arxiv.org/abs/2003.11241

- Code：https://github.com/ZhangLi-CS/GCP_Optimization

**Look-into-Object: Self-supervised Structure Modeling for Object Recognition**

- Paper：No
- Code：https://github.com/JDAI-CV/LIO 

 **Video Object Grounding using Semantic Roles in Language Description**

- Paper：https://arxiv.org/abs/2003.10606
- Code：https://github.com/TheShadow29/vognet-pytorch 

**Dynamic Hierarchical Mimicking Towards Consistent Optimization Objectives**

- Paper：https://arxiv.org/abs/2003.10739
- Code：https://github.com/d-li14/DHM 

**SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization**

- Paper：http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf
- Code：https://github.com/YueJiang-nj/CVPR2020-SDFDiff 

**On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location**

- Paper：https://arxiv.org/abs/2003.07064

- Code：https://github.com/oskyhn/CNNs-Without-Borders

**GhostNet: More Features from Cheap Operations**

- Paper：https://arxiv.org/abs/1911.11907

- Code：https://github.com/iamhankai/ghostnet

**AdderNet: Do We Really Need Multiplications in Deep Learning?** 

- Paper：https://arxiv.org/abs/1912.13200 
- Code：https://github.com/huawei-noah/AdderNet

**Deep Image Harmonization via Domain Verification** 

- Paper：https://arxiv.org/abs/1911.13239 
- Code：https://github.com/bcmi/Image_Harmonization_Datasets

**Blurry Video Frame Interpolation**

- Paper：https://arxiv.org/abs/2002.12259 
- Code：https://github.com/laomao0/BIN

**Extremely Dense Point Correspondences using a Learned Feature Descriptor**

- Paper：https://arxiv.org/abs/2003.00619 
- Code：https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch

**Filter Grafting for Deep Neural Networks**

- Paper：https://arxiv.org/abs/2001.05868
- Code：https://github.com/fxmeng/filter-grafting
- PaperBlog：https://www.zhihu.com/question/372070853/answer/1041569335

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**

- Paper：https://arxiv.org/abs/2003.02824 
- Code：https://github.com/cmhungsteve/SSTDA

**Detecting Attended Visual Targets in Video**

- Paper：https://arxiv.org/abs/2003.02501 

- Code：https://github.com/ejcgt/attention-target-detection

**Deep Image Spatial Transformation for Person Image Generation**

- Paper：https://arxiv.org/abs/2003.00696 
- Code：https://github.com/RenYurui/Global-Flow-Local-Attention

 **Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications** 

- Paper：https://arxiv.org/abs/2003.01455
- Code：https://github.com/bbrattoli/ZeroShotVideoClassification

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/Anonymous20192020/Anonymous_CVPR5767

https://github.com/avirambh/ScopeFlow

https://github.com/csbhr/CDVD-TSP

https://github.com/ymcidence/TBH

https://github.com/yaoyao-liu/mnemonics

https://github.com/meder411/Tangent-Images

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

https://github.com/sjmoran/deep_local_parametric_filters

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/bermanmaxim/AOWS

https://github.com/dc3ea9f/look-into-object 

<a name="Not-Sure"></a>

# Not Sure

**FADNet: A Fast and Accurate Network for Disparity Estimation**

- Paper：还没出来
- Code：https://github.com/HKBU-HPML/FADNet

https://github.com/rFID-submit/RandomFID：Not Sure

https://github.com/JackSyu/AE-MSR：Not Sure

https://github.com/fastconvnets/cvpr2020：Not Sure

https://github.com/aimagelab/meshed-memory-transformer：Not Sure

https://github.com/TWSFar/CRGNet：Not Sure

https://github.com/CVPR-2020/CDARTS：Not Sure

https://github.com/anucvml/ddn-cvprw2020：Not Sure

https://github.com/dl-model-recommend/model-trust：Not Sure

https://github.com/apratimbhattacharyya18/CVPR-2020-Corr-Prior：Not Sure

https://github.com/onetcvpr/O-Net：Not Sure

https://github.com/502463708/Microcalcification_Detection：Not Sure

https://github.com/anonymous-for-review/cvpr-2020-deep-smoke-machine：Not Sure

https://github.com/anonymous-for-review/cvpr-2020-smoke-recognition-dataset：Not Sure

https://github.com/cvpr-nonrigid/dataset：Not Sure

https://github.com/theFool32/PPBA：Not Sure

https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition
