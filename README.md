# Relative Depth Guided Cross Domain Semantic Segmentation for High Resolution Satellite Remote Sensing Images

## Introduction
Official codebase of the paper **Relative Depth Guided Cross Domain Semantic Segmentation for High Resolution Satellite Remote Sensing Images** by ***Yongqi Sun, Yujun Quan, Anzhu Yu, Xin Li, Chenguang Dai, Xuanguagn Liu and Yanfei Zhong***.

Due to sensor, imaging condition, and regional variations, satellite remote sensing images often have cross-domain discrepancies, degrading the inference performance of source-domain trained semantic segmentation models on target domains. Although unsupervised domain adaptation (UDA) and remote sensing foundation models mitigate this issue, UDA requires accessible target-domain images during training, while foundation models drastically increase computational complexity, hindering lightweight deployment. To address these challenges, this paper leverages relative depth maps as prior to guide cross-domain semantic segmentation of satellite remote sensing images. For seamless integration into most end-to-end encoder-decoder models, we make three key contributions: 1) A unified depth-guided framework for optical remote sensing cross-domain segmentation is designed.  2) A multi-scale contrastive learning module is proposed to align optical and relative depth features, enhancing optical feature representation. 3) A refinement strategy using shallow features of relative depth is developed to guide the prediction of semantic masks.Extensive experiments on the public LoveDA dataset and a self-constructed high-resolution Arctic cross-seasonal dataset demonstrate that incorporating relative depth improves segmentation performance. Further feature enhancement and semantic mask refinement via relative depth boost cross-domain capability, outperforming UDA in some cases without target-domain training data. These results validate the correctness, effectiveness, and potential of the proposed framework for cross-domain segmentation of satellite optical remote sensing. 



## Code 

Implemented with MMsegmenataion and Open-CD (for multi-modal).

Available when the paper accepted. 
