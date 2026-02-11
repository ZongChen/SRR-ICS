<p align="center">
  <h1 align="center">SRR-ICS</h1>
  

  
### Method
![SRR-ICS](imgs/overview.png)

This is an official code implementation of SRR-ICS".


### Preparation

Download the datasets:

For privacy reasons, we don't have the dataset's copyright. Please contact authors to get this dataset.

```

Market-1501-v15.09.15/
├── bounding_box_test
├── bounding_box_train
├── gt_bbox
├── gt_query
└── query

MSMT17/
├── bounding_box_test
├── bounding_box_train
└── query

DukeMTMC-reID/
├── bounding_box_test
├── bounding_box_train
└── query

```



## Training
```shell
sh run_usl.sh 
```


### References.

[1] Bianchi, Federico, et al. "Contrastive language-image pre-training for the italian language." arXiv preprint arXiv:2108.08688 (2021).

[2] Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." International Journal of Computer Vision 130.9 (2022): 2337-2348.

[3] Li, Siyuan, Li Sun, and Qingli Li. "Clip-reid: Exploiting vision-language model for image re-identification without concrete text labels." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 1. 2023.

[4] Dai, Zuozhuo, et al. "Cluster contrast for unsupervised person re-identification." Proceedings of the Asian Conference on Computer Vision. 2022.


