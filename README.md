# SRR-ICS

Official implementation of **Structured Re-ranking for Intra-Camera Supervised
Person Re-Identification**.

## Preparation

Install the Python dependencies:

```shell
pip install -r requirements.txt
```

Download the datasets from the official providers. For privacy and license
reasons, this repository does not redistribute the datasets.

Organize the datasets as follows:

```text
data/
├── Market-1501-v15.09.15/
│   ├── bounding_box_test
│   ├── bounding_box_train
│   ├── gt_bbox
│   ├── gt_query
│   └── query
├── MSMT17/
│   ├── bounding_box_test
│   ├── bounding_box_train
│   └── query
└── DukeMTMC-reID/
    ├── bounding_box_test
    ├── bounding_box_train
    └── query
```

If your datasets are stored elsewhere, pass the root with `--data-dir`.

## Training

```shell
bash run_usl.sh
```

Equivalent explicit command:

```shell
python train_clip_ics.py --dataset market1501 --distance ICS \
  --K_search 60 --K_intra 25 --K_cross 15 --tau_intra 2.5 --beta 0.76 \
  --data-dir data
```

Supported datasets are `market1501`, `dukemtmc`, and `msmt17`.

## Evaluation

```shell
python test.py --dataset market1501 --data-dir data --resume logs/train_ics/market1501/model_best.pth.tar
```

Weights & Biases logging is optional. Add `--wandb_enabled` only after installing
and configuring `wandb`.

## References

[1] Bianchi, Federico, et al. "Contrastive language-image pre-training for the italian language." arXiv preprint arXiv:2108.08688 (2021).

[2] Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." International Journal of Computer Vision 130.9 (2022): 2337-2348.

[3] Li, Siyuan, Li Sun, and Qingli Li. "Clip-reid: Exploiting vision-language model for image re-identification without concrete text labels." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 1. 2023.

[4] Dai, Zuozhuo, et al. "Cluster contrast for unsupervised person re-identification." Proceedings of the Asian Conference on Computer Vision. 2022.

[5] Tan, Xuan, Xun Gong, and Yang Xiang. "CLIP-based camera-agnostic feature learning for intra-camera supervised person re-identification." IEEE Transactions on Circuits and Systems for Video Technology 35.5 (2024): 4100-4115.
