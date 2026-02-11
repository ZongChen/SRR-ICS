

# python train_clip_ics.py --dataset 'market1501' --K_search 28 --K_intra 8 --K_cross 10 --tau_intra 2.5 --beta 0.8 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
# python train_clip_ics.py --dataset 'dukemtmc' --K_search 40 --K_intra 25 --K_cross 15 --tau_intra 2.5 --beta 0.6 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
# python train_clip_ics.py --dataset 'dukemtmc' --K_search 50 --K_intra 25 --K_cross 15 --tau_intra 2.5 --beta 0.6 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled

python train_clip_ics.py --dataset 'market1501' --distance CAJ --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
python train_clip_ics.py --dataset 'market1501' --distance UN --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
python train_clip_ics.py --dataset 'market1501' --distance ICS --K_search 60 --K_intra 25 --K_cross 15 --tau_intra 2.5 --beta 0.76 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled

#python train_clip_ics.py --dataset 'market1501' --K_search 28 --K_intra 8 --K_cross 10 --tau_intra 2.5 --beta 0.7 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
# python train_clip_ics.py --dataset 'market1501' --K_search 20 --K_intra 10 --K_cross 10 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled

# python train_clip_ics.py --dataset 'dukemtmc' --K_search 28 --K_intra 8 --K_cross 10 --tau_intra 2.5 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled


# 1227
# python train_clip_ics.py --dataset 'msmt17' --K_search 40 --K_intra 20 --K_cross 20 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
# python train_clip_ics.py --dataset 'msmt17' --K_search 60 --K_intra 10 --K_cross 50 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
# python train_clip_ics.py --dataset 'msmt17' --K_search 40 --K_intra 8 --K_cross 1 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled
# python train_clip_ics.py --dataset 'msmt17' --K_search 28 --K_intra 8 --K_cross 30 --data-dir '/media/deep/SSD/Dataset_ReID' --wandb_enabled