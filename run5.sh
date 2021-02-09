#!/bin/bash
python train_resnet_decom.py --rank_rate_SVD=0.04 --rank_rate_shared=0.02 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.04 --rank_rate_shared=0.02 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.04 --rank_rate_shared=0.008 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.04 --rank_rate_shared=0.008 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.04 --rank_rate_shared=0.004 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.04 --rank_rate_shared=0.004 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.08 --rank_rate_shared=0.04 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
python train_resnet_decom.py --rank_rate_SVD=0.08 --rank_rate_shared=0.04 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=5
