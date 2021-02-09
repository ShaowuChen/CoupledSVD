#!/bin/bash
python train_resnet_decom.py --rank_rate_SVD=0.08 --rank_rate_shared=0.016 --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.08 --rank_rate_shared=0.016 --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.08 --rank_rate_shared=0.008 --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.08 --rank_rate_shared=0.008 --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.2 --rank_rate_shared=0.1 --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.2 --rank_rate_shared=0.1 --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.2 --rank_rate_shared=0.04 --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=2
python train_resnet_decom.py --rank_rate_SVD=0.2 --rank_rate_shared=0.04 --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=2
