#!/bin/bash
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=60 --num_lr=1e-1 --change_lr="[25, 40, 50]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=0
python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=0
