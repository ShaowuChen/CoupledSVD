#!/bin/bash
python train_resnet_decom.py --rank_rate_SVD=0.2 --rank_rate_shared=0.02 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.2 --rank_rate_shared=0.02 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=40 --num_lr=1e-2 --change_lr="[20, 30, 35]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.5 --rank_rate_shared=0.25 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.5 --rank_rate_shared=0.25 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.5 --rank_rate_shared=0.1 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.5 --rank_rate_shared=0.1 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.5 --rank_rate_shared=0.05 --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=7
python train_resnet_decom.py --rank_rate_SVD=0.5 --rank_rate_shared=0.05 --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=10 --num_lr=1e-3 --change_lr="[5, 8]" --gpu=7
