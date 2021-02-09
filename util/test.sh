#-----------exp1 resnet18 resnet34;decom_conv1=True; SVD, TT, JSVD, PCSVD
echo -3
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=SVD  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=TT  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=JSVD  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=PCSVD  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
echo -2
python train_resnet_decom.py --rank_rate_SVD=0.9  --rank_rate_shared=0.3  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.3  --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'


SVD
TT
JSVD

PCSVD
FCSVD
PCSVD+FCSVD

#-----------exp2 ResNet50, ResNet101; ;SVD, TT, (JSVD), PCSVD, FCSVD
echo -1
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=SVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=TT  --decom_conv1=False  --model=resnet50 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=PCSVD  --decom_conv1=False  --model=resnet50 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=FCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --gpu='2'

# echo 0
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=SVD  --decom_conv1=True  --model=resnet101 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=TT  --decom_conv1=True  --model=resnet101 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=JSVD  --decom_conv1=True  --model=resnet101 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=PCSVD  --decom_conv1=True  --model=resnet101 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=FCSVD  --decom_conv1=True  --model=resnet101 --dataset=cifar10 --gpu='2'

#-----------exp3 ResNet18, ResNet34; SVD, TT, (JSVD), PCSVD+FCSVD
echo 1
# python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=SVD  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=TT  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=JSVD  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
# python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=PCSVD+FCSVD  --decom_conv1=True  --model=resnet18 --dataset=cifar10 --gpu='2'
echo 2
python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'
python train_resnet_decom.py --rank_rate_SVD=0.7  --rank_rate_shared=0.1  --method=PCSVD+FCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --gpu='2'


