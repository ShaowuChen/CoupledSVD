import numpy as np 
f = open('./run.sh','w+')
'''
============================================================================================================
                                            exp1
============================================================================================================
'''
rank_rate_SVD = [0.04, 0.08, 0.2, 0.5]
rank_rate_shared_factor = [2,5,10]
epoch = [60,60,40,10]
num_lr = ['1e-1', '1e-1', '1e-2', '1e-3']
change_lr = [[25,40,50], [25,40,50],[20,30,35],[5,8]]

for i, item in enumerate(rank_rate_SVD):
    f.write('python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))
    f.write('python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))


for i, item1 in enumerate(rank_rate_SVD):
    rank_rate_shared = [item1/item2 for item2 in rank_rate_shared_factor]
    for j, item2 in enumerate(rank_rate_shared):
        f.write('python train_resnet_decom.py --rank_rate_SVD=%s --rank_rate_shared=%s --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(item1), str(item2), str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))
        f.write('python train_resnet_decom.py --rank_rate_SVD=%s --rank_rate_shared=%s --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10 --exp_path=exp1 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(item1), str(item2), str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))



'''
============================================================================================================
                                            exp2
============================================================================================================
'''
for i, item in enumerate(rank_rate_SVD):
    f.write('python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))
    f.write('python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))


for i, item1 in enumerate(rank_rate_SVD):
    rank_rate_shared = [item1/item2 for item2 in rank_rate_shared_factor]
    for j, item2 in enumerate(rank_rate_shared):
        f.write('python train_resnet_decom.py --rank_rate_SVD=%s --rank_rate_shared=%s --method=JSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(item1), str(item2), str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))
        f.write('python train_resnet_decom.py --rank_rate_SVD=%s --rank_rate_shared=%s --method=PCSVD  --decom_conv1=True  --model=resnet50 --dataset=cifar10 --exp_path=exp2 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(item1), str(item2), str(epoch[i]) ,str(num_lr[i]), str(change_lr[i])))


'''
============================================================================================================
                                            exp3
============================================================================================================
'''
# epoch = [60,60,40,10]
# num_lr = ['1e-1', '1e-1', '1e-2', '1e-3']
# change_lr = [[25,40,50], [25,40,50],[20,30,35],[5,8]]
# for i, item in enumerate(rank_rate_SVD):
#     f.write('python train_resnet_decom.py --rank_rate_SVD=0.9  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=imagenet  --repeat_exp_times=1 --exp_path=exp3 --epoch=%s --num_lr=%s --change_lr="%s"')
#     f.write('python train_resnet_decom.py --rank_rate_SVD=0.9  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=imagenet  --repeat_exp_times=1 --exp_path=exp3 --epoch=%s --num_lr=%s --change_lr="%s"')


# for i, item1 in enumerate(rank_rate_SVD):
#     rank_rate_shared = [item1/item2 for item2 in rank_rate_shared_factor]
#     for j, item2 in enumerate(rank_rate_shared):
#         f.write('python train_resnet_decom.py --rank_rate_SVD=%s --rank_rate_shared=%s --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=imagenet --repeat_exp_times=1 --exp_path=exp3 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(item1), str(item2))
#         f.write('python train_resnet_decom.py --rank_rate_SVD=%s --rank_rate_shared=%s --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=imagenet --repeat_exp_times=1 --exp_path=exp3 --epoch=%s --num_lr=%s --change_lr="%s"\n'%(str(item1), str(item2))


# Delete
for i in range(8):
    print(i)
    with open('run%d.sh'%(i), 'w+') as f1:
        pass

#Assign
f = open('run.sh','r')
lines = f.readlines()
len_lines = len(lines)
num = int(len_lines/8)
left = len_lines-num*8

for i in range(8):
    with open('run%d.sh'%(i), 'a') as f1:
        f1.write('#!/bin/bash\n')

for i in range(8):
    print(i)
    with open('run%d.sh'%(i), 'a') as f1:
        for item in lines[i*num:(i+1)*num]:
            f1.write(item.replace('\n', ' --gpu=%d\n'%i)) 

assert(left<8)
for i in range(left):
    print(i)
    with open('run%d.sh'%(i), 'a') as f1:
        f1.write(lines[8*num+i].replace('\n', ' --gpu=%d\n'%i)) 




