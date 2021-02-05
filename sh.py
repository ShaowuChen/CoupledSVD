# datasets = ['cifar10', 'imagenet']
# # models = ['resnet34', 'resnet34']
# stru_or_unstru = ['structured','unstructured']
# method1s = ['traditional', 'proposed']
# method2s = ['p1', 'p2', 'p3']
# cr_traditional = []
# cr_share = 
# cr_independent = 
# repeat_exp_times = 

# batch_size = 
# epoch = 
# change_lr
# lr_decay
# ckpt_path='20210128\n''
# gpu
import numpy as np 
f = open('./run.sh','w+')
'''
============================================================================================================
                                            Cifar10
============================================================================================================
'''
epoches_st = [60, 40, 40, 20, 20]
num_lrs_st = ['1e-1','1e-1','1e-2','1e-3','1e-3']
change_lrs_st = [[30,40,50], [20,30,40],[20,30,35],[10,15,18],[10,15,18]]


epoches_un = [50, 50, 40, 20, 20]
num_lrs_un = ['1e-3','1e-3','1e-3','1e-3','1e-3']
change_lrs_un = [[40], [40],[30],[15],[15]]

'''
============================================================================================================
                                            imagenet
============================================================================================================
'''
epoches_st = [30, 30, 30, 20, 20]
num_lrs_st = ['1e-1','1e-1','1e-2','1e-3','1e-3']
change_lrs_st = [[20,25], [20,25],[20,25],[10,15,18],[10,15,18]]

epoches_un = [10, 10, 10, 5, 5]
num_lrs_un = ['1e-3','1e-3','1e-3','1e-3','1e-3']
change_lrs_un = [[7], [7],[7],[2,3],[2,3]]

################## traditioanl #############################

################ unstructured ###################

## without consider cr_share ######
cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]
for i, item in enumerate(cr_traditional):
        f.write('python train_resnet_pruning_unstructured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=unstructured --method1=traditional '+\
         '--cr_traditional=%f --cr_share=None --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, epoches_un[i], str(change_lrs_un[i]), str(num_lrs_un[i])))



############  consider cr_share ######
# cr_traditional = [0.005, 0.01,  0.05,  0.2, 0.5]
cr_traditional = [0.005, 0.01]

for i, item in enumerate(cr_traditional):
    cr_share = [item/num for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
            f.write('python train_resnet_pruning_unstructured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=unstructured --method1=traditional '+\
             '--cr_traditional=%f --cr_share=%f --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_un[i], str(change_lrs_un[i]), str(num_lrs_un[i])))




################ structured ###################

### without consider cr_share ######
cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):
    f.write('python train_resnet_pruning_structured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=structured --method1=traditional '+\
         '--cr_traditional=%s --cr_share=None --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, epoches_st[i], str(change_lrs_st[i]), str(num_lrs_st[i])))



############  consider cr_share ######
cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):
    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in ([5,2] if i==0 else [10,5,2])]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_structured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=structured --method1=traditional '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_st[i], str(change_lrs_st[i]), str(num_lrs_st[i])))

# 




################# Proposed#############################

############### unstructured ###################

################ p1 ##################

# cr_traditional = [0.005, 0.01,  0.05,  0.2, 0.5]
cr_traditional = [0.005, 0.01]
for i, item in enumerate(cr_traditional):
    cr_share = [item/num for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_unstructured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=unstructured --method1=proposed --method2=p1 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_un[i], str(change_lrs_un[i]), str(num_lrs_un[i])))





################# p2 ##################

# cr_traditional = [0.005, 0.01,  0.05,  0.2, 0.5]
cr_traditional = [0.005, 0.01]
for i, item in enumerate(cr_traditional):
    cr_share = [item/num for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_unstructured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=unstructured --method1=proposed --method2=p2 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_un[i], str(change_lrs_un[i]), str(num_lrs_un[i])))



################# p3 ##################

# cr_traditional = [0.005, 0.01,  0.05,  0.2, 0.5]
cr_traditional = [0.005, 0.01]
for i, item in enumerate(cr_traditional):
    cr_share = [item/num for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_unstructured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=unstructured --method1=proposed --method2=p3 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_un[i], str(change_lrs_un[i]), str(num_lrs_un[i])))





# # ################## structured ###################

################# p11 ##################
cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):

    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in ([5,2] if i==0 else [10,5,2])]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_structured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=structured --method1=proposed --method2=p11 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_st[i], str(change_lrs_st[i]), str(num_lrs_st[i])))


################# p12 ##################
cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):
    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in ([5,2] if i==0 else [10,5,2])]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_structured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=structured --method1=proposed --method2=p12 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_st[i], str(change_lrs_st[i]), str(num_lrs_st[i])))



################# p2 ##################
cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):
    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in ([5,2] if i==0 else [10,5,2])]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_structured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=structured --method1=proposed --method2=p2 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_st[i], str(change_lrs_st[i]), str(num_lrs_st[i])))



################# p3 ##################
cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):
    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in ([5,2] if i==0 else [10,5,2])]
    for j, item2 in enumerate(cr_share):
        f.write('python train_resnet_pruning_structured.py --dataset=imagenet --model=resnet34 --stru_or_unstru=structured --method1=proposed --method2=p3 '+\
             '--cr_traditional=%s --cr_share=%s --epoch=%d --change_lr=\'%s\' --num_lr=%s --ckpt_path=20210131\n'%(item, item2, epoches_st[i], str(change_lrs_st[i]), str(num_lrs_st[i])))
f.close()

#Delete
for i in range(8):
    print(i)
    with open('run%d.sh'%(i+20), 'w+') as f1:
        pass

#Assign
f = open('run.sh','r')
lines = f.readlines()
len_lines = len(lines)
num = int(len_lines/8)
left = len_lines-num*8

for i in range(8):
    print(i)
    with open('run%d.sh'%(i+20), 'a') as f1:
        for item in lines[i*num:(i+1)*num]:
            f1.write(item.replace('\n', ' --gpu=%d\n'%i)) 

assert(left<8)
for i in range(left):
    print(i)
    with open('run%d.sh'%(i+20), 'a') as f1:
        f1.write(lines[8*num+i].replace('\n', ' --gpu=%d\n'%i)) 




