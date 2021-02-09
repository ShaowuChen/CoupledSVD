
1、Main approach： SVD & TT & JSVD & PCSVD & FCSVD

    1.1.baseline: SVD & TT
        SVD: Fundanuntal but power;
        TT： If not fold, in 1x1 not 2D tensor, equailed to SVD;
             If fold, need to fold them back, slow them down;

    1.2.Proposed:
        1.2.1.Partly shared
            1.2.1.1. JointSVD: JSVD
            1.2.1.2. PartlyCoupledSVD: PCSVD        
        1.2.2.Fully Coupled SVD: FCSVD

2、Experiments: layer1 暂时不分解(因为处理起来比较麻烦)

BN层，layer1一起finetune
BasicBlock:不分解layer1
BotteleNeck:分解layer1, 只分解Conv2————因为conv1/conv3处理起来麻烦，而且1x1没有分解的必要

    2.1.Cifar10————表格不够多的话，把时间单独拿出来比较
        注意一样的CR选择，方便跨试验比较 
        # (2.1.0 SVD, TT, JSVD, PCSVD on resnet8)

        2.1.1. SVD, TT, JSVD, PCSVD
                    ResNet18, ResNet34;
                    layer1不分解；
                    第一层分解；
                    Focus on high CR;                    
                    w/o&w fine-tune;
                    Time;

        2.1.2. SVD, TT, (JSVD), PCSVD, FCSVD
                    ResNet50, ResNet101;
                    layer1分解；
                    只分解conv2;
                    Focus on high CR;
                    w/o&w fine-tune;
                    Time;

    
    2.2.Imagenet：挑着做
        SVD, TT, (JSVD), PCSVD+FCSVD
            ResNet18? ResNet34; 
            layer1不分解；
            第一层PCSVD，其余层FCSVD；
            Focus on high CR;
            w/o&w fine-tune;
            Time;



    2.3 可视化/能量分析
        先初始化独立分量，则大部分能量集中在独立上；
        先初始化共享分量，则大部分能量集中在共享上；
        但是cr_share, cr_independent的选择又改变了这一事实
        我猜测，高CR下，rank_independent小，rank_share大，效果好的原因是对应的呢能量高
        所以：

        对w/o，w的filter进行能量分析（Norm分布？）

        然后，取某一部分进行可视化展示


3.Code
    3.1 Resnet_decom
        method = SVD/TT/JSVD/PCSVD/FCSVD
        decom_conv1 = True/False
    3.2.train_resnet_decom
        WarmUp
    3.3 get_parameter




TODO:
1.1 、试验设计部分再斟酌下——OK
因为修改了要求:
    BotteleNeck conv1 conv3不分解

1.2（所以还需要取修改conv1x1函数）

2、完成Calculate的编写

3、针对第一层PCSVD，其余层FCSVD模式，需要做一些代码改变

  特别是，对PCSVD,FCSVD传入的layer_list, repeat_list, conv_list, layers_out_channels需要再整整

4、修改get_parameters dict 中PCSVD FCSVD对应的UV变量名字，方便两种模式混合使用 

5、修改JSVD中的U，改为U_JSVD?? 否则回合independent中的U起冲突




维持以O为选择标准？？

写作：
不写rq了，只写压缩率；
对于Coupled，写 shared:Independent = 1:几这样子

'''
=============Bug&Debug===============
'''
#bug:
FCSVD的CR不对，大概率出在decom_conv1上

#debug
应该设置conv_decom=False



python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.2  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10
python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.1  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10
这两项数据不一样一样，有问题