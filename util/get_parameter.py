import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow
import numpy as np 
import copy
import sys
flags = tf.flags
FLAGS=flags.FLAGS


cifar10_Path = {
    'resnet18': None,
    'resnet34': '/mnt/sdb1/CSW/Orig_ckpt/cifar/resnet34/resnet34_top1_0.9510999977588653-227',
    'resnet50': '/mnt/sdb1/CSW/Orig_ckpt/cifar/resnet50/resnet50_top1_0.9502000004053116-278',
    'resnet101': None,
    'resnet152': None
}

imagenet_Path = {
    'resnet18': None,
    'resnet34':'/mnt/sdb1/CSW/Orig_ckpt/imagenet/resnet34/resnet34_top1_0.7102599966526032-5',
    'resnet50': None,
    'resnet101': None,
    'resnet152': None
}
Path = {'cifar10':cifar10_Path, 'imagenet':imagenet_Path}

#include layer1
expansion = 4

orig_Repeat_list = {
    'resnet18': [2,2,2,2],
    'resnet34': [3,4,6,3],
    'resnet50': [3,4,6,3],
    'resnet101': [3,4,23,3],
    'resnet152': [3,8,36,3]
}
orig_Conv_list = {
    'resnet18': [1,2],
    'resnet34': [1,2],
    'resnet50': [1,2,3],
    'resnet101': [1,2,3],
    'resnet152': [1,2,3]
}

#exclude layer1
expansion = 4

Repeat_list = {
    'resnet18': [2,2,2],
    'resnet34': [4,6,3],
    'resnet50': [4,6,3],
    'resnet101': [4,23,3],
    'resnet152': [8,36,3]
}
#BasicBlock: decompose both conv1 and conv2, i.e., the 3x3 kernels 
#BottleBlock: only decompose conv2, i.e., the 3x3 kernel
Conv_list = {
    'resnet18': [1,2],
    'resnet34': [1,2],
    'resnet50': [2],
    'resnet101': [2],
    'resnet152': [2]
}





'''
============================================================
                    API
============================================================
'''
def get_parameter():

    assert(FLAGS.method in ['SVD', 'TT', 'JSVD', 'PCSVD', 'FCSVD', 'PCSVD+FCSVD'])

    path = Path[FLAGS.dataset][FLAGS.model]
    npy_dict = ckpt2npy(path)

    #don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        layer_list = [2,3,4]
        repeat_list = Repeat_list[FLAGS.model]
        conv_list = Conv_list[FLAGS.model]
        layers_out_channels = [128,256,512]
        if FLAGS.method=='FCSVD':
            assert(FLAGS.decom_conv1==False)
    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else:
        layer_list = [1,2,3,4]
        repeat_list = orig_Repeat_list[FLAGS.model]
        conv_list = [2] 
        layers_out_channels = [64,128,256,512]
       

    function = FLAGS.method+'_Parameter'
    if FLAGS.method=='PCSVD+FCSVD':
        assert(FLAGS.decom_conv1==True)
        assert(FLAGS.model in ['resnet18', 'resnet34'])
        function = 'PCSVD_FCSVD_Parameter'

    decomposed_dict, record_dict = eval(function)(copy.deepcopy(npy_dict), layer_list, repeat_list, conv_list, layers_out_channels, eval(FLAGS.rank_rate_SVD), eval(FLAGS.rank_rate_shared))

    weight_dict = {**npy_dict, **decomposed_dict}

    return weight_dict, record_dict



def ckpt2npy(path):
    
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var = reader.get_variable_to_shape_map()
    npy_dict = {}

    for key in var:
        # print(key)
        value = reader.get_tensor(key)
        npy_dict[key] = value

    return npy_dict.copy()



'''
============================================================
                    SVD
============================================================

'''
def SVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):

    SVD_dict = {}
    for i, layer_name in enumerate(layer_list):
        for repeat_th in range(repeat_list[i]):
            for conv in conv_list:

                weight_name = 'layer'+str(layer_name)+'.'+str(repeat_th)+'.conv'+str(conv)+'.weight'
                weight = npy_dict[weight_name] 
                U, V = decompose_SVD_4d(copy.deepcopy(weight), rank_rate_SVD, '4d')
                SVD_dict[weight_name.replace('weight', 'U')] = U
                SVD_dict[weight_name.replace('weight', 'V')] = V

    num_svd = num_SVD(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)
    CR = num_orig_all() / (num_undecomposed_parameter() + num_svd)
    return SVD_dict, {'num_decomposed':num_svd, 'num_undecomposed':num_undecomposed_parameter(), 'CR':CR}


def decompose_SVD_4d(weight, rank_rate_SVD, shape_option:'2d/4d/mix'='4d'):
    [F1, F2, I, O] = weight.shape  # weight[F1,F2,I,O]
    #let the 4-th dimension, O(output channels), as the baseline
    rank=int(O*rank_rate_SVD)

    W = np.reshape(np.transpose(weight,(0,2,1,3)), [F1*I, -1])
    U_2d, S_2d, V_2d = np.linalg.svd(W, full_matrices=True)

    U_2d = np.dot(U_2d[:, 0:rank].copy(),np.diag(S_2d)[0:rank, 0:rank].copy())
    U_4d = np.transpose(np.reshape(U_2d, [F1, I, 1, rank]),(0,2,1,3))

    V_2d = V_2d[0:rank, :].copy()
    V_4d = np.transpose(np.reshape(V_2d,[1,rank,F2,O]),(0,2,1,3))

    if shape_option=='4d':
        return U_4d, V_4d
    elif shape_option=='2d':
        return U_2d, V_2d
    elif shape_option=='mix':
        return U_2d, V_4d
    else:
        assert(0)

def decompose_SVD_2d(weight_2d, rank):

    U_2d, S_2d, V_2d = np.linalg.svd(weight_2d, full_matrices=True)

    U_2d = np.dot(U_2d[:, 0:rank].copy(),np.diag(S_2d)[0:rank, 0:rank].copy())
    V_2d = V_2d[0:rank, :].copy()

    return U_2d, V_2d

def chagne_2D_to_4d(weight_2d, shape, rank, U_or_V=None):
    [F1, F2, I, O] = shape
    if U_or_V=='U':
        weight_4d = np.transpose(np.reshape(weight_2d, [F1, I, 1, rank]),(0,2,1,3))
    elif U_or_V=='V':
        weight_4d = np.transpose(np.reshape(weight_2d, [1,rank,F2,O]),(0,2,1,3))
    else:
        assert(0)
    return weight_4d


#num of parameters in the svd based decomposed layers under rank_rate_svd
def num_SVD(rank_rate_svd, layer_list, repeat_list, conv_list, layers_out_channels, _option=True):
    num = 0
    # print(len(Repeat_list))
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        Rank_svd = [int(item*rank_rate_svd) for item in layers_out_channels]
        #parameters of decomposed conv except conv1 of repeat0     
        for i in range(len(repeat_list)):
            num += 3*layers_out_channels[i]*Rank_svd[i]*2 * (2*repeat_list[i]-1)#each BasicBlock contains 2 3x3conv

        if _option:
            #parameters of decomposed conv1 of repeat0     
            if FLAGS.decom_conv1==True:
                for i in range(len(repeat_list)):#let the 4-th dimension, O(output channels), as the baseline
                    num += 3*(layers_out_channels[i]/2)*Rank_svd[i] + 3*layers_out_channels[i]*Rank_svd[i]

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        Rank_svd = [int(item*rank_rate_svd) for item in layers_out_channels]
        for i in range(len(repeat_list)):
            num += 3*layers_out_channels[i]*Rank_svd[i]*2 * repeat_list[i]#each one contains 1 3x3conv

    return num




'''
============================================================
                    TT
============================================================
'''
def TT_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):

    rank_rate_TT = calculate_tt_rate(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)

    TT_dict = {}
    for i, layer_name in enumerate(layer_list):
        for repeat_th in range(repeat_list[i]):
            for conv in conv_list:

                weight_name = 'layer'+str(layer_name)+'.'+str(repeat_th)+'.conv'+str(conv)+'.weight'
                weight = npy_dict[weight_name] 

                G1, G2, G3 = decompose_TT(copy.deepcopy(weight), rank_rate_TT)

                TT_dict[weight_name.replace('weight', '1')] = G1
                TT_dict[weight_name.replace('weight', '2')] = G2
                TT_dict[weight_name.replace('weight', '3')] = G3

    num_tt = num_TT(rank_rate_TT, layer_list, repeat_list, conv_list, layers_out_channels)
    CR = num_orig_all() / (num_undecomposed_parameter() + num_tt)
    return TT_dict, {'num_decomposed':num_tt, 'num_undecomposed':num_undecomposed_parameter(), 'CR':CR}

def decompose_TT(value, rank_rate_TT):
    shape = value.shape
    (h,w,i,o) = shape

    assert (len(shape) == 4)
    assert (o >= i)

    rank_i=int(rank_rate_TT*i)
    rank_o=int(rank_rate_TT*o)

    # [H,W,I,O]
    value_2d = np.reshape(value, [h * w * i, o])
    U2, S2, V2 = np.linalg.svd(value_2d)

    # max rank: o
    V2 = np.matmul(np.diag(S2)[:rank_o, :rank_o], V2[:rank_o, :])
    U2_cut = U2[:, :rank_o]

    # to [i,h,w,rank2] and then [i, hw*rank2]
    U2_cut = np.transpose(np.reshape(U2_cut, [h, w, i, rank_o]), [2, 0, 1, 3])
    U2_cut = np.reshape(U2_cut, [i, h * w * rank_o])

    U1, S1, V1 = np.linalg.svd(U2_cut)
    # max rank: i
    assert(rank_i<=len(S1))
    V1 = np.matmul(np.diag(S1)[:rank_i, :rank_i], V1[:rank_i, :])
    U1_cut = U1[:, :rank_i]

    G3 = np.reshape(V2, [1, 1, rank_o, o])
    G2 = np.transpose(np.reshape(V1, [rank_i, h, w, rank_o]), [1, 2, 0, 3])
    G1 = np.reshape(U1_cut, [1, 1, i, rank_i])

    return G1,G2,G3


 

def calculate_tt_rate(rank_rate_svd, layer_list, repeat_list, conv_list, layers_out_channels):
    #r_1 is based on 'I'; r_2 is based on 'O'
    #i.e., r_1 = int(I*rank_rate_tt)
    #r_2 = int(O*rank_rate_tt)

    #rr=rank_rate_tt which is need to solve from function.
    # num_tt = sum( 1*1*i * (i * rr)  + 9 * (i*rr) * (o*rr) + 1*1*o*(o*rr)=sum(9*i*o) * (rr^2) + sum((i*i+o*o))*rr) = num_svd
    #solve function: a = sum(9*i*o), b = sum((i*i+o*o)), c=-num_svd
    # i_or_o = 1

    a = 0
    b = 0    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        
        for i in range(len(layers_out_channels)): #I==O
            a += (9*layers_out_channels[i]*layers_out_channels[i])*(2*repeat_list[i]-1)#sum of 3*3*I*O
            #each BasicBlock contains 2 3x3conv
            b += (layers_out_channels[i]*layers_out_channels[i] + layers_out_channels[i]*layers_out_channels[i])*(2*repeat_list[i]-1)#sum of I*I+O*O

        #parameters of decomposed conv1 of repeat0     
        if FLAGS.decom_conv1==True:#I=O/2
            for i in range(len(layers_out_channels)):
                a += 9*layers_out_channels[i]/2*layers_out_channels[i]#3*3*I*O
                b += layers_out_channels[i]/2*layers_out_channels[i]/2 + layers_out_channels[i]*layers_out_channels[i]#I*I+O*O

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 

        for i in range(len(layers_out_channels)):#I==O
            #each one contains 1 3x3conv
            a += (9*layers_out_channels[i]*layers_out_channels[i])*repeat_list[i]
            b += (layers_out_channels[i]*layers_out_channels[i] + layers_out_channels[i]*layers_out_channels[i])*repeat_list[i]

    num_svd = num_SVD(rank_rate_svd, layer_list, repeat_list, conv_list, layers_out_channels)
    c = -num_svd

    delta=b*b-4*a*c
    rank_rate_tt1 = (-b + np.sqrt(delta))/(2*a)
    rank_rate_tt2 = (-b - np.sqrt(delta))/(2*a)

    if delta<0 or rank_rate_tt1<=0:
        print('no')
        sys.exit(0)
    else:
        return rank_rate_tt1

#num of parameters in the TT based decomposed layers under rank_rate_tt
def num_TT(rank_rate_tt, layer_list, repeat_list, conv_list, layers_out_channels):
    #tt resnet34 分解层总参数量
    if rank_rate_tt<=0:
        assert(0)

    num = 0
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        #parameters of decomposed conv except conv1 of repeat0     
        for i in range(len(repeat_list)):
            num += (layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt)*2 +\
                   9*int(layers_out_channels[i]*rank_rate_tt)*int(layers_out_channels[i]*rank_rate_tt)) * (2*repeat_list[i]-1)#each BasicBlock contains 2 3x3conv

        #parameters of decomposed conv1 of repeat0     
        if FLAGS.decom_conv1==True:
            for i in range(len(repeat_list)):
                num += layers_out_channels[i]/2 * int(layers_out_channels[i]/2*rank_rate_tt)+\
                       layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt) +\
                       9*int(layers_out_channels[i]*rank_rate_tt)*int(layers_out_channels[i]/2*rank_rate_tt)

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        for i in range(len(repeat_list)):
            num += (layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt)+\
                   layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt) +\
                   9*int(layers_out_channels[i]*rank_rate_tt)*int(layers_out_channels[i]*rank_rate_tt))*repeat_list[i]#each one contains 1 3x3conv

    return num



'''
============================================================
                    JSVD
============================================================
'''

def JSVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):
    assert(len(layer_list)==len(repeat_list)==len(layers_out_channels))
    rank_rate_JSVD = calculate_JSVD_rate(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)

    JSVD_dict = {}
    JSVD_matmul_dict = {}
    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:
            if str(conv)=='1' and FLAGS.decom_conv1==False:
                name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
            else:
                name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

            parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]            

            parameter_stacked = np.vstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(parameter_list)])
            
            shape = parameter_list[-1].shape
            rank_JSVD = int(shape[3]*rank_rate_JSVD)
            U_2d_stacked, V_share_2d = decompose_SVD_2d(parameter_stacked, rank_JSVD)

            V_share_4d = chagne_2D_to_4d(V_share_2d, shape, rank_JSVD, 'V')
            JSVD_dict[ 'layer'+str(layer_name)+'.conv'+str(conv)+'.V_shared'] = V_share_4d

            (H, W, I, O) = parameter_list[-1].shape
            width = H*I
            for j, item in enumerate(name_list):
                if str(conv)=='1' and FLAGS.decom_conv1==True:
                    if j==0:
                        start = 0 
                        end = int(width/2)
                    else:
                        start = int(width/2) + width * (j-1)
                        end = int(width/2) + width * j
                else:
                    start = width*j
                    end =  width*(j+1)

                JSVD_dict[item+'.U_partly_shared'] = chagne_2D_to_4d(U_2d_stacked[start:end].copy(), parameter_list[j].shape, rank_JSVD, 'U')

    num_jsvd = num_JSVD(rank_rate_JSVD, layer_list, repeat_list, conv_list, layers_out_channels)
    CR = num_orig_all() / (num_undecomposed_parameter() + num_jsvd)
    return JSVD_dict, {'num_decomposed':num_jsvd,'num_undecomposed':num_undecomposed_parameter(),'rank_rate_calculate':rank_rate_JSVD, 'CR':CR}


def calculate_JSVD_rate(rank_rate_svd, layer_list, repeat_list, conv_list, layers_out_channels):
    #num_JSVD = num_U + num_V_shared
    #         = sum(W*I*int(r_JSVD*O))+H*O*int(r_JSVD*O)
    #         = (sum(W*I*O)+H*O*O) * r_JSVD
    #         = num_SVD
    #coefficient = sum(W*I*O)+H*O*O
    #r_JSVD = num_SVD/(W*I*O+H*O*O)

    coefficient = 0
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        for i in range(len(layers_out_channels)): #I==O
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*(2*repeat_list[i]-1)#U;except conv1 of repeat0;each BasicBlock contains 2 3x3conv
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*2#V_shared;each BasicBlock contains 2 3x3conv

        #parameters of decomposed conv1 of repeat0     
        if FLAGS.decom_conv1==True:
            for i in range(len(layers_out_channels)):
                coefficient +=3*layers_out_channels[i]/2*layers_out_channels[i] #U;I=O/2

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        for i in range(len(layers_out_channels)):#I==O
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*repeat_list[i]#U; each one contains 1 3x3conv
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]#V
    num_svd =  num_SVD(rank_rate_svd, layer_list, repeat_list, conv_list, layers_out_channels)
    rank_rate_JSVD = num_svd / coefficient

    return rank_rate_JSVD

#num of parameters in the JSVD based decomposed layers under rank_rate_JSVD
def num_JSVD(rank_rate_JSVD, layer_list, repeat_list, conv_list, layers_out_channels):
    assert(len(repeat_list)==len(layers_out_channels))

    num = 0
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        #parameters of decomposed conv except conv1 of repeat0     
        for i in range(len(layers_out_channels)):#I==O
            num += 3*layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_JSVD) * (2*repeat_list[i]-1)#U_independent; each BasicBlock contains 2 3x3conv
            num += 3*layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_JSVD) * 2#V_shared; each BasicBlock contains 2 3x3conv

        #parameters of decomposed conv1 of repeat0     
        if FLAGS.decom_conv1==True:
            for i in range(len(layers_out_channels)):
                num += 3*layers_out_channels[i]/2*int(layers_out_channels[i]*rank_rate_JSVD) #U; I=O/2

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        for i in range(len(layers_out_channels)):#I==O
            num += 3*layers_out_channels[i]*int(layers_out_channels[i]*rank_rate_JSVD)*repeat_list[i]#U_independent;each one contains 1 3x3conv
            num += 3*layers_out_channels[i]*int(layers_out_channels[i]*rank_rate_JSVD)#V_shared;each one contains 1 3x3conv

    return num


'''
============================================================
                    PCSVD
============================================================
'''
def PCSVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, rank_rate_shared, *useless_vars):
    rank_rate_independent = calculate_PCSVD_independent_rate(rank_rate_SVD, rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)

    for i in range(FLAGS.Iter_times):
        if i==0:
            partly_shared_matmul_dict = 0 #Initialized
        independent_dict, independent_matmul_dict = Independent_component_PCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, partly_shared_matmul_dict, rank_rate_independent)
        partly_shared_dict, partly_shared_matmul_dict = Partly_shared_component_PCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, independent_matmul_dict, rank_rate_shared)

    num_pcsvd = num_PCSVD(rank_rate_SVD, rank_rate_shared, rank_rate_independent, layer_list, repeat_list, conv_list, layers_out_channels)
    CR = num_orig_all() / (num_undecomposed_parameter() + num_pcsvd)

    record_dict = {'num_decomposed':num_pcsvd,
                    'num_undecomposed':num_undecomposed_parameter(),
                   'rank_rate_shared':rank_rate_shared,
                   'rank_rate_calculate':rank_rate_independent,
                   'rank_rate_calculate/rank_rate_shared': rank_rate_independent/rank_rate_shared,
                   'CR':CR }
    assert(independent_dict.keys()&partly_shared_dict.keys()==set())
    return {**independent_dict, **partly_shared_dict}, record_dict

def Independent_component_PCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, partly_shared_matmul_dict, rank_rate_independent):
    independent_dict = {}
    independent_matmul_dict = {}
    for i, layer_name in enumerate(layer_list):
        for repeat in range(repeat_list[i]):
            for conv in conv_list:
                if repeat==0 and str(conv)=='1' and FLAGS.decom_conv1==False:
                    pass
                else:
                    orig_weight_name = 'layer' + str(layer_name) + '.' + str(repeat) + '.conv' + str(conv) + '.weight'
                    partly_share_weight_name = 'layer' + str(layer_name) + '.' + str(repeat) + '.conv' + str(conv) +  '.U_matmul_V_shared'
                    indep_weight_name = 'layer' + str(layer_name) + '.' + str(repeat) + '.conv' + str(conv) + '.'
                    if partly_shared_matmul_dict==0:
                        U_4d, V_4d = decompose_SVD_4d(copy.deepcopy(npy_dict[orig_weight_name])-0, rank_rate_independent, shape_option='4d')
                    else:
                        U_4d, V_4d = decompose_SVD_4d(npy_dict[orig_weight_name]-partly_shared_matmul_dict[partly_share_weight_name], rank_rate_independent, shape_option='4d')
                    independent_dict[indep_weight_name+'U'] = U_4d
                    independent_dict[indep_weight_name+'V'] = V_4d

                    independent_matmul_dict[indep_weight_name+'U_matmul_V'] = recover_SVD(U_4d, V_4d)

    return independent_dict, independent_matmul_dict


def Partly_shared_component_PCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, independent_matmul_dict, rank_rate_shared, *useless_vars):
    partly_shared_dict = {}
    partly_shared_matmul_dict = {}
    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:
            if str(conv)=='1' and FLAGS.decom_conv1==False:
                name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
            else:
                name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

            if independent_matmul_dict==0: #Initialization
                assert(0) #not used currently
                parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]
            else:
                parameter_list = [ npy_dict[name +'.weight']- independent_matmul_dict[name + '.U_matmul_V'] for name in name_list]

            parameter_stacked = np.vstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(parameter_list)])

            shape = parameter_list[-1].shape
            rank_shared = int(shape[3]*rank_rate_shared)

            U_2d_stacked, V_share_2d = decompose_SVD_2d(parameter_stacked, rank_shared)
            V_share_4d = chagne_2D_to_4d(V_share_2d, shape, rank_shared, 'V')
            partly_shared_dict[ 'layer'+str(layer_name)+'.conv'+str(conv)+'.V_shared'] = V_share_4d

            (H,W,I,O) = parameter_list[-1].shape
            width = H*I
            for j, item in enumerate(name_list):
                if str(conv)=='1' and FLAGS.decom_conv1==True:
                    if j==0:
                        start = 0 
                        end = int(width/2)
                    else:
                        start = int(width/2) + width * (j-1)
                        end = int(width/2) + width * j
                else:
                    start = width*j
                    end =  width*(j+1)

                partly_shared_dict[item+'.U_partly_shared'] = chagne_2D_to_4d(U_2d_stacked[start:end].copy(), parameter_list[j].shape, rank_shared, 'U')
                partly_shared_matmul_dict[item+'.U_matmul_V_shared'] = recover_SVD(copy.deepcopy(partly_shared_dict[item+'.U_partly_shared']), V_share_4d.copy()) 
            assert(end==U_2d_stacked.shape[0])

    return partly_shared_dict, partly_shared_matmul_dict


def recover_SVD(U_4d, V_4d):
    h, _, i, r1 = U_4d.shape
    assert(_==1)
    _, w, r2, o = V_4d.shape
    assert(r1==r2)
    assert(_==1)

    U_2d = np.reshape(np.transpose(U_4d, [0,2,1,3]), [h*i, r1])
    V_2d = np.reshape(np.transpose(V_4d, [0,2,1,3]), [r2, w*o])
    W_2d = np.dot(U_2d, V_2d)
    W_4d = np.transpose(np.reshape(W_2d, [h,i,w,o]), [0,2,1,3])

    return W_4d


def calculate_PCSVD_independent_rate(rank_rate_SVD, rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels):
    num_partly_share = num_JSVD(rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)
    num_svd = num_SVD(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)
    num_independent = num_svd - num_partly_share
    if num_independent<=0:
        sys.exit(0)

    #num_independent = sum(U)+sum(V) = sum(3*I*O*rank_rate_independent)+sum(3*O*O*rank_rate_independent)
    #rank_rate_independent = num_independent/(sum(3*I*O)+sum(3*O*O))

    coefficient = 0
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        for i in range(len(layers_out_channels)): #I==O
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*(2*repeat_list[i]-1)#U_independent;except conv1 of repeat0；each BasicBlock contains 2 3x3conv
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*(2*repeat_list[i]-1)#V_independent;except conv1 of repeat0；each BasicBlock contains 2 3x3conv

        #parameters of decomposed conv1 of repeat0     
        if FLAGS.decom_conv1==True:
            for i in range(len(layers_out_channels)):
                coefficient +=3*layers_out_channels[i]/2*layers_out_channels[i] #U_independent;I=O/2
                coefficient += 3*layers_out_channels[i]*layers_out_channels[i]  #V_independent;

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        for i in range(len(layers_out_channels)):#I==O
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*repeat_list[i]#U_independent;I==O；each one contains 2 3x3conv
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*repeat_list[i]#V_independent;I==O；each one contains 2 3x3conv
    rank_rate_independent = num_independent / coefficient

    return rank_rate_independent

def num_PCSVD(rank_rate_SVD, rank_rate_shared, rank_rate_independent, layer_list, repeat_list, conv_list, layers_out_channels):
    num_share = num_JSVD(rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)

    # rank_rate_independent = calculate_PCSVD_independent_rate(rank_rate_SVD, rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)
    num_independent = num_SVD(rank_rate_independent, layer_list, repeat_list, conv_list, layers_out_channels)

    return num_share+num_independent




'''
============================================================
                    FCSVD
============================================================
'''
def FCSVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, rank_rate_shared, *useless_vars):
    rank_rate_independent = calculate_FCSVD_independent_rate(rank_rate_SVD, rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)

    for i in range(FLAGS.Iter_times):
        if i==0:
            shared_matmul_dict = 0 #Initialized
        independent_dict, independent_matmul_dict = Independent_component_FCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, shared_matmul_dict, rank_rate_independent)
        shared_dict, shared_matmul_dict = Shared_component_FCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, independent_matmul_dict, rank_rate_shared)

    num_fcsvd = num_FCSVD(rank_rate_shared, rank_rate_independent, layer_list, repeat_list, conv_list, layers_out_channels)
    CR = num_orig_all() / (num_undecomposed_parameter() + num_fcsvd)

    record_dict = {'num_decomposed':num_fcsvd,
                    'num_undecomposed':num_undecomposed_parameter(),
                   'rank_rate_shared':rank_rate_shared,
                   'rank_rate_calculate':rank_rate_independent,
                   'rank_rate_calculate/rank_rate_shared': rank_rate_independent/rank_rate_shared,
                   'CR':CR }
    assert(independent_dict.keys()&shared_dict.keys()==set())
    return {**independent_dict, **shared_dict}, record_dict


def calculate_FCSVD_independent_rate(rank_rate_SVD, rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels):
    num_share = num_fully_share(rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)
    num_svd = num_SVD(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels, _option=False)
    num_independent = num_svd - num_share
    if num_independent<=0:
        sys.exit(0)

    #num_independent = sum(U)+sum(V) = sum(3*I*O*rank_rate_independent)+sum(3*O*O*rank_rate_independent)
    #rank_rate_independent = num_independent/(sum(3*I*O)+sum(3*O*O))

    coefficient = 0
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        for i in range(len(layers_out_channels)): #I==O
            # assert(0)
            '''
            in this mode, the conv1 of repeat0 won't be decomposed
            '''
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*(2*repeat_list[i]-1)#U;except conv1 of repeat0
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*(2*repeat_list[i]-1)#V;except conv1 of repeat0

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        for i in range(len(layers_out_channels)):#I==O
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*repeat_list[i]#U;I==O
            coefficient += 3*layers_out_channels[i]*layers_out_channels[i]*repeat_list[i]#V;I==O
    rank_rate_independent = num_independent / coefficient

    return rank_rate_independent

def Independent_component_FCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, shared_matmul_dict, rank_rate_independent):
    independent_dict = {}
    independent_matmul_dict = {}
    for i, layer_name in enumerate(layer_list):
        for repeat in range(repeat_list[i]):
            for conv in conv_list: #In this mode, the repeat0.conv1 is not deocmposed
                if repeat==0 and str(conv)=='1':
                    pass
                else:
                    orig_weight_name = 'layer' + str(layer_name) + '.' + str(repeat) + '.conv' + str(conv) + '.weight'
                    shared_weight_name = 'layer' + str(layer_name) + '.conv' + str(conv) +  '.U_shared_matmul_V_shared'
                    indep_weight_name = 'layer' + str(layer_name) + '.' + str(repeat) + '.conv' + str(conv) + '.'

                    if shared_matmul_dict==0:
                        U_4d, V_4d = decompose_SVD_4d(copy.deepcopy(npy_dict[orig_weight_name])-0, rank_rate_independent, shape_option='4d')
                    else:
                        U_4d, V_4d = decompose_SVD_4d(npy_dict[orig_weight_name]-shared_matmul_dict[shared_weight_name], rank_rate_independent, shape_option='4d')

                    independent_dict[indep_weight_name+'U'] = U_4d
                    independent_dict[indep_weight_name+'V'] = V_4d

                    independent_matmul_dict[indep_weight_name+'U_matmul_V'] = recover_SVD(U_4d, V_4d)

    return independent_dict, independent_matmul_dict

def Shared_component_FCSVD(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, independent_matmul_dict, rank_rate_shared):
    shared_dict = {}
    shared_matmul_dict = {}

    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:
            if str(conv)=='1': #In this mode, the repeat0.conv1 is not deocmposed
                name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
            else:
                name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

            if independent_matmul_dict==0: #Initialization
                assert(0) #not used currently
                parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]
            else:
                parameter_list = [ npy_dict[name +'.weight']- independent_matmul_dict[name + '.U_matmul_V'] for name in name_list]

            U_shared_4d, V_shared_4d = decompose_SVD_4d(np.mean(parameter_list, axis=0), rank_rate_shared, '4d')

            shared_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.U_shared'] = U_shared_4d
            shared_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.V_shared'] = V_shared_4d

            shared_matmul_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.U_shared_matmul_V_shared'] = recover_SVD(U_shared_4d, V_shared_4d)

    return shared_dict, shared_matmul_dict



def num_fully_share(rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels):
    assert(len(repeat_list)==len(layers_out_channels))

    num = 0
    # don't decompose layer1; don't decompose conv1 of repeat0
    if FLAGS.model in ['resnet18', 'resnet34']:
        assert(len(repeat_list)==3)
        #parameters of decomposed conv
        for i in range(len(layers_out_channels)):#I==O
            num += 3*layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_shared)*2#U_shared; each BasicBlock contains 2 3x3conv
            num += 3*layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_shared)*2 #V_shared; each BasicBlock contains 2 3x3conv

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        for i in range(len(layers_out_channels)):#I==O
            num += 3*layers_out_channels[i]*int(layers_out_channels[i]*rank_rate_shared)#U_shared; each one contains 1 3x3conv
            num += 3*layers_out_channels[i]*int(layers_out_channels[i]*rank_rate_shared)#V_shared; each one contains 1 3x3conv

    return num

def num_FCSVD(rank_rate_shared, rank_rate_independent, layer_list, repeat_list, conv_list, layers_out_channels):
    # rank_rate_independent = calculate_FCSVD_independent_rate(rank_rate_SVD, rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels):
    num_independent = num_SVD(rank_rate_independent, layer_list, repeat_list, conv_list, layers_out_channels,_option=False)

    num_share = num_fully_share(rank_rate_shared, layer_list, repeat_list, conv_list, layers_out_channels)
    return num_independent + num_share



'''
============================================================
                   PCSVD+FCSVD for ResNet34
============================================================
'''
def PCSVD_FCSVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, rank_rate_shared, *useless_vars):


    '''=================PCSVD for conv1 of layer2——4'================='''

    #num for independent
    num_conv1_svd = 0
    for i,item in enumerate(layers_out_channels):
        num_conv1_svd += 3*item/2*int(item*rank_rate_SVD) + 3*item*int(item*rank_rate_SVD)#repeat0.conv1
        num_conv1_svd += (3*item*int(item*rank_rate_SVD) + 3*item*int(item*rank_rate_SVD))*(repeat_list[i]-1)#other conv1

    #num for shared
    num_shared_pcsvd = 0
    for i,item in enumerate(layers_out_channels):
        num_shared_pcsvd += 3*item/2 * int(item*rank_rate_shared)#U of repeat0.conv1
        num_shared_pcsvd += 3*item * int(item*rank_rate_shared)*(repeat_list[i]-1)# U of other conv1
        num_shared_pcsvd += 3*item * int(item*rank_rate_shared)#V_shared

    num_independent_pcsvd = num_conv1_svd - num_shared_pcsvd

    #calculate rank_rate_independent
    coefficient_independent_pcsvd = 0
    for i,item in enumerate(layers_out_channels):
        coefficient_independent_pcsvd += 3*item/2*item + 3*item*item#U_independent + V_independent of  repeat0.conv1
        coefficient_independent_pcsvd += 3*item*item*(repeat_list[i]-1)*2#U_independent + V_independent;except conv1 of repeat0
    rank_rate_independent_pcsvd = num_independent_pcsvd/coefficient_independent_pcsvd

    #parameter optimizaition iteratively
    for i in range(FLAGS.Iter_times):
        if i==0:
            PCSVD_shared_matmul_dict = 0 #Initialized
        PCSVD_independent_dict, PCSVD_independent_matmul_dict = Independent_component_PCSVD(npy_dict, layer_list, repeat_list, [1], layers_out_channels, PCSVD_shared_matmul_dict, rank_rate_independent_pcsvd)
        PCSVD_shared_dict, PCSVD_shared_matmul_dict = Partly_shared_component_PCSVD(npy_dict, layer_list, repeat_list, [1], layers_out_channels, PCSVD_independent_matmul_dict, rank_rate_shared)
    pcsvd_dict = {**PCSVD_independent_dict, **PCSVD_shared_dict}

    #calculate num and CR
    num_independent_pcsvd = 0
    for i,item in enumerate(layers_out_channels):
        num_independent_pcsvd += 3*(item/2)*int(item*rank_rate_independent_pcsvd) + 3*item*int(item*rank_rate_independent_pcsvd)#repeat0.conv1
        num_independent_pcsvd += 3*item*int(item*rank_rate_independent_pcsvd)*2 * (repeat_list[i]-1)#other conv1
               
    num_shared_pcsvd = 0
    for i,item in enumerate(layers_out_channels):#I==O
        num_shared_pcsvd += 3*item/2*int(item*rank_rate_shared) #U_partly_shared of repeat0.conv1; I=O/2
        num_shared_pcsvd += 3*item * int(item*rank_rate_shared) * (repeat_list[i]-1)#U_partly_shared of other.conv1; I=O
        num_shared_pcsvd += 3*item * int(item*rank_rate_shared)#V_shared
                    
    num_pcsvd = num_independent_pcsvd + num_shared_pcsvd
    

    '''=================FCSVD for conv2 of layer2——4================='''
    #num for independent
    num_conv2_svd = 0
    for i,item in enumerate(layers_out_channels):
        num_conv2_svd += (3*item*int(item*rank_rate_SVD) + 3*item*int(item*rank_rate_SVD))*repeat_list[i]#conv2
    
    #num for shared
    num_shared_fcsvd = 0
    for i,item in enumerate(layers_out_channels):
        num_shared_fcsvd += 3*item*int(item*rank_rate_shared)# U_shared of  conv2
        num_shared_fcsvd += 3*item*int(item*rank_rate_shared)#V_shared

    num_independent_fcsvd = num_conv2_svd - num_shared_fcsvd

    #calculate rank_rate_independent
    coefficient_independent_fcsvd = 0
    for i,item in enumerate(layers_out_channels):
        coefficient_independent_fcsvd += 3*item*item*repeat_list[i]*2#U_independent + V_independent of conv2
    rank_rate_independent_fcsvd = num_independent_fcsvd/coefficient_independent_fcsvd

    #parameter optimizaition iteratively
    for i in range(FLAGS.Iter_times):
        if i==0:
            FCSVD_shared_matmul_dict = 0 #Initialized
        FCSVD_independent_dict, FCSVD_independent_matmul_dict = Independent_component_FCSVD(npy_dict, layer_list, repeat_list, [2], layers_out_channels, FCSVD_shared_matmul_dict, rank_rate_independent_fcsvd)
        FCSVD_shared_dict, FCSVD_shared_matmul_dict = Shared_component_FCSVD(npy_dict, layer_list, repeat_list, [2], layers_out_channels, FCSVD_independent_matmul_dict, rank_rate_shared)
    fcsvd_dict = {**FCSVD_independent_dict, **FCSVD_shared_dict}

    #calculate num and CR
    num_independent_fcsvd = 0
    for i,item in enumerate(layers_out_channels):
        num_independent_fcsvd += 3*item*int(item*rank_rate_independent_fcsvd)*2 *repeat_list[i]#U,V of conv2

    num_shared_fcsvd = 0
    for i,item in enumerate(layers_out_channels):#I==O
        num_shared_fcsvd += 3*item * int(item*rank_rate_shared)#U_shared of conv2
        num_shared_fcsvd += 3*item * int(item*rank_rate_shared)#V_shared of conv2

    num_fcsvd = num_independent_fcsvd + num_shared_fcsvd
    CR = num_orig_all() / (num_undecomposed_parameter() + num_pcsvd + num_fcsvd)
    '''record_dict'''
    record_dict = {'num_decomposed':num_pcsvd+num_fcsvd,
                'num_undecomposed':num_undecomposed_parameter(),
               'rank_rate_shared':rank_rate_shared,
               'rank_rate_calculate':(rank_rate_independent_pcsvd+rank_rate_independent_fcsvd)/2,
               'rank_rate_calculate/rank_rate_shared': (rank_rate_independent_pcsvd+rank_rate_independent_fcsvd)/2/rank_rate_shared,
               'CR':CR }
    assert(pcsvd_dict.keys()&fcsvd_dict.keys()==set())
    return {**pcsvd_dict, **fcsvd_dict}, record_dict


'''
============================================================
                Functions about CR calculation
============================================================
'''
def num_orig_all():

    orig_repeat_list = orig_Repeat_list[FLAGS.model]
    layers_out_channels = [64,128,256,512]

    kernal_size_first_conv = 3 if FLAGS.dataset in ['cifar10', 'cifar100'] else 7
    classes = 10 if FLAGS.dataset=='cifar10' else 100 if FLAGS.dataset=='cifar100' else 1000
    out_dimension = 512 if FLAGS.model in ['resnet18', 'resnet34'] else 2048

    #parameters of first conv / fc weight, fc bias
    num = kernal_size_first_conv*kernal_size_first_conv*3*64 + out_dimension*classes + classes

    #BasicBlock
    if FLAGS.model in ['resnet18', 'resnet34']:
        num += 64*128 + 128*256 + 256*512 #shortcut
        num += 3*3*64*64 + 3*3*64*128 + 3*3*128*256 + 3*3*256*512 #repeat0.conv1 of each layer
        for i in range(len(orig_repeat_list)):#3x3conv
            num += 3*3*layers_out_channels[i]*layers_out_channels[i]*(2*orig_repeat_list[i]-1)

    #BottleNeck:
    else:
        num += 64*256 + 256*512 + 512*1024 + 1024*2048 #shortcut
        num += 64*64 + 3*3*64*64 + 3*3*64*64 + 64*256 + (256*64 + 3*3*64*64 + 64*256)*(orig_repeat_list[0]-1)#layer1
        num += 256*128 + 3*3*128*128 + 128*512 + (512*128 + 3*3*128*128 + 128*512)*(orig_repeat_list[1]-1)#layer2
        num += 512*256 + 3*3*256*256 + 256*1024 + (1024*256 + 3*3*256*256 + 256*1024)*(orig_repeat_list[2]-1)#layer3
        num += 1024*512 + 3*3*512*512 + 512*2048 + (2048*512 + 3*3*512*512 + 512*2048)*(orig_repeat_list[3]-1)#layer4

    return num


def num_undecomposed_parameter():

    orig_repeat_list = orig_Repeat_list[FLAGS.model]
    layers_out_channels = [64,128,256,512]

    kernal_size_first_conv = 3 if FLAGS.dataset in ['cifar10', 'cifar100'] else 7
    classes = 10 if FLAGS.dataset=='cifar10' else 100 if FLAGS.dataset=='cifar100' else 1000
    out_dimension = 512 if FLAGS.model in ['resnet18', 'resnet34'] else 2048

    #parameters of first conv / fc weight, fc bias
    num = kernal_size_first_conv*kernal_size_first_conv*3*64 + out_dimension*classes + classes

    #BasicBlock
    if FLAGS.model in ['resnet18', 'resnet34']:
        num += 64*128 + 128*256 + 256*512 #shortcut
        num += (3*3*64*64*2) * orig_repeat_list[0] #layer1
        if FLAGS.decom_conv1==False:
            num += 3*3*64*128 + 3*3*128*256 +3*3*256*512 #conv1 of repeat0

    #BottleNeck: don't decompose 1x1 layer
    else:
        num += 64*256 + 256*512 + 512*1024 + 1024*2048 #shortcut
        num += 64*64 + 64*256 + (256*64 + 64*256)*(orig_repeat_list[0]-1) #layer1
        num += 256*128 + 128*512 + (512*128 + 128*512)*(orig_repeat_list[1]-1) #layer2
        num += 512*256 + 256*1024 + (1024*256 + 256*1024)*(orig_repeat_list[2]-1) #layer3
        num += 1024*512 + 512*2048 + (2048*512 + 512*2048)*(orig_repeat_list[3]-1) #layer4

    return num



'''
============================================================
                    Testing Code
============================================================
'''
# flags.DEFINE_integer('Iter_times', 2, 'Iterations times for parameters optimization ')
# flags.DEFINE_string('rank_rate_SVD', None, 'rank_rate_SVD')
# flags.DEFINE_string('rank_rate_shared', None, 'rank_rate_shared')
# flags.DEFINE_string('method', None, "'SVD', 'TT', 'JSVD', 'PCSVD', 'FCSVD'")
# flags.DEFINE_bool('decom_conv1', False, 'decompose xx.0.conv1  or not ')
# flags.DEFINE_string('model', 'resnet34', 'model name')
# flags.DEFINE_string('dataset', 'imagenet', 'imagenet/cifar10/cifar100')


# dic, record_dict = get_parameter()
# # print('rank_rate_SVD', FLAGS.rank_rate_SVD)
# # print('rank_rate_shared', FLAGS.rank_rate_shared)
# print('CR', record_dict['CR'])
# print('num_decomposed_parameter ', record_dict['num_decomposed'])
# print('num_undecomposed_parameter', record_dict['num_undecomposed'])
# # print('\n')
#python get_parameter.py --rank_rate_SVD=  --rank_rate_shared=  --method=  --decom_conv1=  --model= --dataset=
# python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.1  --method=SVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10
# python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.1  --method=TT  --decom_conv1=True  --model=resnet34 --dataset=cifar10
# python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.1  --method=JSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10
# python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.1  --method=PCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10
# python get_parameter.py --rank_rate_SVD=0.5  --rank_rate_shared=0.1  --method=FCSVD  --decom_conv1=True  --model=resnet34 --dataset=cifar10

