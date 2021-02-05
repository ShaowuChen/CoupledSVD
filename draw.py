import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import os
'''
============================================================================================================
                                            Cifar10
============================================================================================================
'''

st = 1
un = 1

tr = 1
pr = 1  

p1 = 1
p2 = 1
p3 = 1

if st+un==st:
    fig = plt.figure(1)
    ax2 = plt.subplot(1,1,1)
    ax2.set_title('Structured Pruning')
    ax2.set_xlabel("Connected Ratio")
    ax2.set_ylabel("Acc/%")
elif st+un==un:
    fig = plt.figure(1)
    ax1 = plt.subplot(1,1,1)
    ax1.set_title('Unstructured Pruning')
    ax1.set_xlabel("Connected Ratio")
    ax1.set_ylabel("Acc/%")

else:
    fig = plt.figure(1)
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)

    ax1.set_title('Unstructured Pruning')
    ax1.set_xlabel("Connected Ratio")
    ax1.set_ylabel("Acc/%")


    ax2.set_title('Structured Pruning')
    ax2.set_xlabel("Connected Ratio")
    ax2.set_ylabel("Acc/%")


def path(stru_or_unstru, method1, cr_traditional, method2='None', cr_share='None', date='20210127', dataset='cifar10', model='resnet34', extra_mark_path=''):


    root_path = '/home/test01/sambashare/sdd/Coupled_Pruning/ckpt'

    if stru_or_unstru=='structured':
        if method1=='traditional':
            ckpt_path1 = '/'+stru_or_unstru +extra_mark_path  + '/' + method1 +'/'+dataset+'/'+model
        else:
            ckpt_path1 = '/'+stru_or_unstru +extra_mark_path  + '/' + method1 +'/'+method2+ '/'+dataset+'/'+model

        tr_list = cr_traditional.split('/')
        sh_list = cr_share.split('/')
        if cr_share=='None':
            ckpt_path = root_path + '/' + date + ckpt_path1  +'/'+'tr'+tr_list[0]+'_'+tr_list[1] + '_sh'+cr_share
        else:
            ckpt_path = root_path + '/' + date + ckpt_path1  +'/'+'tr'+tr_list[0]+'_'+tr_list[1] + '_sh'+sh_list[0]+'_'+sh_list[1]

    else:
        cr_traditional = '%.6f'%cr_traditional if method1=='traditional' else '%s'%str(cr_traditional)
        cr_share = cr_share if (cr_share=='None' or cr_share==None) else '%.6f'%cr_share if method1=='traditional' else '%s'%str(cr_share)
        # cr_share = cr_share if (cr_share=='None' or cr_share==None) else '%f'%cr_share

        if method1=='traditional':
            ckpt_path1 = '/'+stru_or_unstru +extra_mark_path  + '/' + method1 +'/'+dataset+'/'+model
        else:
            ckpt_path1 = '/'+stru_or_unstru +extra_mark_path  + '/' + method1 +'/'+method2+ '/'+dataset+'/'+model

        ckpt_path = root_path + '/' + date + ckpt_path1  +'/'+'tr'+cr_traditional + '_sh'+cr_share


    
    overall_log_path = root_path + '/' + date + ckpt_path1 + '/overall.log'
    mark_overall = ckpt_path1  +'/'+'tr'+cr_traditional + '_sh'+cr_share+'\n'
    return ckpt_path, overall_log_path, mark_overall

def find_and_write(path):
    boolen = os.path.exists(path+'/log.npy')
    if boolen:
        f = np.load(path+'/log.npy', allow_pickle=True).item()
        y = f['ave_TOP1_5'][0]
        print(y)
    else:
        print(path)
        y = None
    return y


################## traditioanl #############################

################ unstructured ###################

#### without consider cr_share ######
tr_un_wo0_x = []
tr_un_wo0_y = []
cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]
for i, item in enumerate(cr_traditional):
    ckpt_path, overall_log_path, mark_overall = path('unstructured', 'traditional', item)
    y = find_and_write(ckpt_path)
    if y!=None:
        tr_un_wo0_x.append(item)
        tr_un_wo0_y.append(y)    
print('\n')

#if un and tr:
    #ax1.plot(tr_un_wo0_x, tr_un_wo0_y, color = 'r', marker="*", label="tradotional")

############  consider cr_share ######
cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]

for i in range(3):
    globals()['tr_un_w%d_x'%i]=[]
    globals()['tr_un_w%d_y'%i]=[]

for i, item in enumerate(cr_traditional):

    cr_share = [item/num for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('unstructured', 'traditional', item, cr_share=item2)
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['tr_un_w%d_x'%j].append(item)
            globals()['tr_un_w%d_y'%j].append(y)
    
print('\n')
if un and tr:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax1.plot(globals()['tr_un_w%d_x'%i], globals()['tr_un_w%d_y'%i], color = 'g', marker=marker[i], label="traditional_crshare/%d"%divi[i])


################## structured ###################

##### without consider cr_share ######
# tr_st_wo0_x = []
# tr_st_wo0_y = []
# cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
# for i, item in enumerate(cr_traditional):
#     ckpt_path, overall_log_path, mark_overall = path('structured', 'traditional', item)
#     y = find_and_write(ckpt_path)
#     if y!=None:
#         tr_st_wo0_x.append(eval(item))
#         tr_st_wo0_y.append(y)   
# if st and tr:
    # ax2.plot(tr_st_wo0_x, tr_st_wo0_y, color = 'r', marker="*", label='traditional')




print('\n')
############  consider cr_share ######
for i in range(3):
    globals()['tr_st_w%d_x'%i]=[]
    globals()['tr_st_w%d_y'%i]=[]

cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):

    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('structured', 'traditional', item, cr_share=item2)
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['tr_st_w%d_x'%j].append(eval(item))
            globals()['tr_st_w%d_y'%j].append(y)

print('\n')
if st and tr:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax2.plot(globals()['tr_st_w%d_x'%i], globals()['tr_st_w%d_y'%i], color = 'g', marker=marker[i], label="traditional_crshare/%d"%divi[i])


################## Proposed#############################

################ unstructured ###################

################# p11 ##################
for i in range(3):
    globals()['pr_un_p11_%d_x'%i]=[]
    globals()['pr_un_p11_%d_y'%i]=[]

cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]
for i, item in enumerate(cr_traditional):

    cr_share = [item/num for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('unstructured', 'proposed', item, cr_share=item2, method2='p1')

        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_un_p11_%d_x'%j].append(item)
            globals()['pr_un_p11_%d_y'%j].append(y)
print('\n')
if un and pr and p1:    
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax1.plot(globals()['pr_un_p11_%d_x'%i], globals()['pr_un_p11_%d_y'%i], color = 'b', marker=marker[i], label="p11_crshare/%d"%divi[i])

# ################# p12 ##################
# for i in range(3):
#     globals()['pr_un_p12_%d_x'%i]=[]
#     globals()['pr_un_p12_%d_y'%i]=[]

# cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]
# for i, item in enumerate(cr_traditional):

#     cr_share = [item/num for num in [10,5,2]]
#     for j, item2 in enumerate(cr_share):
#         ckpt_path, overall_log_path, mark_overall = path('unstructured', 'proposed', item, cr_share=item2, method2='p11')

#         y = find_and_write(ckpt_path)
#         if y!=None:
#             globals()['pr_un_p12_%d_x'%j].append(item)
#             globals()['pr_un_p12_%d_y'%j].append(y)
# print('\n')
# if un and pr and p1:    
#     for i in range(3):
#         marker = ['*', "p", "+"]
#         divi = [10, 5, 2]
#         ax1.plot(globals()['pr_un_p12_%d_x'%i], globals()['pr_un_p12_%d_y'%i], color = 'blueviolet', marker=marker[i], label="p12_crshare/%d"%divi[i])




################# p2 ##################
for i in range(3):
    globals()['pr_un_p2_%d_x'%i]=[]
    globals()['pr_un_p2_%d_y'%i]=[]

cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]
for i, item in enumerate(cr_traditional):
    cr_share = [item/num for num in [10,5,2]]

    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('unstructured', 'proposed', item, cr_share=item2, method2='p2')

        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_un_p2_%d_x'%j].append(item)
            globals()['pr_un_p2_%d_y'%j].append(y)
print('\n')
if un and pr and p2:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax1.plot(globals()['pr_un_p2_%d_x'%i], globals()['pr_un_p2_%d_y'%i], color = 'k', marker=marker[i], label="p2_crshare/%d"%divi[i])



################# p3 ##################
for i in range(3):
    globals()['pr_un_p3_%d_x'%i]=[]
    globals()['pr_un_p3_%d_y'%i]=[]

cr_traditional = [0.005, 0.01, 0.05,  0.2, 0.5]
for i, item in enumerate(cr_traditional):
    cr_share = [item/num for num in [10,5,2]]

    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('unstructured', 'proposed', item, cr_share=item2, method2='p3')
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_un_p3_%d_x'%j].append(item)
            globals()['pr_un_p3_%d_y'%j].append(y)

if un and pr and p3:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax1.plot(globals()['pr_un_p3_%d_x'%i], globals()['pr_un_p3_%d_y'%i], color = 'gold', marker=marker[i], label="p3_crshare/%d"%divi[i])
    print('\n')
# ################## structured ###################

################# p11 ##################.
for i in range(3):
    globals()['pr_st_p11_%d_x'%i]=[]
    globals()['pr_st_p11_%d_y'%i]=[]

cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]

for i, item in enumerate(cr_traditional):

    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('structured', 'proposed', item, cr_share=item2, method2='p11', date='20210128')
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_st_p11_%d_x'%j].append(eval(item))
            globals()['pr_st_p11_%d_y'%j].append(y)
print('\n')
if st and pr and p1:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax2.plot(globals()['pr_st_p11_%d_x'%i], globals()['pr_st_p11_%d_y'%i], color = 'b', marker=marker[i], label="p11_crshare/%d"%divi[i])
    print('\n')

################# p12 ##################.
for i in range(3):
    globals()['pr_st_p12_%d_x'%i]=[]
    globals()['pr_st_p12_%d_y'%i]=[]

cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]

for i, item in enumerate(cr_traditional):

    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('structured', 'proposed', item, cr_share=item2, method2='p12', date='20210128')
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_st_p12_%d_x'%j].append(eval(item))
            globals()['pr_st_p12_%d_y'%j].append(y)
print('\n')
if st and pr and p1:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax2.plot(globals()['pr_st_p12_%d_x'%i], globals()['pr_st_p12_%d_y'%i], color = 'blueviolet', marker=marker[i], label="p12_crshare/%d"%divi[i])
    print('\n')

################# p2 ##################
for i in range(3):
    globals()['pr_st_p2_%d_x'%i]=[]
    globals()['pr_st_p2_%d_y'%i]=[]

cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):

    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('structured', 'proposed', item, cr_share=item2, method2='p2')
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_st_p2_%d_x'%j].append(eval(item))
            globals()['pr_st_p2_%d_y'%j].append(y)
if st and pr and p2:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax2.plot(globals()['pr_st_p2_%d_x'%i], globals()['pr_st_p2_%d_y'%i], color = 'k', marker=marker[i], label="p2_crshare/%d"%divi[i])
    print('\n')

print('\n')
################# p3 ##################
for i in range(3):
    globals()['pr_st_p3_%d_x'%i]=[]
    globals()['pr_st_p3_%d_y'%i]=[]

cr_traditional = ['4/128', '8/128', '32/128', '64/128', '96/128' ]
for i, item in enumerate(cr_traditional):

    cr_share = [str(np.ceil(eval(item.split('/')[0])/num).astype(int))+'/128' for num in [10,5,2]]
    for j, item2 in enumerate(cr_share):
        ckpt_path, overall_log_path, mark_overall = path('structured', 'proposed', item, cr_share=item2, method2='p3')
        y = find_and_write(ckpt_path)
        if y!=None:
            globals()['pr_st_p3_%d_x'%j].append(eval(item))
            globals()['pr_st_p3_%d_y'%j].append(y)

if st and pr and p3:
    for i in range(3):
        marker = ['*', "p", "+"]
        divi = [10, 5, 2]
        ax2.plot(globals()['pr_st_p3_%d_x'%i], globals()['pr_st_p3_%d_y'%i], color = 'gold', marker=marker[i], label="p3_crshare/%d"%divi[i])
    print('\n')

Xfmt='%.1f%%'
plt.legend(loc='lower right')
# xticks = mtick.FormatStrFormatter(Xfmt)





plt.show()
