# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
import csv
# ourself libs
from model_initiation import model_init
from data_preprocess import data_set
import math
from MoDe import unlearning_MoDe

from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain


def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    # all_global_models, all_client_models 为保存起来所有的old FL models
    print(5 * "#" + "  Federated Learning Start" + 5 * "#")
    FL_params.if_retrain = False
    std_time = time.time()
    init_global_model.train()
    global_model = copy.deepcopy(init_global_model)
    old_GMs, old_CMs = FL_Train(global_model, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_learn = (std_time - end_time)
    file_name = 'memory_list/old_CMs_list[' + str(FL_params.global_epoch) + ' ' + str(FL_params.N_client) + ' ' + str(
        FL_params.data_name) + '].csv'
    with open(file_name, 'w', newline='') as file:
        # 使用csv.writer将列表写入CSV文件
        writer = csv.writer(file)
        writer.writerow(old_CMs)
        writer.writerow(old_GMs)
    print(
        'start time ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(std_time)) + '    end time ' + time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('total time ' + str(int(time_learn)) + ' s')
    print(5 * "#" + "  Federated Learning End" + 5 * "#")

    print('\n')
    """
    4.2 unlearning  a client，Federated Unlearning 进行遗忘学习
    """
    print(5 * "#" + "  Federated Unlearning Start  " + 5 * "#")
    std_time = time.time()
    # 将参数IF_unlearning =True设置为global_train_once跳过被遗忘的用户，节省计算时间
    FL_params.if_unlearning = True
    # 设置参数forget_client_IDx来标记需要遗忘的用户IDX
    # FL_params.forget_client_idx = 2
    temp_old_GMs = copy.deepcopy(old_GMs)
    unlearn_GMs = unlearning(temp_old_GMs, old_CMs, client_loaders, test_loader, FL_params)

    end_time = time.time()
    time_unlearn = (std_time - end_time)
    print(5 * "#" + "  Federated Unlearning End  " + 5 * "#")

    # """
    # 4.2 unlearning  a client，Federated Unlearning 进行遗忘学习
    # """
    # print(5 * "#" + "  MoDe Unlearning Start  " + 5 * "#")
    # std_time = time.time()
    # # 将参数IF_unlearning =True设置为global_train_once跳过被遗忘的用户，节省计算时间
    # FL_params.if_unlearning = True
    # # 设置参数forget_client_IDx来标记需要遗忘的用户IDX
    # # FL_params.forget_client_idx = 2
    # need_forget_model = copy.deepcopy(old_GMs[-1])
    # unlearn_MoDe = unlearning_MoDe(need_forget_model, client_loaders, FL_params)
    #
    # end_time = time.time()
    # time_unlearn = (std_time - end_time)
    # print(" MoDe time consuming = {} secods".format(-time_unlearn))
    # print(5 * "#" + "  MoDe Unlearning End  " + 5 * "#")

    print('\n')
    """4.3 遗忘一个客户端，联邦学习不校准"""
    # print(5*"#"+"  Federated Unlearning without Calibration Start  "+5*"#")
    # std_time = time.time()
    # uncali_unlearn_GMs = unlearning_without_cali(old_GMs, old_CMs, FL_params)
    # end_time = time.time()
    # time_unlearn_no_cali = (std_time - end_time)
    # print(5*"#"+"  Federated Unlearning without Calibration End  "+5*"#")

    print(5 * "#" + " My Federated Unlearning Start  " + 5 * "#")
    # std_time = time.time()
    # model_forget, model_forget_withRe, time_total_noRe, time_total_Re
    need_forget_model = copy.deepcopy(old_GMs[-1])
    need_forget_model.train()
    my_unlearn, my_unlearn_withRe, my_unlearn_withRe2, time_unlearn_my_noRe, time_unlearn_my_Re, time_unlearn_my_Re2 \
        = unlearning_my(need_forget_model,client_loaders, test_loader, FL_params)
    # end_time = time.time()
    # time_unlearn_my = (std_time - end_time)
    print(5 * "#" + " My Federated Unlearning End  " + 5 * "#")

    FL_params.if_retrain = True
    if (FL_params.if_retrain):
        print('\n')
        print(5 * "#" + "  Federated Retraining Start  " + 5 * "#")
        std_time = time.time()
        # FL_params.N_client = FL_params.N_client - 1
        # client_loaders.pop(FL_params.forget_client_idx)
        # retrain_GMs, _ = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
        re_global_model = copy.deepcopy(init_global_model)
        retrain_GMs = FL_Retrain(re_global_model, client_loaders, test_loader, FL_params)
        end_time = time.time()
        time_retrain = (std_time - end_time)
        print(5 * "#" + "  Federated Retraining End  " + 5 * "#")
    else:
        print('\n')
        print(5 * "#" + "  No Retraining " + 5 * "#")
        retrain_GMs = list()

    # my_unlearn = unlearn_MoDe

    print(" Learning time consuming = {} secods".format(-time_learn))
    print(" Unlearning time consuming = {} secods".format(-time_unlearn))
    # print(" Unlearning no Cali time consuming = {} secods".format(-time_unlearn_no_cali))
    print(" Retraining time consuming = {} secods".format(-time_retrain))
    print(" My Unlearning without Retrain time consuming = {} secods".format(-time_unlearn_my_noRe))
    print(" My Unlearning with Retrain time consuming = {} secods".format(-time_unlearn_my_Re))
    print(" My Unlearning with Retrain time consuming = {} secods".format(-time_unlearn_my_Re2))
    # time_list = [-time_unlearn_my_noRe, -time_unlearn_my_Re, -time_unlearn_my_Re2,
    #              -time_unlearn, -time_unlearn_no_cali,
    #              -time_retrain, -time_learn]
    time_list = [-time_unlearn_my_noRe, -time_unlearn_my_Re, -time_unlearn_my_Re2,
                 -time_unlearn,
                 -time_retrain, -time_learn]

    # return old_GMs, unlearn_GMs, uncali_unlearn_GMs, retrain_GMs, \
    #     my_unlearn, my_unlearn_withRe, my_unlearn_withRe2, time_list
    return old_GMs, unlearn_GMs, retrain_GMs, \
        my_unlearn, my_unlearn_withRe, my_unlearn_withRe2, time_list
    # return old_GMs, unlearn_GMs, uncali_unlearn_GMs, my_unlearn


def unlearning(old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    """


    Parameters
    ----------
    old_global_models : list of DNN models
        In standard federated learning, all the global models from each round of training are saved.
    old_client_models : list of local client models
        In standard federated learning, the server collects all user models after each round of training.
    client_data_loaders : list of torch.utils.data.DataLoader
        This can be interpreted as each client user's own data, and each Dataloader corresponds to each user's data
    test_loader : torch.utils.data.DataLoader
        The loader for the test set used for testing
    FL_params : Argment（）
        The parameter class used to set training parameters

    Returns
    -------
    forget_global_model : One DNN model that has the same structure but different parameters with global_moedel
        DESCRIPTION.

    """

    if (FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')

    if (not (FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(
            range(FL_params.N_client)))
    if (FL_params.unlearn_interval == 0 or FL_params.unlearn_interval > FL_params.global_epoch):
        raise ValueError(
            'FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')

    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)

    forget_client = FL_params.forget_client_idx
    for ii in range(FL_params.global_epoch):  # 把每一个轮次的被遗忘客户端都忘掉，相当于刷新了一遍old_client_models
        temp = old_client_models[ii * FL_params.N_client: ii * FL_params.N_client + FL_params.N_client]  # 长度为一个N_client
        temp.pop(forget_client)  # 在Unlearn过程中，会弹出被遗忘用户保存的模型
        old_client_models.append(temp)  # 一个append让old_client_models长度+1，类型是list（9个Net_mnist组成）而不是前面200个的Net_mnist型
    old_client_models = old_client_models[-FL_params.global_epoch:]  # 从220个中取后20个list：20轮未被遗忘的客户端的每一次的模型

    GM_intv = np.arange(0, FL_params.global_epoch + 1, FL_params.unlearn_interval,
                        dtype=np.int16())  # 返回一个有终点和起点的固定步长的排列 [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    CM_intv = GM_intv - 1  # [-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    CM_intv = CM_intv[1:]  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

    selected_GMs = [old_global_models[ii] for ii in GM_intv]  # 长度21
    selected_CMs = [old_client_models[jj] for jj in CM_intv]  # 长度20

    """1. 首先，完成从初始模型到第一轮全局训练的模型叠加"""
    """
    由于在FL训练开始时，inIT_model不包含关于被遗忘用户的任何信息，您只需要覆盖其他保留用户的局部模型，您可以在第一轮全局训练后获得全局模型。
    """
    epoch = 0
    unlearn_global_models = list()
    unlearn_global_models.append(copy.deepcopy(selected_GMs[0]))

    new_global_model = fedavg(selected_CMs[epoch])  # 聚合剔除目标客户端的其余客户端
    unlearn_global_models.append(copy.deepcopy(new_global_model))  # 光剔除没矫正的全局模型
    print("Federated Unlearning Global Epoch  = {}".format(epoch))

    """2. Then, the first round of global model as a starting point, the model is gradually corrected"""
    """
    该步骤将第一轮全局训练得到的全局模型作为新的训练起点，利用保留用户的数据进行少量训练(少量意味着减少局部迭代次数，即减少每个用户的局部训练轮数。
    参数forget_local_epoch_ratio用于控制和减少局部训练轮数。)获取每个保留用户的局部模型参数的迭代方向，从new_global_model开始。注意，用户模型的这一部分是ref_client_models。
    然后，我们使用从未忘记的FL训练中保存的old_client_models和old_global_models，以及当我们忘记一个用户时得到的ref_client_models和new_global_Model，来构建下一轮的全局模型


    (ref_client_models - new_global_model) / ||ref_client_models - new_global_model||，表示模型参数迭代的方向，从一个删除用户的新全局模型开始。将方向标记为step_direction

    ||old_client_models - old_global_model||，表示从旧的全局模型开始，删除一个用户后，模型参数迭代的步长。一步step_length

    因此，新的参考模型的最终方向是step_direction*step_length + new_global_model。
    """
    """
    这部分直观说明:通常在IID数据中，数据分片后，模型参数迭代的方向大致相同。其基本思想是充分利用保存在标准FL训练中的客户端模型参数数据，然后通过修正这部分参数，将其应用于忘记用户的新全局模型的迭代。

    For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
    for unblearning FL：newGM_t-->newCM0, newCM1, newCM2, newCM3--> newGM_t+1
    oldGM_t和newGM_t本质上代表了不同的训练起点。然而，在IID数据下，oldCM和newCM应该在大致相同的方向收敛。
    因此，我们使用newCM -newgm_t作为起点，在用户数据上训练更少轮数，得到newCM，然后使用(newCM -newgm_t)/|| newCM -newgm_t ||作为当前的遗忘设置。
    模型参数迭代方向。以|| oldcm-oldgm_t ||作为迭代步骤，最后使用|| oldcm-oldgm_t ||*(newcm-newgm_t)/|| newcm-newgm_t |0 |1对新模型进行迭代。
    FedEraser 迭代公式: newGM_t+1 = newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

    """

    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(
        FL_params.local_epoch * FL_params.forget_local_epoch_ratio)  # 计算大于等于该值的最小整数，也就是计算遗忘学习的轮次，即E_cali
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    FL_params.global_epoch = CM_intv.shape[0]  # 大小是由FL_params.unlearn_interval(控制模型参数保存的轮数)决定，越大越小

    print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(FL_params.global_epoch):  #
        if (epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))
        global_model = unlearn_global_models[epoch]  # 是遗忘掉目标客户端之后经过FedAvg的模型

        new_client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)

        new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch + 1], global_model)

        unlearn_global_models.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch
    return unlearn_global_models

def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将全局模型移至GPU
    global_model_before_forget.to(device)
    global_model_after_forget.to(device)

    old_param_update = {}
    new_param_update = {}
    new_global_model_state = global_model_after_forget.state_dict()

    return_model_state = {}

    assert len(old_client_models) == len(new_client_models)  # 对布尔类型进行检验，如果不是true直接报错

    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = torch.zeros_like(global_model_before_forget.state_dict()[layer], device=device)
        new_param_update[layer] = torch.zeros_like(global_model_before_forget.state_dict()[layer], device=device)
        return_model_state[layer] = torch.zeros_like(global_model_before_forget.state_dict()[layer], device=device)

        for ii in range(len(new_client_models)):
            old_client_models[ii].to(device)
            new_client_models[ii].to(device)

            old_param_update[layer] += old_client_models[ii].state_dict()[layer].to(device)
            new_param_update[layer] += new_client_models[ii].state_dict()[layer].to(device)

        old_param_update[layer] /= len(new_client_models)
        new_param_update[layer] /= len(new_client_models)

        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]

        step_length = torch.norm(old_param_update[layer])
        step_direction = new_param_update[layer] / torch.norm(new_param_update[layer])

        return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

    return_global_model = copy.deepcopy(global_model_after_forget)
    return_global_model.load_state_dict(return_model_state)

    return return_global_model


def unlearning_without_cali(old_global_models, old_client_models, FL_params):
    """


    Parameters
    ----------
    old_client_models : list of DNN models
        All user local update models are saved during the federated learning and training process that is not forgotten.
    FL_params : parameters
        All parameters in federated learning and federated forgetting learning

    Returns
    -------
    global_models : List of DNN models
        In each update round, the client model of the user who needs to be forgotten is removed, and the parameters of other users' client models are directly superimposing to form the new Global Model of each round

    """
    """
    The basic process is as follows：For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
                 For unlearning FL：newGM_t-->The parameters of oldCM and oldGM were directly leveraged to update global model--> newGM_t+1
    The update process is as follows：newGM_t+1 = (oldCM - oldGM_t) + newGM_t
    """
    if (FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')

    if (not (FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(
            range(FL_params.N_client)))
    forget_client = FL_params.forget_client_idx

    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii * FL_params.N_client: ii * FL_params.N_client + FL_params.N_client]
        temp.pop(forget_client)
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]

    uncali_global_models = list()
    uncali_global_models.append(copy.deepcopy(old_global_models[0]))
    epoch = 0
    uncali_global_model = fedavg(old_client_models[epoch])
    uncali_global_models.append(copy.deepcopy(uncali_global_model))
    print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))

    """
    new_GM_t+1 = newGM_t + (oldCM_t - oldGM_t)

    For standard federated learning:oldGM_t --> oldCM_t --> oldGM_t+1
    For accumulatring:    newGM_t --> (oldCM_t - oldGM_t) --> oldGM_t+1
    对于未标定的联邦遗忘学习，利用标准联邦学习中未遗忘用户的参数更新直接覆盖新的全局模型，得到下一轮新的全局模型。
    """
    old_param_update = dict()  # (oldCM_t - oldGM_t)
    return_model_state = dict()  # newGM_t+1

    for epoch in range(FL_params.global_epoch):
        if (epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))

        current_global_model = uncali_global_models[epoch]  # newGM_t
        current_client_models = old_client_models[epoch]  # oldCM_t
        old_global_model = old_global_models[epoch]  # oldGM_t
        # global_model_before_forget = old_global_models[epoch]#old_GM_t

        for layer in current_global_model.state_dict().keys():
            # State variable initialization
            old_param_update[layer] = 0 * current_global_model.state_dict()[layer]
            return_model_state[layer] = 0 * current_global_model.state_dict()[layer]

            for ii in range(len(current_client_models)):
                old_param_update[layer] += current_client_models[ii].state_dict()[layer]
            old_param_update[layer] /= (ii + 1)  # oldCM_t

            old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[
                layer]  # 参数： oldCM_t - oldGM_t

            return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[
                layer]  # newGM_t + (oldCM_t - oldGM_t)

        return_global_model = copy.deepcopy(old_global_models[0])
        return_global_model.load_state_dict(return_model_state)

        uncali_global_models.append(return_global_model)

    return uncali_global_models


def unlearning_my(need_forget_model, client_loaders, test_loader, FL_params):
    std_time = time.time()
    # model = copy.deepcopy(old_global_models[-1])
    model = copy.deepcopy(need_forget_model)
    model_forget = copy.deepcopy(model)
    # forget_client = FL_params.forget_client_idx
    forget_client = FL_params.forget_client_idx
    loaders_rect = []
    # 将idx和loader对应起来
    for ii in range(FL_params.N_client):
        if ii == forget_client:
            forget_loader = client_loaders[ii]
        else:
            loaders_rect.append(client_loaders[ii])

    model.eval()

    # 计算被遗忘的客户端的FIM
    Fisher_forget_client = FIM_computing(model, forget_loader)  # 所有客户端是并行计算FIM的，所以计时只记一次

    end_time = time.time()
    time1 = (std_time - end_time)

    # 计算剩余客户端的FIM，并求平均值
    # fisher_rect_list = []
    # for loader in loaders_rect:
    #     fisher_rect_list.append(FIM_computing(model, loader))

    f_rect_list = []
    for loader in loaders_rect:
        model_tmp = copy.deepcopy(need_forget_model)
        # 将字典中的参数加载到模型
        model_tmp.load_state_dict(FIM_computing(model, loader))
        f_rect_list.append(model_tmp)
    file_name = 'memory_list/FIM_list[' + str(FL_params.global_epoch) + ' ' + str(FL_params.N_client) + ' ' + str(
        FL_params.data_name) + '].csv'
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(f_rect_list)

    fisher_rect_list = []
    for loader in loaders_rect:
        fisher_rect_list.append(FIM_computing(model, loader))

    std_time = time.time()

    FIM_rect_avg = {}
    for client_rect in fisher_rect_list:
        for key, value in client_rect.items():
            if key in FIM_rect_avg:
                FIM_rect_avg[key] += value
            else:
                FIM_rect_avg[key] = value
    aa = FL_params.N_client - 1
    for key in FIM_rect_avg:
        FIM_rect_avg[key] /= (FL_params.N_client - 1)

    '''进行遗忘'''
    # alpha = 1
    # lamb = 1
    weight_model = model.state_dict()
    weight_model_dampen = copy.deepcopy(weight_model)
    FIM_rect_avg_comparison = {}

    for key in FIM_rect_avg:
        FIM_rect_avg_comparison[key] = FL_params.alpha * FIM_rect_avg[key]

    for key in weight_model:
        # 保持所有张量在GPU上
        FIM_ratio = (FIM_rect_avg[key] / Fisher_forget_client[key])
        beta = torch.min(FL_params.lamb * FIM_ratio, torch.tensor(1.0).to(FIM_ratio.device))  # 确保常量1也在相同的设备上
        weight_model_dampen[key] = torch.where(Fisher_forget_client[key] > FIM_rect_avg_comparison[key],
                                               weight_model[key] * beta, weight_model[key])

    model_forget.load_state_dict(weight_model_dampen)
    end_time = time.time()
    time_total_noRe = (std_time - end_time) + time1

    '''进行适当的精确度提升，再训练几次'''
    # g_reduce_rate = 0.05
    # l_reduce_rate = 0.1
    global_epoch_init = copy.deepcopy(FL_params.global_epoch)
    local_epoch_init = copy.deepcopy(FL_params.local_epoch)

    ############# retrain1 ##############
    std_time = time.time()
    FL_params.g_reduce_rate = 0.05
    FL_params.l_reduce_rate = 0.2  # 2轮本地训练
    FL_params.global_epoch = math.ceil(FL_params.global_epoch * FL_params.g_reduce_rate)
    FL_params.local_epoch = math.ceil(FL_params.local_epoch * FL_params.l_reduce_rate)

    FL_params.if_retrain = True
    re1_model_forget = copy.deepcopy(model_forget)
    model_forget_tmp = FL_Retrain(re1_model_forget, client_loaders, test_loader, FL_params)
    model_forget_withRe = model_forget_tmp[-1]

    FL_params.global_epoch = copy.deepcopy(global_epoch_init)
    FL_params.local_epoch = copy.deepcopy(local_epoch_init)

    end_time = time.time()
    time_Re = (std_time - end_time)
    time_total_Re = time_total_noRe + time_Re

    ############# retrain2 ##############
    std_time = time.time()
    FL_params.g_reduce_rate = 0.1
    FL_params.l_reduce_rate = 0.5  # 2轮本地训练
    FL_params.global_epoch = math.ceil(FL_params.global_epoch * FL_params.g_reduce_rate)
    FL_params.local_epoch = math.ceil(FL_params.local_epoch * FL_params.l_reduce_rate)

    FL_params.if_retrain = True
    re2_model_forget = copy.deepcopy(model_forget)
    model_forget_tmp2 = FL_Retrain(re2_model_forget, client_loaders, test_loader, FL_params)
    model_forget_withRe2 = model_forget_tmp2[-1]

    end_time = time.time()
    time_Re2 = (std_time - end_time)
    time_total_Re2 = time_total_noRe + time_Re2

    ############## 参数还原 #############

    FL_params.global_epoch = copy.deepcopy(global_epoch_init)
    FL_params.local_epoch = copy.deepcopy(local_epoch_init)

    return model_forget, model_forget_withRe, model_forget_withRe2, time_total_noRe, time_total_Re, time_total_Re2



def FIM_computing(model, loader):

    # model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # 初始化 Fisher 信息矩阵的变量
    fisher_information = {}
    for name, param in model.named_parameters():
        fisher_information[name] = torch.zeros_like(param, device=device)

    # 对 client_loader 中的所有数据进行处理
    for data in loader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # 更新 Fisher 信息矩阵的估计值
        for name, param in model.named_parameters():
            fisher_information[name] += param.grad.data.pow(2)

    # 计算 Fisher 信息矩阵的平均值
    for name, value in fisher_information.items():
        fisher_information[name] = value / len(loader.dataset)

    return fisher_information


def Grad_computing(model, loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化 grad 信息矩阵的变量
    grad_information = {}
    for name, param in model.named_parameters():
        grad_information[name] = torch.zeros_like(param, device=device)

    # 对 client_loader 中的所有数据进行处理
    for data in loader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # 更新 grad 信息矩阵的估计值
        for name, param in model.named_parameters():
            grad_information[name] += param.grad.data

    # 计算 Fisher 信息矩阵的平均值
    for name, value in grad_information.items():
        grad_information[name] = value / len(loader.dataset)

    return grad_information





