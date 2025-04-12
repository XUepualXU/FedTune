# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
# %%
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time

# ourself libs
from model_initiation import model_init
from data_preprocess import data_init, data_init_with_shadow
from FL_base import global_train_once
from FL_base import fedavg
from FL_base import test
import matplotlib.pyplot as plt
import math
import random

from FL_base import FL_Train, FL_Retrain
from Fed_Unlearn_base_alpha_lamb import federated_learning_unlearning
from membership_inference import train_attack_model, attack

"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 50
        self.N_client = 10
        self.data_name = 'mnist'  # purchase, cifar10, mnist, adult
        self.global_epoch = 10  # 20 50
        # self.communication_epoch = 5
        self.local_epoch = 10  # 10 20

        # Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.0001

        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()  # 判断当前电脑有没有CUDA
        self.use_gpu = True
        self.train_with_test = False

        # Federated Unlearning Settings
        self.unlearning_type = 'FedEraser'  # FedEraser, MyUL
        self.unlearn_interval = 5  # 用于控制模型参数保存的轮数。1表示本文中每轮N_itv保存一次的参数
        self.forget_client_idx = 2  # 如果要忘记，将None更改为客户端索引

        self.if_retrain = False  # 如果设置为True，则使用FL-Retrain函数重新训练全局模型，并且丢弃forget_client_IDx号码对应的用户数据.
        # 如果设置为False，则只输出最终训练完成后的全局模型

        self.if_unlearning = False  # 如果设置为False, global_train_once函数不会跳过需要遗忘的用户;
        # 如果设置为True, global_train_once函数在训练时跳过被遗忘的用户

        self.forget_local_epoch_ratio = 0.5  # 当选择一个用户被遗忘后，其他用户需要在各自的数据集中进行多轮在线训练，得到模型收敛的大方向，从而提供模型收敛的大方向。
        # forget_local_epoch_ratio*local_epoch 是我们需要得到每个局部模型的收敛方向时进行局部训练的轮数

        # self.mia_oldGM = False
        # 关于我的参数的设置
        self.g_reduce_rate = 0.05  # 再训练提升准确度的过程需要多少比率的全局迭代次数和本地训练次数
        self.l_reduce_rate = 0.1

        self.alpha = 0.3  # 用于Fisher的对比
        self.lamb = 2.5  # 用于参数抑制
        self.param_string = '1 0.9 0.8 0.3'


def Federated_Unlearning(data_name, alpha, lamb, selected_clients):
    """Step 1.设置联邦遗忘学习的参数"""
    FL_params = Arguments()

    FL_params.lamb = lamb
    FL_params.alpha = alpha
    FL_params.data_name = data_name
    print(20 * '=' + " We use dataset: " + FL_params.data_name + ' 所有参与的客户端为：' + str(selected_clients) + '  其中被遗忘的客户端为：' + str(
        FL_params.forget_client_idx) + "  抑制程度：alpha = " + str(FL_params.alpha) + " lamb = " + str(
        FL_params.lamb))

    # FL_params.param_string = '0.9 0.9 0.8 0.2'
    torch.manual_seed(FL_params.seed)  # 设置随机种子
    # kwargs for data loader 数据加载器的Kwargs
    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))

    """Step 2. 构建联邦学习所需的必要用户私有数据集，以及公共测试集"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!\n")
    # 根据不同的数据集设置不同的模型，并且加载数据
    init_global_model = model_init(FL_params.data_name)
    client_all_loaders, test_loader = data_init(FL_params)

    # selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
    # FL_params.forget_client_idx = random.choice(selected_clients)
    client_loaders = list()
    rest_client_loaders = list()
    # for idx in selected_clients:
    #     client_loaders.append(client_all_loaders[idx])
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])
        if idx != selected_clients[FL_params.forget_client_idx]:
            rest_client_loaders.append(client_all_loaders[idx])

    # client_all_loaders = client_loaders[selected_clients]
    # client_loaders, test_loader, shadow_client_loaders, shadow_test_loader = data_init_with_shadow(FL_params)
    """
    This section of the code gets the initialization model init Global Model 这部分代码获取初始化模型init全局模型
    User data loader for FL training Client_loaders and test data loader Test_loader FL训练的用户数据加载器Client_loaders和测试数据加载器Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadowl_test_loader 用于隐蔽FL训练的用户数据加载器，Shadow_client_loaders和测试数据加载器Shadowl_test_loader
    """

    """
    Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration
    这一部分里的federated_learning_unlearning()函数中存在不同的遗忘方法 ！！！！
    """
    print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...\n")

    # old_GMs, unlearn_GMs, uncali_unlearn_GMs, retrain_GMs, my_unlearn, my_unlearn_withRe, my_unlearn_withRe2, time_list \
    #     = federated_learning_unlearning(init_global_model, selected_clients, client_loaders, test_loader, FL_params)
    old_GMs, unlearn_GMs, retrain_GMs, my_unlearn, my_unlearn_withRe, my_unlearn_withRe2, time_list \
        = federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params)

    # if(FL_params.if_retrain == True):
    #
    #     t1 = time.time()
    #     retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
    #     t2 = time.time()
    #     print("Time using = {} seconds".format(t2-t1))

    """Step 4  基于目标全局模型在client_loaders和test_loaders上的输出，构建了成员推理攻击模型。在这种情况下，我们只在训练结束时对模型进行MIA攻击"""

    """ MIA:基于oldGM模型的输出，建立MIA攻击模型，然后使用该攻击模型攻击unlearn GM，如果攻击准确率明显下降，说明我们的unlearn方法确实能够有效的去除用户信息"""
    print(60 * '=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = copy.deepcopy(old_GMs[T_epoch])
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    target_loader = client_loaders[FL_params.forget_client_idx]

    rect_dataset = [dl.dataset for dl in rest_client_loaders]
    concatented_dataset = ConcatDataset(rect_dataset)
    combined_rect_loader = DataLoader(concatented_dataset, batch_size=FL_params.local_batch_size, shuffle=True)

    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard  ")
    target_model = copy.deepcopy(old_GMs[T_epoch])
    (pre_old, rec_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    acc_old = Computing_Accuracy(test_loader, target_model, FL_params)
    acc_old_target = Computing_Accuracy(target_loader, target_model, FL_params)
    acc_old_rect = Computing_Accuracy(combined_rect_loader, target_model, FL_params)
    loss_old = evaluate_model(test_loader, target_model, FL_params)
    loss_old_target = evaluate_model(target_loader, target_model, FL_params)
    loss_old_rest = evaluate_model(combined_rect_loader, target_model, FL_params)

    FL_params.if_retrain = True
    if (FL_params.if_retrain == True):
        print("Attacking against FL Retrain  ")
        target_model = copy.deepcopy(retrain_GMs[T_epoch])
        (pre_retrain, rec_retrain) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
        acc_retrain = Computing_Accuracy(test_loader, target_model, FL_params)
        acc_retrain_target = Computing_Accuracy(target_loader, target_model, FL_params)
        acc_retrain_rect = Computing_Accuracy(combined_rect_loader, target_model, FL_params)
        loss_retrain = evaluate_model(test_loader, target_model, FL_params)
        loss_retrain_target = evaluate_model(target_loader, target_model, FL_params)
        loss_retrain_rest = evaluate_model(combined_rect_loader, target_model, FL_params)

    print("Attacking against FL Unlearn  ")
    target_model = copy.deepcopy(unlearn_GMs[T_epoch])
    (pre_unlearn, rec_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)
    acc_unlearn = Computing_Accuracy(test_loader, target_model, FL_params)
    acc_unlearn_target = Computing_Accuracy(target_loader, target_model, FL_params)
    acc_unlearn_rect = Computing_Accuracy(combined_rect_loader, target_model, FL_params)
    loss_unlearn = evaluate_model(test_loader, target_model, FL_params)
    loss_unlearn_target = evaluate_model(target_loader, target_model, FL_params)
    loss_unlearn_rest = evaluate_model(combined_rect_loader, target_model, FL_params)

    print("Attacking against MY_unlearn  ")  # FedAccum
    target_model = copy.deepcopy(my_unlearn)
    (pre_my_unlearn, rec_my_unlearn) = attack(target_model, attack_model, client_loaders, test_loader,
                                              FL_params)
    acc_my_unlearn = Computing_Accuracy(test_loader, target_model, FL_params)  # 测试集
    acc_my_unlearn_target = Computing_Accuracy(target_loader, target_model, FL_params)  # 遗忘集
    acc_my_unlearn_rect = Computing_Accuracy(combined_rect_loader, target_model, FL_params)
    loss_my_unlearn = evaluate_model(test_loader, target_model, FL_params)
    loss_my_unlearn_target = evaluate_model(target_loader, target_model, FL_params)
    loss_my_unlearn_rest = evaluate_model(combined_rect_loader, target_model, FL_params)

    print("Attacking against MY_unlearn with Retrain ")  # FedAccum
    target_model = copy.deepcopy(my_unlearn_withRe)
    (pre_my_unlearn_Re, rec_my_unlearn_Re) = attack(target_model, attack_model, client_loaders, test_loader,
                                                    FL_params)
    acc_my_unlearn_Re = Computing_Accuracy(test_loader, target_model, FL_params)
    acc_my_unlearn_Re_target = Computing_Accuracy(target_loader, target_model, FL_params)
    acc_my_unlearn_Re_rect = Computing_Accuracy(combined_rect_loader, target_model, FL_params)
    loss_my_unlearn_Re = evaluate_model(test_loader, target_model, FL_params)
    loss_my_unlearn_Re_target = evaluate_model(target_loader, target_model, FL_params)
    loss_my_unlearn_Re_rest = evaluate_model(combined_rect_loader, target_model, FL_params)

    print("Attacking against MY_unlearn with Retrain2 ")  # FedAccum
    target_model = copy.deepcopy(my_unlearn_withRe2)
    (pre_my_unlearn_Re2, rec_my_unlearn_Re2) = attack(target_model, attack_model, client_loaders, test_loader,
                                                      FL_params)
    acc_my_unlearn_Re2 = Computing_Accuracy(test_loader, target_model, FL_params)
    acc_my_unlearn_Re2_target = Computing_Accuracy(target_loader, target_model, FL_params)
    acc_my_unlearn_Re2_rect = Computing_Accuracy(combined_rect_loader, target_model, FL_params)
    loss_my_unlearn_Re2 = evaluate_model(test_loader, target_model, FL_params)
    loss_my_unlearn_Re2_target = evaluate_model(target_loader, target_model, FL_params)
    loss_my_unlearn_Re2_rest = evaluate_model(combined_rect_loader, target_model, FL_params)

    pre_data = [pre_my_unlearn, pre_my_unlearn_Re, pre_my_unlearn_Re2, pre_unlearn, pre_retrain, pre_old]
    rec_data = [rec_my_unlearn, rec_my_unlearn_Re, rec_my_unlearn_Re2, rec_unlearn, rec_retrain, rec_old]
    test_acc = [acc_my_unlearn, acc_my_unlearn_Re, acc_my_unlearn_Re2, acc_unlearn, acc_retrain, acc_old]
    forget_acc = [acc_my_unlearn_target, acc_my_unlearn_Re_target, acc_my_unlearn_Re2_target, acc_unlearn_target,
                       acc_retrain_target, acc_old_target]
    rest_acc = [acc_my_unlearn_rect, acc_my_unlearn_Re_rect, acc_my_unlearn_Re2_rect, acc_unlearn_rect,
                     acc_retrain_rect, acc_old_rect]
    test_loss = [loss_my_unlearn, loss_my_unlearn_Re, loss_my_unlearn_Re2, loss_unlearn, loss_retrain, loss_old]
    forget_loss = [loss_my_unlearn_target, loss_my_unlearn_Re_target, loss_my_unlearn_Re2_target, loss_unlearn_target,
                   loss_retrain_target, loss_old_target]
    rest_loss = [loss_my_unlearn_rest, loss_my_unlearn_Re_rest, loss_my_unlearn_Re2_rest, loss_unlearn_rest,
                 loss_retrain_rest, loss_old_rest]

    return test_acc, forget_acc, rest_acc, test_loss, forget_loss, rest_loss, pre_data, rec_data

def Computational_prediction_probability(client_loader, model, FL_params):
    device = torch.device("cuda" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    model.to(device)
    model.eval()
    # probabilities_sum = torch.zeros((1, num_classes))  # 用于存储预测概率总和的张量
    probabilities_sum = 0
    num_samples = 0  # 用于跟踪处理的样本数量

    with torch.no_grad():
        for data, target in client_loader:  # 不需要目标标签
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(data)
            probabilities = torch.softmax(output, dim=1)  # 使用softmax获取类别概率
            probabilities_from_target = []
            for ii in range(len(probabilities)):
                targ = int(target[ii])
                probabilities_from_target.append(float(probabilities[ii][targ]))
            probabilities_sum += sum(probabilities_from_target)  # 对批次内的概率进行求和
            num_samples += data.size(0)  # 更新样本数量

    # 计算平均概率
    avg_probabilities = probabilities_sum / num_samples

    return avg_probabilities

def Computing_Accuracy(data_loader, model, FL_params):
    device = torch.device("cuda" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    model.to(device)
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def evaluate_model(data_loader, model, FL_params):
    # device = torch.device("cuda")
    device = torch.device("cuda" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    model.to(device)
    model.eval()  # 将模型设置为评估模式

    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():  # 在评估期间不计算梯度
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)  # 将数据和标签移至设定的设备
            output = model(data)  # 生成预测
            loss = criterion(output, target)  # 计算损失
            total_loss += loss.item() * data.size(0)  # 累积损失
            total_samples += data.size(0)  # 计数总样本数量

    average_loss = total_loss / total_samples  # 计算平均损失
    print(f'loss: {average_loss}%')
    return average_loss

if __name__ == '__main__':
    start_time = time.time()
    print(60 * '* ' + 'start time ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)) + '\n')

    times = 3  # 重复实验取平局值的次数

    ''' ********************** 简单场景 ***************************** '''
    # 横坐标：不同的污染程度
    print(120 * '=')
    test_list = []
    forget_list = []
    rest_list = []
    test_list_loss = []
    forget_list_loss = []
    rest_list_loss = []
    pre_list = []
    rec_list = []

    N_total_client = 50
    # client_list = [[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19],[20,21,22,23,24,25,26,27,28,29],
    #                [30,31,32,33,34,35,36,37,38,39],[40,41,42,43,44,45,46,47,48,49]]
    # client_list = [20, 34, 11, 7, 26, 41, 46, 32, 1, 16]
    client_list = range(0, 11)
    data_name = 'purchase'  # purchase, cifar10, mnist, adult

    alpha_list = [0.3]
    lamb_list = np.arange(0.1, 2.5, 0.5) # lamb
    # alpha_list = np.arange(0, 5, 0.5) # alpha
    # lamb_list = [7.0]
    # for alpha in alpha_list:
    for alpha in alpha_list:
        for lamb in lamb_list:
            avg_test_acc = []
            avg_forget_acc = []
            avg_rest_acc = []
            avg_test_loss = []
            avg_forget_loss = []
            avg_rest_loss = []
            avg_pre = []
            avg_rec = []
            for time_i in range(0, times):
                test_acc, forget_acc, rest_acc, test_loss, forget_loss, rest_loss, \
                    pre, rec = Federated_Unlearning(data_name, alpha, lamb, client_list)
                avg_test_acc.append(test_acc)
                avg_forget_acc.append(forget_acc)
                avg_rest_acc.append(rest_acc)
                avg_test_loss.append(test_loss)
                avg_forget_loss.append(forget_loss)
                avg_rest_loss.append(rest_loss)
                avg_pre.append(pre)
                avg_rec.append(rec)

            test_list.append(np.mean(avg_test_acc, axis=0))
            forget_list.append(np.mean(avg_forget_acc, axis=0))
            rest_list.append(np.mean(avg_rest_acc, axis=0))
            test_list_loss.append(np.mean(avg_test_loss, axis=0))
            forget_list_loss.append(np.mean(avg_forget_loss, axis=0))
            rest_list_loss.append(np.mean(avg_rest_loss, axis=0))
            pre_list.append(np.mean(avg_pre, axis=0))
            rec_list.append(np.mean(avg_rec, axis=0))

    T_test_list = list(zip(*test_list))
    T_forget_list = list(zip(*forget_list))
    T_rest_list = list(zip(*rest_list))
    T_test_list_loss = list(zip(*test_list_loss))
    T_forget_list_loss = list(zip(*forget_list_loss))
    T_rest_list_loss = list(zip(*rest_list_loss))
    T_pre_list = list(zip(*pre_list))
    T_rec_list = list(zip(*rec_list))

    print('T_test_list = ' + str(T_test_list))
    print('T_forget_list = ' + str(T_forget_list))
    print('T_rest_list = ' + str(T_rest_list))
    print('T_test_list_loss = ' + str(T_test_list_loss))
    print('T_forget_list_loss = ' + str(T_forget_list_loss))
    print('T_rest_list_loss = ' + str(T_rest_list_loss))
    print('T_pre_list = ' + str(T_pre_list))
    print('T_rec_list = ' + str(T_rec_list))

    line_label = ['FedDampen', 'FedOptDam - 1l', 'FedOptDam - 2l',
               'FedEraser', 'FedRetain', 'FedAvg']
    title_label = ['test data accuracy', 'forget data accuracy', 'rest data accuracy',
                   'test data loss', 'forget data loss', 'rest data loss', 'precision', 'recall']
    y_label = ['accuracy', 'accuracy', 'accuracy',
                   'loss', 'loss', 'loss', 'precision', 'recall']

    if len(alpha_list) == 1:
        x_values = lamb_list
    if len(lamb_list) == 1:
        x_values = alpha_list

    list_list = [T_test_list, T_forget_list, T_rest_list, T_test_list_loss, T_forget_list_loss, T_rest_list_loss,
                 T_pre_list, T_rec_list]

    for i, list_i in enumerate(list_list):
        for index, column in enumerate(list_i):
            plt.plot(x_values, column, marker='.', label=line_label[index])
        plt.legend()

        plt.title(title_label[i] + " (" + data_name + ")")
        if len(alpha_list) == 1:
            plt.xlabel("lamb")
        if len(lamb_list) == 1:
            plt.xlabel('alpha')
        plt.ylabel(y_label[i])
        plt.xticks(x_values)

        plt.show()


    end_time = time.time()
    print(50 * '* ' + 'start time ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
          + '>>>  end time ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
