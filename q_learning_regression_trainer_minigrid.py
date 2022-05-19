import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions import Categorical
#from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, TensorDataset
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3 import DQN, A2C, PPO
from gym_minigrid.wrappers import *
import time
from os import listdir
from os.path import isfile, join
import copy
import random
import csv
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser(description='Q Learning Trainer')
parser.add_argument('--run_test', '-rt', action='store_true', default=False)
parser.add_argument('--agent_play', '-ap', action='store_true', default=False) #use the agent to play the game
parser.add_argument('--check_all_door_pos', '-cad', action='store_true', default=False)
parser.add_argument('--check_log', '-cl', action='store_true', default=False)
args = parser.parse_args()

import collections


num_actual_actions = 3
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pred_error_details(log_matrix):
    effective_action_num = 3
    pong_action_map = {0: "STAY", 1: "UP  ", 2: "DOWN"}
    total_wrong = np.sum(log_matrix)
    for i in range(effective_action_num):
        for j in range(effective_action_num):
            if not i==j:
                print("[ predicted action:", pong_action_map[i], "and correct action:", pong_action_map[j], "] takes up around", int(log_matrix[i,j]*100/total_wrong), "percent of all wrong predictions.")

def plot_log_matrix(log_matrix, title="Agent Success Per Positions", option=0):
    log_matrix[:,:,0] = np.rot90(log_matrix[:,:,0],k=3)
    log_matrix[:,:,0] = np.flip(log_matrix[:,:,0],axis=1)

    
    if option==0:
        log_matrix[:,:,1] = np.rot90(log_matrix[:,:,1],k=3)
        log_matrix[:,:,1] = np.flip(log_matrix[:,:,1],axis=1)        
        #plot the correct or fail count matrix
        wall_pos = log_matrix[:,:,0]+log_matrix[:,:,1]
        to_plot_matrix = log_matrix[:,:,0].copy() + 1
        #print(to_plot_matrix)
        to_plot_matrix[wall_pos==0] = 0
        #print(to_plot_matrix)
    elif option==1:
        #plot std matrix
        to_plot_matrix = log_matrix[:,:,0].copy()
    plt.imshow(to_plot_matrix)
    plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.show()

def plot_vector_field(q_dict, map_size_x=19, map_size_y=19, use_normed_vector=True):
    #first initalize a mesh grid
    x,y = np.meshgrid(np.linspace(0,map_size_x,map_size_x),np.linspace(0,map_size_y,map_size_y))

    #calculate the 2D Vector with u and v in each direction

    u = q_dict[:,:,0,2]-q_dict[:,:,2,2]  # length is calculated as the difference of Q of going forward in two opposite directions
    v = q_dict[:,:,3,2]-q_dict[:,:,1,2]  # 
    u = np.rot90(u,k=1)
    #u = np.flip(u,axis=1)
    v = np.rot90(v,k=1)
    #v = np.flip(v,axis=1)

    print("u shape", np.shape(u))
    if use_normed_vector:
        vector_len = np.sqrt(np.power(u,2)+np.power(v,2))
        vector_len[vector_len==0] = 1
        normalizer = 1/vector_len
        normed_u = np.multiply(u, normalizer)
        normed_v = np.multiply(v, normalizer)
        plt.quiver(x,y,normed_u,normed_v)
    else:
        plt.quiver(x,y,u,v)
    plt.show()

def label_processing_Pong(targets):
    #action space for Pong is actuall 3. Combine same actions:

    targets = targets.numpy()

    for i in range(len(targets)):
        if targets[i] == 1: 
            targets[i] = 0
        elif targets[i] == 2: 
            targets[i] = 1
        elif targets[i] == 4: 
            targets[i] = 1
        elif targets[i] == 3: 
            targets[i] = 2
        elif targets[i] == 5: 
            targets[i] = 2

    return torch.from_numpy(targets)

def get_npy_data_unprocessed(data_path):
    #not converting to index based labels
    inputs = np.array([])
    targets = np.array([])
    
    for i in range(len(data_path)):
        data_temp = np.load(data_path[i], allow_pickle=True)
        
        inputs_temp = np.array([d["state"] for d in data_temp])
        targets_temp = np.array([d["Q_per_action"] for d in data_temp])
        #print(np.shape(inputs_temp), np.shape(targets_temp))
        #print(data_path[i], inputs_temp)
        if i==0:
            inputs = inputs_temp
            targets = targets_temp
        else:
            inputs = np.concatenate((inputs, inputs_temp), axis=0)
            targets = np.concatenate((targets, targets_temp), axis=0)
    print(np.shape(inputs), np.shape(targets))
    #inputs = np.squeeze(inputs, axis=1)
    #targets = np.squeeze(targets, axis=2)
    #print(np.shape(inputs), np.shape(targets))
    return inputs, targets

def get_action_from_torch_categorical(raw_q_values, use_normed_value=True):
    if use_normed_value:
        raw_q_values = raw_q_values.cpu().numpy()
        raw_q_std = np.std(raw_q_values)
        #print(raw_q_std)
        raw_q_values = raw_q_values-np.mean(raw_q_values)
        raw_q_values = raw_q_values/raw_q_std  
        #raw_q_values = (raw_q_values-np.min(raw_q_values))/(np.max(raw_q_values)-np.min(raw_q_values))

        raw_q_values = torch.from_numpy(raw_q_values[0])
    #print("raw q values:", raw_q_values)
    dist = Categorical(logits=raw_q_values)
    action = dist.sample().cpu().item()
    #print("Sampled action:", action)
    return action

def get_action_from_distribution(raw_q_values, use_advatange_value=False):
    #print("Sample action from distribution")
    #print("original raw Q:", raw_q_values)
    if use_advatange_value:
        raw_q_values = raw_q_values.cpu().numpy()
        raw_q_std = np.std(raw_q_values)
        #print(raw_q_std)
        raw_q_values = raw_q_values-np.mean(raw_q_values)
        raw_q_values = raw_q_values/raw_q_std
        #print("advantage values:", raw_q_values)
        m = torch.nn.Softmax(dim=1)
        sm = m(torch.from_numpy(raw_q_values)).numpy()[0]
    else:
        m = torch.nn.Softmax(dim=1)
        sm = m(raw_q_values.cpu()).numpy()[0]

    #print("After softmax: ", sm)
    arg_sort = np.argsort(sm)
    #print("arg_sort", arg_sort)
    sorted_sm = sm.copy()
    sorted_sm = np.sort(sorted_sm)
    #print("Sorted: ", sorted_sm)
    cdf = []
    cdf.append(sorted_sm[0])
    for i in range(len(sorted_sm)-1):
        cdf.append(cdf[-1]+sorted_sm[i+1])

    #print("cdf is: ", cdf)
    cdf = np.asarray(cdf)
    one_rand = random.uniform(0,1)
    #print("rand value is: ", one_rand)
    rand_place = np.searchsorted(cdf, one_rand, side='left')
    selected_action = np.argwhere(arg_sort==rand_place).flatten()
    #print("Selected action:", selected_action)
    return selected_action
    

def target_checker(target):
    target = target.numpy()
    print("Histogram of the targets: ", np.histogram(target, bins=[0,1,2,3,4,5,6]))

def get_file_list(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if (isfile(join(data_dir, f)) and f.endswith('.npy'))]
        
    for i in range(len(onlyfiles)):
        onlyfiles[i] = data_dir+onlyfiles[i]
    #print(onlyfiles)

    return onlyfiles

def load_all_datapoints(data_dir, normalize_q=False):
    #get the top percentage of the data according to STD
    data_list, targets = get_npy_data_unprocessed(file_path)
    total_data_num = np.shape(data_list)[0]
    #std_argsort.reverse()
    if normalize_q:
        #target_std = np.std(targets, axis=1)
        #target_mean = np.mean(targets, axis=1)
        #print("target_mean shape: ", np.shape(target_mean))
        #targets = (targets-target_mean[:,None])/(target_std[:,None]+1e-10)
        #target_min = np.min(targets, axis=1)
        #target_max = np.max(targets, axis=1)
        #print("target_min shape: ", np.shape(target_min))
        #targets = (targets-target_min[:,None])/(target_max[:,None]-target_min[:,None]+1e-11)
        target_std = np.std(targets)
        target_mean = np.mean(targets)
        print("target_mean : ", target_mean, "target_std: ", target_std)
        targets = (targets-target_mean)/(target_std+1e-11)
        print(targets)

    inputs = data_list
    targets = targets
    print(np.shape(inputs))
    inputs = np.swapaxes(inputs, 1, 3)
    inputs = np.swapaxes(inputs, 2, 3)
    inputs = inputs.astype(np.float32)
    inputs = inputs/255.0
    #targets = (targets-np.min(targets))/(np.max(targets)-np.min(targets))
    #targets = (targets-np.mean(targets))/np.std(targets)
    #print(targets)
    #targets = targets + 2
    #print(np.max(targets))
    #print(np.min(targets))
    #targets = targets/21.0
    targets = targets.astype(np.float32)
    targets = torch.from_numpy(targets)
    #targets = targets.argmax(dim=1, keepdim=True)
    #targets = torch.squeeze(targets)

    print(np.shape(inputs), np.shape(targets))
    #print(targets)
    #data_checker(inputs)
    #target_checker(targets)

    return torch.from_numpy(inputs), targets

def select_datapoints(data_dir, percentage, normalize_q=False):
    #get the top percentage of the data according to STD
    data_list, targets = get_npy_data_unprocessed(file_path)
    total_data_num = np.shape(data_list)[0]
    data_std = np.std(targets, axis=1)
    std_argsort = np.argsort(data_std)
    
    to_index = int(percentage/100*total_data_num)
    #std_argsort.reverse()
    return_index = std_argsort[(-1-to_index):-1]
    inputs = data_list[return_index]
    targets = targets[return_index]
    print(np.shape(inputs))
    print("average std of selected datapoints: ", np.mean(np.std(targets, axis=1)))
    #inputs = np.squeeze(inputs, axis=1)
    #targets = np.squeeze(targets, axis=2)

    inputs = np.swapaxes(inputs, 1, 3)
    inputs = np.swapaxes(inputs, 2, 3)
    inputs = inputs.astype(np.float32)
    inputs = inputs/255.0
    #targets = (targets-np.min(targets))/(np.max(targets)-np.min(targets))
    #targets = (targets-np.mean(targets))/np.std(targets)
    #print(targets)
    #targets = targets + 2
    #print(np.max(targets))
    #print(np.min(targets))
    #targets = targets/21.0
    if normalize_q:
        target_std = np.std(targets, axis=1)
        target_mean = np.mean(targets, axis=1)
        print("target_mean shape: ", np.shape(target_mean))
        targets = (targets-target_mean[:,None])/(target_std[:,None]+1e-10)

    targets = torch.from_numpy(targets)
    #targets = targets.argmax(dim=1, keepdim=True)
    targets = torch.squeeze(targets)

    print(np.shape(inputs), np.shape(targets))
    #print(targets)
    #data_checker(inputs)
    #target_checker(targets)

    return torch.from_numpy(inputs), targets

def load_with_uncrucual_cases_processed(data_dir, percentage, balance_label=False):
    #for the percentage below Q value std ranking, mark as the same label
    inputs, targets = get_npy_data_unprocessed(file_path)
    total_data_num = np.shape(inputs)[0]
    data_std = np.std(targets, axis=1)
    std_argsort = np.argsort(data_std)
    
    process_index = int(percentage/100*total_data_num)
    keep_index = int(percentage/100*total_data_num)+int(percentage/100*total_data_num/3)
    #std_argsort.reverse()
    keep_index = std_argsort[(-1-keep_index):]
    change_index = std_argsort[0:(-1-process_index)]


    inputs = np.swapaxes(inputs, 1, 3)
    inputs = np.swapaxes(inputs, 2, 3)
    inputs = inputs.astype(np.float32)
    inputs = inputs/255.0
    targets = torch.from_numpy(targets)
    targets = targets.argmax(dim=1, keepdim=True)
    targets = torch.squeeze(targets)
    targets = label_processing_Pong(targets)

    targets[change_index] = 3
    if balance_label:
        inputs = inputs[keep_index]
        targets = targets[keep_index]
    print(np.shape(inputs), np.shape(targets))
    print(targets)
    #data_checker(inputs)

    return torch.from_numpy(inputs), targets




def load_partial_weight(target_model, saved_file):
    data, params, pytorch_variables = load_from_zip_file(saved_file, device=device)
    policy = params['policy']

    all_keys = list(policy.keys())

    change_flag = 0
    for para in target_model.parameters():
        
        if(change_flag<6):
            print("Loading layer", all_keys[change_flag])
            print("Target network layer dim:", np.shape(para))
            print("Source network layer dim:", np.shape(policy[all_keys[change_flag]]))
            para.data.copy_(policy[all_keys[change_flag]].cpu())
            para.requires_grad = False

        change_flag += 1


def data_checker(data):
    '''
    for i in range(990):
        #print(data[i])
        if np.array_equal(data[i], data[i+1000]):
            print("==============Warning: at the same state for step", i, i+1000, "===============")
        if np.array_equal(data[i], data[i+2000]):
            print("==============Warning: at the same state for step", i, i+2000, "===============")
        if np.array_equal(data[i], data[i+3000]):
            print("==============Warning: at the same state for step", i, i+3000, "===============")
        if np.array_equal(data[i+1000], data[i+2000]):
            print("==============Warning: at the same state for step", i+1000, i+2000, "===============")
        if np.array_equal(data[i+1000], data[i+3000]):
            print("==============Warning: at the same state for step", i+1000, i+3000, "===============")
        if np.array_equal(data[i+2000], data[i+3000]):
            print("==============Warning: at the same state for step", i+2000, i+3000, "===============")
    '''
    same_cnt = 0
    for i in range(4000):
      for j in range(i,4000):
        if i!=j and np.array_equal(data[i], data[j]):
            print("==============Warning: at the same state for step", i, j, "===============")
            same_cnt += 1
    print("total same state: ", same_cnt)





def get_npy_data(data_path):
    inputs = np.array([])
    targets = np.array([])
    
    for i in range(len(data_path)):
        data_temp = np.load(data_path[i], allow_pickle=True)
        
        inputs_temp = np.array([d["state"] for d in data_temp])
        targets_temp = np.array([d["Q_per_action"] for d in data_temp])
        #print(np.shape(inputs_temp), np.shape(targets_temp))
        #print(data_path[i], inputs_temp)
        if i==0:
            inputs = inputs_temp
            targets = targets_temp
        else:
            inputs = np.concatenate((inputs, inputs_temp), axis=0)
            targets = np.concatenate((targets, targets_temp), axis=0)
    
    inputs = np.squeeze(inputs, axis=1)
    targets = np.squeeze(targets, axis=2)

    inputs = np.swapaxes(inputs, 1, 3)
    inputs = np.swapaxes(inputs, 2, 3)
    inputs = inputs.astype(np.float32)
    inputs = inputs/255.0
    #targets = (targets-np.min(targets))/(np.max(targets)-np.min(targets))
    #targets = (targets-np.mean(targets))/np.std(targets)
    #print(targets)
    #targets = targets + 2
    #print(np.max(targets))
    #print(np.min(targets))
    #targets = targets/21.0
    targets = torch.from_numpy(targets)
    targets = targets.argmax(dim=1, keepdim=True)
    targets = torch.squeeze(targets)
    print(np.shape(inputs), np.shape(targets))
    print(targets)
    #data_checker(inputs)

    return torch.from_numpy(inputs), targets

def check_model_all_init_pos(model_class, model_save_name, env_name, door_loc=None, plot_heat=True, verbose=False, log_to_list=False, sample_from_dist=True, result_list=None, device_id=None, generate_vector_field=False):
    #print(door_loc)
    device = torch.device(device_id)
    start_time = time.time()
    if door_loc is None:
        door_loc_list = [[(9,1),(9,17)], [(8,9),(10,9)]]
    else:
        door_loc_list = door_loc
    env = gym.make(env_name, door_loc_list=door_loc_list)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)

    model = model_class()
    model.load_state_dict(torch.load(model_save_name))
    model.eval()
    model.to(device)

    obs = env.reset()
    rand_noise_injection = 0
    per_eps_steps = 0
    reset_cnt = 0
    print_fail = 1
    fail_cnt = 0
    predicted_q_val_matrix = np.zeros((19,19,4,3))
    log_matrix = np.zeros((19,19,2), dtype=int) #record the number of success and fails in each pos.
    total_reset_to_run = (8*8*4+4)*4 + 5 #make sure it go through one iteration
    first_agent_init_pos = env.get_init_agent_loc()
    agent_init_pos = env.get_init_agent_loc()

    #for i in range(19):
    #    for j in range(19):
    #        print(i,j,env.check_if_door(i,j))
    #exit()
    end_time = time.time()
    print("Env setup time: ", end_time-start_time)
    if verbose: print("Door locations:", door_loc_list)
    start_time = time.time()
    while reset_cnt<=total_reset_to_run:

        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 1, 2)
        #obs = np.unsqueeze(obs, 0)
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)
        obs = obs/255.0
        obs = torch.from_numpy(obs)
        obs = obs.to(device)

        with torch.no_grad():
            output = model(obs, add_rand=rand_noise_injection, this_device=device)
        if per_eps_steps==0: predicted_q_val_matrix[agent_init_pos[0], agent_init_pos[1], agent_init_pos[2], :] = output.cpu().numpy().flatten()    
        
        action = output.argmax(dim=1)
        action = action.cpu().item()

        if sample_from_dist:
            
            sample_rate = 1 #set to 1 if always sample from distribution
            one_rand = random.uniform(0,1)
            if one_rand<sample_rate:
                #action = get_action_from_distribution(output)
                action = get_action_from_torch_categorical(output, use_normed_value=True)
        

        obs, rewards, dones, info = env.step(action)
        #env.render()
        #if reset_cnt==583 or reset_cnt==711:
        #    env.render()
        if generate_vector_field: dones = 1
        per_eps_steps += 1
        if dones:
            if verbose and sample_from_dist: print("using sample_from_dist")
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] += 1
            #if verbose: print("Done at loc:", env.get_init_agent_loc())
            #print(log_matrix[:,:,0])
            obs = env.reset()
            per_eps_steps = 0
            reset_cnt += 1
            print_fail = 1
            #break
            agent_init_pos = env.get_init_agent_loc()
            if first_agent_init_pos==agent_init_pos: 
                #one iteration has been finished.
                break
            
        if per_eps_steps>90 and print_fail:
            log_matrix[agent_init_pos[0], agent_init_pos[1], 1] += 1
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] -= 1
            fail_cnt += 1
            print_fail = 0
            if verbose: print("Fail to finish at pos:", agent_init_pos[0], agent_init_pos[1], "reset cnt: ", reset_cnt, ". Total fail cnt: ", fail_cnt)
    #exit()
    end_time = time.time()
    print("Env running time: ", end_time-start_time)
    if verbose: print(log_matrix[:,:,0])
    goal_loc_i, goal_loc_j = env.get_goal_loc()
    log_matrix[goal_loc_i,goal_loc_j,0] = 0
    log_matrix[goal_loc_i,goal_loc_j,1] = 4
    predicted_q_val_matrix[goal_loc_i,goal_loc_j,:,:] = 0
        
    if plot_heat:
        plot_log_matrix(log_matrix)
    if generate_vector_field: plot_vector_field(predicted_q_val_matrix, use_normed_vector=False)
    if log_to_list:
        #print("Appending result to list")
        one_datapoint = {'door_loc': door_loc, 'fail_cnt': fail_cnt, 'log_matrix':log_matrix}
        result_list.append(copy.deepcopy(one_datapoint))
    print("total failed: ",fail_cnt)
    return fail_cnt, log_matrix

def plot_q_values(train_predicted_q_values, test_predicted_q_values, train_ground_truth, test_ground_truth):




    x_ticks_pos = np.arange(0,159,20)
    #x_ticks_pos = np.arange(0, 319, 20)
    #x_ticks_lables = ['0', '200k', '400k', '600k', '800k','1M']

    #x_ticks_pos = np.arange(0,11999,2000)
    #x_ticks_lables = ['0', '2M', '4M', '6M', '8M','10M']

    '''
    for zoom into 200k steps
    '''
    #x_ticks_pos = np.arange(0,240,40)
    #x_ticks_lables = ['0', '40k', '80k', '120k', '160k','200k']

    '''
    for zoom into 500k ticks
    '''
    #x_ticks_pos = np.arange(0,600,100)
    #x_ticks_lables = ['0', '100k', '200k', '300k', '400k','500k']





    #For plotting one reward
    fig1, ax1= plt.subplots()
    ax1.plot([s for s in range (len(train_predicted_q_values))],train_predicted_q_values,  alpha=0.5, color='red', label='Training q-values', linewidth = 1)
    ax1.plot([s for s in range(len(train_ground_truth))], train_ground_truth, alpha=0.5, color='green',
             label='Training Ground Truth', linewidth=1)
    #ax1.fill_between([s for s in range (len(avg_1))], avg_1-sd_1, avg_1+sd_1, color='red', alpha=0.3)
    ax1.plot([s for s in range (len(test_predicted_q_values))],test_predicted_q_values,  alpha=0.5, color='blue', label='Test q-valus', linewidth = 1)
    ax1.plot([s for s in range(len(test_ground_truth))], test_ground_truth, alpha=0.5, color='orange',
             label='Test Ground Truth', linewidth=1)
    # ax1.fill_between([s for s in range (len(avg_2))], avg_2-sd_2, avg_2+sd_2, color='blue', alpha=0.3)
    ax1.set_xticks(x_ticks_pos)
    #ax1.set_xticklabels(x_ticks_lables)
    fig1.set_size_inches(10, 5)
    #plt.figure(figsize=(10,5))
    #plt.title("Prediction Accuracy")
    plt.title("Predicted q values by regression network")


    for ax in fig1.get_axes():

        #ax.legend(loc='lower right')
        ax.legend(loc='upper right')
        #ax.set_ylabel("Accuracy (%)")
        ax.set_ylabel("q values")
        #ax.set_ylim([-60,55])
        ax.set_xlabel("Epoch")
        #ax.set(xlabel='Epoch', ylabel='Accuracy')
    #plt.grid(linestyle='dotted')
    plt.grid(axis='y',linestyle='dotted')
    #plt.savefig('prediction_acc.png')
    plt.savefig('regression_q_values.png')
def get_ground_truth():

    inputs, targets = load_all_datapoints(file_path, normalize_q=False)





    return ground_truth_avg
class Net_a(nn.Module): # Ar#2 structure
    def __init__(self):
        super(Net_a, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 4)
        self.bn2 = nn.BatchNorm2d(64)
        #self.conv3 = nn.Conv2d(64, 64, 3, 1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 128)
        #self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = F.relu(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc3(x)
        return x

class Net_gamma(nn.Module):#Ar4
    def __init__(self):
        super(Net_gamma, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)



    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
         
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        #x = F.log_softmax(x, dim=1)
        return x

class Net_gammaX2(nn.Module):#Ar#4-5 structure
    def __init__(self):
        super(Net_gammaX2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*2, 8, 4)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.conv2 = nn.Conv2d(64*2, 128*2, 4, 2)
        self.bn2 = nn.BatchNorm2d(128*2)
        self.conv3 = nn.Conv2d(128*2, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
         
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        #x = F.log_softmax(x, dim=1)
        return x


class Net_gammaX25(nn.Module):  # Ar#4-5-5 structure
    def __init__(self):
        super(Net_gammaX25, self).__init__()
        self.conv1 = nn.Conv2d(3, 64 * 4, 8, 4)
        self.bn1 = nn.BatchNorm2d(64 * 4)
        self.conv2 = nn.Conv2d(64 * 4, 128 * 2, 4, 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.conv3 = nn.Conv2d(128 * 2, 128*2, 3, 1)
        self.bn3 = nn.BatchNorm2d(128*2)
        # self.dropout1 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.sigmoid(x)
        x = F.relu(x)
        # x = self.dropout1(x)

        x = torch.flatten(x, 1)
        # print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)

        x = self.fc2(x)
        # print(x)
        # x = F.relu(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc3(x)

        # x = F.log_softmax(x, dim=1)
        return x

class Net_gammaX4(nn.Module): #Ar#5 structure
    def __init__(self):
        super(Net_gammaX4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*4, 8, 4)
        self.bn1 = nn.BatchNorm2d(64*4)
        self.conv2 = nn.Conv2d(64*4, 128*4, 4, 2)
        self.bn2 = nn.BatchNorm2d(128*4)
        self.conv3 = nn.Conv2d(128*4, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
         
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        #x = F.log_softmax(x, dim=1)
        return x



class Net_delta(nn.Module): #Ar#1 structure
    def __init__(self):
        super(Net_delta, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, 4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 4)
        self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(256, 128, 3, 1)
        #self.bn3 = nn.BatchNorm2d(128)
        #self.conv4 = nn.Conv2d(128, 128, 3, 2)
        #self.bn4 = nn.BatchNorm2d(128)        
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = F.sigmoid(x)
        #x = F.relu(x)
        #x = self.dropout1(x)

        #x = self.conv4(x)
        #x = self.bn4(x)
        #x = F.sigmoid(x)
        #x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.dropout2(x)
         
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        #x = F.log_softmax(x, dim=1)
        return x


class Net_default(nn.Module): #DQN structure
    def __init__(self):
        super(Net_default, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        
        if add_rand:
            #print(torch.mean(x))
            x_shape = np.shape(x)
            rand = torch.empty(x_shape).normal_(mean=0, std=0.3).to(this_device)
            #print(np.shape(rand))
            x = x + rand
 
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        #x = F.log_softmax(x, dim=1)
        return x

class Net_0(nn.Module):
    def __init__(self):
        super(Net_0, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4) #in_channels, out_channels, kernel_size, stride
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        #self.dropout1 = nn.Dropout(0.7)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        #x = F.log_softmax(x, dim=1)
        return x

'''
class Net_0(nn.Module):
    def __init__(self):
        super(Net_0, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*16*32, num_actual_actions)
        #self.fc2 = nn.Linear(128, num_actual_actions)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        #x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.dropout2(x)
        #x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x
'''


class Net_1(nn.Module): #Ar#4 structure #this is designed to be larger than net_default
    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        #x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval = 600
    #loss_func = nn.BCELoss()
    #loss_func = nn.MSELoss()
    #loss_func = nn.CrossEntropyLoss()
    loss_func = nn.L1Loss()
    #loss_func = nn.SmoothL1Loss(beta=0.001)

    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        optimizer.zero_grad()
        output = model(data, add_rand=False)
        #print(target)
        #output = output.argmax(dim=1, keepdim=True)
        #target = target.argmax(dim=1, keepdim=True)
        #print(target)
        loss = loss_func(output, target)
        #loss = F.nll_loss(output, target)
        #print("check this output:")
        #print(data)
        #print(output)
        #print(target)
        #loss = loss_func(output, target)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        #if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
    print(train_loss, len(train_loader))
    #print("Train loss", train_loss)
    train_loss /= len(train_loader)
    print('=============================Train Epoch: {} \tLoss: {:.6f}='.format(epoch, train_loss))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    predicted_q_values = 0
    ground_truth_q_values = 0
    correct = 0
    #loss_func = nn.BCELoss()
    #loss_func = nn.MSELoss()
    #loss_func = nn.CrossEntropyLoss()
    loss_func = nn.L1Loss()
    #loss_func = nn.SmoothL1Loss(beta=0.001)
    if args.run_test:
        wrong_pred_log = np.zeros((6,6))

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data, add_rand=False)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            predicted_q_values += np.mean(output.detach().to('cpu').numpy())
            ground_truth_q_values+= np.mean(target.detach().to('cpu').numpy())

            #print(output)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred = output.argmax(dim=1)  # get the index of the max log-probability
            #target_label = target.argmax(dim=1, keepdim=True)
            #print(pred)
            #print(target)
            #print(pred.eq(target_label).sum())
            #correct += pred.eq(target.view_as(pred)).sum().item()
            #correct += pred.eq(target).sum().item()

            if args.run_test:
                #print("Checking this run: ")
                #print(target)
                #print(output)
                #wrong_pred_log = np.zeros((6,6))
                #pred_actions = pred.cpu().numpy()
                #target_actions = target.cpu().numpy()
                #for i in range(pred_actions.size):
                    #if not pred_actions[i] == target_actions[i]:
                        #wrong_pred_log[pred_actions[i], target_actions[i]] += 1
                np.set_printoptions(precision=3)
                np.set_printoptions(suppress=True)
                print(target.cpu().numpy()[0:100])
                print(output.cpu().numpy()[0:100])
                print(batch_idx)

    #if args.run_test:
        #pred_error_details(wrong_pred_log)

    print("test loader len: ", len(test_loader))
    #acc_percent = 100. * correct / len(test_loader.dataset)
    #print(len(test_loader.dataset))
    #test_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader)
    predicted_q_values /= len(test_loader)
    ground_truth_q_values/= len(test_loader)
    print("Test loss: ", test_loss)
    #print("Avg test loss", test_loss/len(test_loader.dataset))
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), acc_percent))

    return test_loss, test_loss, predicted_q_values, ground_truth_q_values

def check_all_door_log(log_dir):
    load_test = np.load(log_dir, allow_pickle=True)
    fail_cnt_list = []
    
    for i in load_test:
        fail_cnt_list.append(i['fail_cnt'])
        #print(i['door_loc'], i['fail_cnt'])
        if i['door_loc']==[[(9, 1), (9, 11)], [(1, 9), (10, 9)]]:
            print((1039-i['fail_cnt'])/1039)
            plot_log_matrix(i['log_matrix'])

    fail_cnt_list = np.asarray(fail_cnt_list)-4 #in this setting, the target location creates extra 4 fails for all runs
    fail_rate_list = fail_cnt_list/1039
    print("Total datapoints: ", np.shape(fail_rate_list))
    print("Mean fail rate: ", np.mean(fail_rate_list))
    print("Fail rate std: ", np.std(fail_rate_list))
    
    plt.hist(fail_rate_list)
    plt.xlabel('Fail Rate')
    plt.ylabel('Count')
    plt.show()


if args.check_log:
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_1_2.npy'
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_2_noise_inference.npy'
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_regression_8_config_trained_net_0.npy'
    all_door_log_dir = 'all_door_performance_log_exact_q_learning_regression_8_config_trained_net_0.npy'
    #all_door_log_dir = 'temp_log.npy'
    check_all_door_log(all_door_log_dir)
    exit()

#model_save_folder = "./model_save/q_learning_minigrid/"
#model_save_title = "regression_exact_q_fourroomssetconfigs_gamma09_net_gamma_50pc_training_run3"
model_save_folder = "/datalake/homes/atefeh/q-learning_generalization_experiments_results/model_save/regression_models/"

model_save_title = "regression_exact_q_fourroomssetconfigs_net_default_duplicating_xueyuan_experiments_run5_100Percent_A4-5-5Structure"
#model_save_title = "testing_q-values"
#model_save_title = "regression_exact_q_fourroomssetconfigs_net_default"
model_save_dir = model_save_folder+model_save_title+".pt"
#model_save_dir = "./model_save/q_learning_minigrid/regression_exact_q_fourrooms_gamma09_net_0_dataSplitTest.pt"
#env_name = 'MiniGrid-Empty-Random-6x6-v0'
env_name = 'MiniGrid-FourRoomsCustom-v0'
# model = Net_gamma() #my first set of experiments
#model = Net_default()
#model = Net_delta()
#model = Net_a()
#model = Net_gammaX4()
#model = Net_gammaX2()
model = Net_gammaX25()
model_class = Net_gammaX25
print("Model Saving Dir:", model_save_dir)

if args.check_all_door_pos:
    #check_model_all_init_pos(model_save_name=model_save_name, env_name=env_name)
    check_all_doors = 1
    multi_thread_proc = 1
    #check all possible door location. Will take a long time to finish.
    #if args.check_one_config: check_all_doors=0

    if check_all_doors:
        if multi_thread_proc:
            import threading
            max_thread_num = 1
            all_door_log_save_dir = 'all_door_performance_log_exact_q_learning_regression_8_config_trained_net_default_action_from_torch_categorical_test_3.npy'
            all_door_log = []
            total_time = 0
            thread_list = []
            for door_a_i in range(4):
              for door_a_j in range(4):
                for door_b_i in range(4):
                    for door_b_j in range(4):
                        door_loc_temp = [[(9,door_a_i*2+4),(9,door_a_j*2+11)], [(door_b_i*2+5,9),(door_b_j*2+15,9)]]#[[(9,door_a_i*2+5),(9,door_a_j*2+15)], [(door_b_i*2+5,9),(door_b_j*2+14,9)]]
                        if len(thread_list)<max_thread_num:
                            device_id = "cuda:"+str((len(thread_list)+2)%4) #separate the load into all GPUs
                            print("Running on device: ", device_id)
                            thread_temp = threading.Thread(target=check_model_all_init_pos, args=(model_class, model_save_dir, env_name, door_loc_temp, True, True, True, False, all_door_log, device_id, False))
                            thread_temp.start()
                            thread_list.append(thread_temp)
                        else:
                            start_time = time.time()
                            print("Waiting For Current Threads to Finish")
                            for t in thread_list: t.join()
                            thread_list = []
                            print("Saving the dataset to ", all_door_log_save_dir)
                            end_time = time.time()
                            total_time += (end_time-start_time)
                            print("Iteration at ", door_a_i, door_a_j, door_b_i, door_b_j, "total time used: ", total_time)
                            np.save(all_door_log_save_dir, all_door_log)
                            print("Current Threads Finished")
                            device_id = "cuda:"+str((len(thread_list)+2)%4) #separate the load into all GPUs
                            print("Running on device: ", device_id)
                            thread_temp = threading.Thread(target=check_model_all_init_pos, args=(model_class, model_save_dir, env_name, door_loc_temp, True, True, True, False, all_door_log, device_id, False))
                            thread_temp.start()
                            thread_list.append(thread_temp)

            #print(all_door_log)
            for t in thread_list: t.join()
            thread_list = []

            print("Saving the dataset to ", all_door_log_save_dir)
            np.save(all_door_log_save_dir, all_door_log)
    exit()

#file_path = ["Q_learning_dataset_PongNoFrameskip-v4_seed_1.npy"]
#file_dir = "/data/xushe/rl_dev/data_save/q_learning_doorkey8x8/"
#file_dir = "/data/xushe/rl_dev/data_save/q_learning_empty6x6_exact_q/"
#file_dir = "/data/xushe/rl_dev/data_save/q_learning_fourrooms_gamma09_exact_q/"
file_dir = "/data/xushe/rl_dev/data_save/Q_learning_dataset_FourRoomsSetConfigs_gamma09_exact_q_table_2/"
#file_dir = "data_save/experiments/Q_learning_dataset_FourRoomsSetConfigs_Random_door_pos_gamma09_exact_q_table_experiment16/"
file_path = get_file_list(file_dir)
#inputs, targets = get_npy_data(file_path)
#if not args.agent_play:
#inputs, targets = select_datapoints(file_path, percentage=50)
inputs, targets = load_all_datapoints(file_path, normalize_q=False)
data_size = len(inputs)
print(data_size)
#print(inputs)
#print(targets)
train_dataset = TensorDataset(inputs, targets)

#load another folder as test set
file_dir = "/data/xushe/rl_dev/data_save/Q_learning_dataset_FourRoomsSetConfigs_gamma09_exact_q_table/"
#file_dir ="data_save/experiments/test_data/Q_learning_dataset_FourRoomsSetConfigs_Random_door_pos_gamma09_exact_q_table_test1/"
file_path = get_file_list(file_dir)
#inputs, targets = load_all_datapoints(file_path, normalize_q=False)
## Atefeh: use 30% when redoing experiments for Fig 3 in the paper
inputs, targets = select_datapoints(file_path, percentage=30, normalize_q=False)
test_set = TensorDataset(inputs, targets)
train_set = train_dataset


g_cpu = torch.Generator()
g_cpu.manual_seed(2147483647) #use manual seed to make the data spliting the same for each run
reduced_to_ratio = 0.5
train_set, _ = torch.utils.data.random_split(train_dataset, [int(reduced_to_ratio*data_size), (data_size-(int(reduced_to_ratio*data_size)))], generator=g_cpu)
print("Reducing the training set to ", len(train_set))


train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048)


'''
#split train and test from one dataset
data_size = len(inputs)
print(data_size)
#print(inputs)
#print(targets)
train_dataset = TensorDataset(inputs, targets)
g_cpu = torch.Generator()
g_cpu.manual_seed(2147483647) #use manual seed to make the data spliting the same for each run

if args.run_test:
    train_set, test_set = torch.utils.data.random_split(train_dataset, [int(0.3*data_size), (data_size-(int(0.3*data_size)))], generator=g_cpu)
else:
    print("Split for ", int(0.3*data_size), "and", (data_size-(int(0.7*data_size))))
    train_set, test_set = torch.utils.data.random_split(train_dataset, [int(0.3*data_size), (data_size-(int(0.3*data_size)))], generator=g_cpu)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024)
'''
#inputs, targets = load_with_uncrucual_cases_processed(file_path, percentage=10, balance_label=True)
#exit()


#print(train_set)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
#inputs, targets = get_npy_data(file_path[-1:])
#test_dataset = TensorDataset(inputs, targets)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda:3")
#exit()
print("Total Number of Parameters in the model:", count_parameters(model))
trained_model_to_load_path = ""
#load_partial_weight(model, trained_model_to_load_path)



if args.run_test or args.agent_play:
    #model = Net_0()
    model.load_state_dict(torch.load(model_save_dir))
    model.eval()
    #model = model.to(device)

model = model.to(device)
#for para in model.parameters():
#    print(para)
    
#optimizer = optim.Adadelta(model.parameters(), lr=5e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,momentum=0.9, weight_decay=5e-4)

if args.agent_play:
    #env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=10)
    #env = VecFrameStack(env, n_stack=4)
    env = gym.make(env_name)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    obs = env.reset()
    while True:
        #print(np.shape(obs))
        
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 1, 2)
        #obs = np.unsqueeze(obs, 0)
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)
        obs = obs/255.0
        obs = torch.from_numpy(obs)
        obs = obs.to(device)
        with torch.no_grad():
            output = model(obs)
        action = output.argmax(dim=1)
        action = action.cpu().item()

        #if action==1: action=2
        #elif action==2: action=3
        action = action
        #print(action, end=" ")
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()
        
    exit()

scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
train_loss = []
test_loss = []
train_predicted_q_values = []
test_predicted_q_values = []
train_ground_truth = []
test_ground_truth = []
save_iter = 100
for epoch in range(1, 300 + 1):
    
    if args.run_test:
        test(model, device, test_loader)
        break
    else:
        per_epoch_time = time.time()
        per_train_time = time.time()
        train(model, device, train_loader, optimizer, epoch)
        print("Training takes:", time.time()-per_train_time)

        _, loss_temp, qvalues_temp, ground_truth_tmp = test(model, device, train_loader)
        train_loss.append(loss_temp)
        train_predicted_q_values.append(qvalues_temp)
        print ("train_ground_truth", ground_truth_tmp)
        train_ground_truth.append(ground_truth_tmp)
        _, loss_temp, qvalues_temp, ground_truth_tmp = test(model, device, test_loader)
        test_loss.append(loss_temp)
        test_predicted_q_values.append(qvalues_temp)
        print("test_ground_truth", ground_truth_tmp)
        test_ground_truth.append(ground_truth_tmp)
        scheduler.step()
        print("This epoch takes:", time.time()-per_epoch_time)


    if epoch%save_iter==0 and not epoch==0:
        print("Saving model to ", model_save_dir)
        torch.save(model.state_dict(), model_save_dir)   
        print("Saving Done.")

if not args.run_test:
    print("Saving model to ", model_save_dir)
    torch.save(model.state_dict(), model_save_dir)   
    print("Saving Done.")

train_loss_save_dir = model_save_folder+model_save_title+"_train_loss"+".csv"
print("Training set size ", len(train_set))
print("Total Number of Parameters in the model:", count_parameters(model))

with open(train_loss_save_dir, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(train_loss)

test_loss_save_dir = model_save_folder+model_save_title+"_test_loss"+".csv"

with open(test_loss_save_dir, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(test_loss)






if not args.run_test:
    #plot_save_dir = "/datalake/homes/atefeh/q-learning_generalization_experiments_results/plots/3d_run_100percent-test-dataset"
    plot_save_dir = "/datalake/homes/atefeh/q-learning_generalization_experiments_results/plots/"
    plot_title = "regression_exact_q_fourroomssetconfigs_net_default_duplicating_xueyuan_experiments_run5_100Percent_A4-5-5Structure_loss"
    #plot_title = "testing_q-values"
    plot_save_path = plot_save_dir+plot_title
    plt.figure(figsize=(10,5))
    plt.title("Loss Value During Training")
    plt.plot(test_loss,label="test")
    plt.plot(train_loss,label="train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(plot_save_path + ".png")


    #plot_q_values(train_predicted_q_values, test_predicted_q_values, train_ground_truth, test_ground_truth)



