import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
from torch.utils.data import Dataset, TensorDataset
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3 import DQN, A2C, PPO
from gym_minigrid.wrappers import *

from os import listdir
from os.path import isfile, join
import time
import copy
import torch.nn.functional as F
import argparse
import csv

parser = argparse.ArgumentParser(description='Q Learning Trainer')
parser.add_argument('--run_test', '-rt', action='store_true', default=False)
parser.add_argument('--agent_play', '-ap', action='store_true', default=False) #use the agent to play the game
parser.add_argument('--check_all_door_pos', '-cad', action='store_true', default=False)
parser.add_argument('--check_log', '-cl', action='store_true', default=False)
args = parser.parse_args()

import collections


num_actual_actions = 3

def plot_log_matrix(log_matrix):
    log_matrix[:,:,0] = np.rot90(log_matrix[:,:,0],k=3)
    log_matrix[:,:,0] = np.flip(log_matrix[:,:,0],axis=1)
    log_matrix[:,:,1] = np.rot90(log_matrix[:,:,1],k=3)
    log_matrix[:,:,1] = np.flip(log_matrix[:,:,1],axis=1)

    '''
    success_matrix = log_matrix[:,:,0]
    #print("matrix dim 1", log_matrix[:,:,1])
    #print("wall_pos_1", wall_pos)
    #wall_pos = np.argwhere(wall_pos==0)
    #print(wall_pos)
    plt.imshow(success_matrix)
    plt.colorbar(orientation='horizontal')
    plt.title("Agent Success Per Positions")
    plt.show()
    '''
    
    wall_pos = log_matrix[:,:,0]+log_matrix[:,:,1]
    to_plot_matrix = log_matrix[:,:,0] + 1
    #print(to_plot_matrix)
    to_plot_matrix[wall_pos==0] = 0
    #print(to_plot_matrix)
    plt.imshow(to_plot_matrix)
    plt.colorbar(orientation='horizontal')
    plt.title("Agent Success Per Positions")
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

    show_input_image = 0
    
    for i in range(len(data_path)):
        data_temp = np.load(data_path[i], allow_pickle=True)
        
        inputs_temp = np.array([d["state"] for d in data_temp])
        targets_temp = np.array([d["Q_per_action"] for d in data_temp])
        if show_input_image:
            print("Showing the input images...")
            plt.imshow(inputs_temp[0,:,:,:])
            plt.show()
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

def target_checker(target):
    target = target.numpy()
    print("Histogram of the targets: ", np.histogram(target, bins=[0,1,2,3,4,5,6]))

def get_file_list(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        
    for i in range(len(onlyfiles)):
        onlyfiles[i] = data_dir+onlyfiles[i]
    #print(onlyfiles)

    return onlyfiles

def save_exact_q_to_csv(data):
    np.savetxt("fourrooms_exact_q_gamma09.csv", data, delimiter=",")
    print("Saving exact Q values to a file and exiting")
    exit()

def load_all_datapoints(data_dir):
    #get the top percentage of the data according to STD
    #data_list, targets = get_npy_data_unprocessed(file_path)
    ##### Atefeh: I changed file_path to data_dir because nothing was getting load
    data_list, targets = get_npy_data_unprocessed(data_dir)
    total_data_num = np.shape(data_list)[0]
    print("data list",len(data_list))
    print ("target", np.shape(targets))
    #std_argsort.reverse()

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
    #save_exact_q_to_csv(targets)
    targets = torch.from_numpy(targets)
    targets = targets.argmax(dim=1, keepdim=True)
    targets = torch.squeeze(targets)

    print(np.shape(inputs), np.shape(targets))
    #print(targets)
    #data_checker(inputs)
    target_checker(targets)

    return torch.from_numpy(inputs), targets

def select_datapoints(data_dir, percentage):
    #get the top percentage of the data according to STD
    #data_list, targets = get_npy_data_unprocessed(file_path)
    ##### Atefeh: I changed file_path to data_dir because nothing was getting load
    data_list, targets = get_npy_data_unprocessed(data_dir)
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
    #### I have change axis=0 because I was getting an error
    #print("average std of selected datapoints: ", np.mean(np.std(targets, axis=0)))
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
    targets = torch.from_numpy(targets)
    targets = targets.argmax(dim=1, keepdim=True)
    targets = torch.squeeze(targets)

    print(np.shape(inputs), np.shape(targets))
    #print(targets)
    #data_checker(inputs)
    target_checker(targets)

    return torch.from_numpy(inputs), targets

def load_with_uncrucual_cases_processed(data_dir, percentage, balance_label=False):
    #for the percentage below Q value std ranking, mark as the same label
    #inputs, targets = get_npy_data_unprocessed(file_path)
    ##### Atefeh: I changed file_path to data_dir because nothing was getting load
    inputs, targets = get_npy_data_unprocessed(data_dir)
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

import ray
ray.init()
@ray.remote
class Simulator(object):
    def __init__(self, env_name, door_loc_list=None):
        #self.env = gym.make("Pong-v0")
        self.env = gym.make(env_name, door_loc_list=door_loc_list)
        self.env = RGBImgObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)
        #self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def get_init_agent_loc(self):
        return self.env.get_init_agent_loc()

    def get_goal_loc(self):
        return self.env.get_goal_loc()

def ray_multienv_check_model_all_init_pos(model_func, model_save_name, env_name, door_loc=None, plot_heat=True, verbose=False, log_to_list=False, result_list=None, device_id=None):
    #print(door_loc)
    start_time = time.time()
    if door_loc is None:
        door_loc_list = [[(9,1),(9,17)], [(8,9),(10,9)]]
    else:
        door_loc_list = door_loc
    env = gym.make(env_name, door_loc_list=door_loc_list)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    simulator = Simulator.remote(env_name=env_name, door_loc_list=door_loc_list)

    simulator_obs = []
    simulator_obs.append(simulator.reset.remote())
    obs = ray.get(simulator_obs)[0]

    device = torch.device(device_id)    
    model = model_func()
    model.load_state_dict(torch.load(model_save_name))
    model.eval()
    model.to(device)

    sample_action_from_distribution = 1
    rand_noise_injection = 0
    per_eps_steps = 0
    reset_cnt = 0
    print_fail = 1
    fail_cnt = 0
    log_matrix = np.zeros((19,19,2), dtype=int) #record the number of success and fails in each pos.
    total_reset_to_run = (8*8*4+4)*4 + 5 #make sure it go through all possible starting states

    simulator_stats = []
    simulator_stats.append(simulator.get_init_agent_loc.remote())
    first_agent_init_pos = ray.get(simulator_stats)[0]
    agent_init_pos = ray.get(simulator_stats)[0]

    #for i in range(19):
    #    for j in range(19):
    #        print(i,j,env.check_if_door(i,j))
    #exit()
    end_time = time.time()
    if verbose: print("Env on device", device_id, "setup time: ", end_time-start_time)
    
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
        action = output.argmax(dim=1)
        action = action.cpu().item()
        if sample_action_from_distribution:
            action = get_action_from_torch_categorical(output, use_normed_value=True)

        obs, rewards, dones, info = env.step(action)


        #print(action, end=" ")
        #obs, rewards, dones, info = env.step(action)
        simulator_obs = []
        simulator_obs.append(simulator.step.remote(action))
        obs, rewards, dones, info = ray.get(simulator_obs)[0]
        #env.render()
        #if reset_cnt==583 or reset_cnt==711:
        #    env.render()

        per_eps_steps += 1
        if dones:
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] += 1
            #print("Done at loc:", env.get_init_agent_loc())
            #print(log_matrix[:,:,0])
            #obs = env.reset()

            simulator_obs = []
            simulator_obs.append(simulator.reset.remote())
            obs = ray.get(simulator_obs)[0]

            per_eps_steps = 0
            reset_cnt += 1
            print_fail = 1
            #break
            #agent_init_pos = env.get_init_agent_loc()
            simulator_stats = []
            simulator_stats.append(simulator.get_init_agent_loc.remote())
            agent_init_pos = ray.get(simulator_stats)[0]     
            #print("first pos and cur pos: ",first_agent_init_pos, agent_init_pos)
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
    print("Env on device", device_id, "running time: ", end_time-start_time, "total_fail: ", fail_cnt)
    if verbose: print(log_matrix[:,:,0])
    #goal_loc_i, goal_loc_j = env.get_goal_loc()
    simulator_stats = []
    simulator_stats.append(simulator.get_goal_loc.remote())
    goal_loc_i, goal_loc_j = ray.get(simulator_stats)[0]    
    log_matrix[goal_loc_i,goal_loc_j,0] = 0
    log_matrix[goal_loc_i,goal_loc_j,1] = 4
    if plot_heat:
        plot_log_matrix(log_matrix)

    if log_to_list:
        #print("Appending result to list")
        one_datapoint = {'door_loc': door_loc, 'fail_cnt': fail_cnt, 'log_matrix':log_matrix}
        result_list.append(copy.deepcopy(one_datapoint))
 
    return fail_cnt, log_matrix




def check_model_all_init_pos(model_save_name, env_name, door_loc=None, plot_heat=True, verbose=False, log_to_list=False, result_list=None, device_id=None, generate_vector_field=False):
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

    model = Net_default()
    model.load_state_dict(torch.load(model_save_name))
    model.eval()
    model.to(device)

    obs = env.reset()
    rand_noise_injection = 0
    sample_action_from_distribution = 0
    per_eps_steps = 0
    reset_cnt = 0
    print_fail = 1
    fail_cnt = 0
    time_each_part = 0
    if time_each_part:
        timing_obs_transform = []
        timing_action_taking = []
        timing_env_step = []

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
    
    start_time = time.time()
    while reset_cnt<=total_reset_to_run:
        if time_each_part: time_temp = time.time()
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 1, 2)
        #obs = np.unsqueeze(obs, 0)
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)
        obs = obs/255.0
        obs = torch.from_numpy(obs)
        obs = obs.to(device)
        if time_each_part: 
            #print("obs transform takes: ", time.time()-time_temp)
            timing_obs_transform.append(time.time()-time_temp)
            time_temp = time.time()

        with torch.no_grad():
            output = model(obs, add_rand=rand_noise_injection, this_device=device)
        #print(np.shape(output.cpu().numpy().flatten()))
        if per_eps_steps==0: predicted_q_val_matrix[agent_init_pos[0], agent_init_pos[1], agent_init_pos[2], :] = output.cpu().numpy().flatten()
        action = 0
        if sample_action_from_distribution:
            action = get_action_from_torch_categorical(output, use_normed_value=True)
        else:
            action = output.argmax(dim=1)
            action = action.cpu().item()

        if time_each_part: 
            #print("action taking  takes: ", time.time()-time_temp)
            timing_action_taking.append(time.time()-time_temp)
            time_temp = time.time()
        obs, rewards, dones, info = env.step(action)
        if time_each_part: 
            #print("environme step takes: ", time.time()-time_temp)
            timing_env_step.append(time.time()-time_temp)
            time_temp = time.time()        
        #env.render()
        #if reset_cnt==583 or reset_cnt==711:
        #    env.render()
        if generate_vector_field: dones = 1

        per_eps_steps += 1
        if dones:
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] += 1
            #print("Done at loc:", env.get_init_agent_loc())
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
            if time_each_part: 
                print("obs transform takes: ", np.mean(timing_obs_transform))
                print("action taking takes: ", np.mean(timing_action_taking))
                print("env steping takes: ", np.mean(timing_env_step))            
        if per_eps_steps>90 and print_fail:
            log_matrix[agent_init_pos[0], agent_init_pos[1], 1] += 1
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] -= 1
            fail_cnt += 1
            print_fail = 0
            if verbose: print("Fail to finish at pos:", agent_init_pos[0], agent_init_pos[1], "reset cnt: ", reset_cnt, ". Total fail cnt: ", fail_cnt)
    #exit()
    end_time = time.time()
    print("Env running time: ", end_time-start_time, "total_fail: ", fail_cnt)
    if verbose: print(log_matrix[:,:,0])
    goal_loc_i, goal_loc_j = env.get_goal_loc()
    log_matrix[goal_loc_i,goal_loc_j,0] = 0
    log_matrix[goal_loc_i,goal_loc_j,1] = 4
    predicted_q_val_matrix[goal_loc_i,goal_loc_j,:,:] = 0
    if generate_vector_field: plot_vector_field(predicted_q_val_matrix, use_normed_vector=False)    
    if plot_heat:
        plot_log_matrix(log_matrix)

    if log_to_list:
        #print("Appending result to list")
        one_datapoint = {'door_loc': door_loc, 'fail_cnt': fail_cnt, 'log_matrix':log_matrix}
        result_list.append(copy.deepcopy(one_datapoint))
 
    return fail_cnt, log_matrix


def check_all_door_log(log_dir):
    load_test = np.load(log_dir, allow_pickle=True)
    fail_cnt_list = []
    
    for i in load_test:
        fail_cnt_list.append(i['fail_cnt'])
        #print(i['door_loc'], i['fail_cnt'])
        #if i['door_loc']==[[(9, 3), (9, 12)], [(5, 9), (10, 9)]]:
        #    plot_log_matrix(i['log_matrix'])

    fail_cnt_list = np.asarray(fail_cnt_list)-4 #in this setting, the target location creates extra 4 fails for all runs
    fail_rate_list = fail_cnt_list/1039
    print("Total datapoints: ", np.shape(fail_rate_list))
    print("Mean fail rate: ", np.mean(fail_rate_list))
    print("Fail rate std: ", np.std(fail_rate_list))
    
    plt.hist(fail_rate_list)
    plt.xlabel('Fail Rate')
    plt.ylabel('Count')
    plt.show()


class Net_default(nn.Module):
    def __init__(self):
        super(Net_default, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=False, this_device="cuda:2"):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.sigmoid(x)   
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        if add_rand:
            #print(torch.mean(x))
            x_shape = np.shape(x)
            rand = torch.empty(x_shape).normal_(mean=0, std=1.0).to(this_device)
            #print(np.shape(rand))
            x = x + rand


        x = self.conv3(x)
        x = self.bn3(x)
        #x = F.sigmoid(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)        
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)



        x = self.fc2(x)
        #print(x)
        #x = F.relu(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)
        
        
        x = F.log_softmax(x, dim=1)
        return x


class Net_0(nn.Module):
    def __init__(self):
        super(Net_0, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4) #in_channels, out_channels, kernel_size, stride
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.bn3 = nn.BatchNorm2d(32)
        #self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.bn4 = nn.BatchNorm2d(64)        
        #self.dropout1 = nn.Dropout(0.3)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_actual_actions)
        #self.fc3 = nn.Linear(128, num_actual_actions)

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

        #x = self.conv4(x)
        #x = self.bn4(x)
        #x = F.sigmoid(x)
        #x = F.relu(x)
        #x = self.dropout1(x)


        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fc3(x)
        #print(x)
        #x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x



class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 8, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_actual_actions)

    def forward(self, x, add_rand=True, this_device="cuda:2"):
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
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        #print(np.shape(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

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

        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval = 600
    #loss_func = nn.BCELoss()
    #loss_func = nn.MSELoss()
    loss_func = nn.CrossEntropyLoss()
    use_l1_reg = 0
    l1_reg_lambda = 0.0005
    use_l2_reg = 0
    l2_reg_lambda = 1e-6

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        optimizer.zero_grad()
        output = model(data)
        #print(target)
        #output = output.argmax(dim=1, keepdim=True)
        #target = target.argmax(dim=1, keepdim=True)
        #print(target)
        loss = loss_func(output, target)
        #loss = F.nll_loss(output, target)

        if use_l1_reg:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            #print("original loss", loss, ", l1_norm", l1_norm)
            loss = loss + l1_reg_lambda * l1_norm
        if use_l2_reg:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #print("original loss", loss, ", l1_norm", l2_norm)
            loss = loss + l2_reg_lambda * l2_norm

        #print("check this output:")
        #print(data)
        #print(output)
        #print(target)
        #loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    #loss_func = nn.BCELoss()
    #loss_func = nn.MSELoss()
    loss_func = nn.CrossEntropyLoss()

    if args.run_test:
        wrong_pred_log = np.zeros((6,6))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss

            #print(output)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            #target_label = target.argmax(dim=1, keepdim=True)
            #print(pred)
            #print(target)
            #print(pred.eq(target_label).sum())
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target).sum().item()

            if args.run_test:
                #print("Checking this run: ")
                #print(target)
                #print(output)
                #wrong_pred_log = np.zeros((6,6))
                pred_actions = pred.cpu().numpy()
                target_actions = target.cpu().numpy()
                for i in range(pred_actions.size):
                    if not pred_actions[i] == target_actions[i]:
                        wrong_pred_log[pred_actions[i], target_actions[i]] += 1

    if args.run_test:
        pred_error_details(wrong_pred_log)


    acc_percent = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader)
    print("Test loss: ", test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc_percent))

    return acc_percent, test_loss

if args.check_log:
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_default_2.npy'
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_2_noise_inference.npy'
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_default_l1_reg_lambda_0d0005_action_sampling.npy'
    all_door_log_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_default_l2_reg_lambda_1e-6_action_sampling.npy'
    #all_door_log_dir = 'all_door_performance_log_exact_q_learning_regression_8_config_trained_net_0_action_from_dist_normed.npy'
    #all_door_log_dir = 'all_door_performance_log_DQN_8_config_trained_gamma09_buffer5e5_2e6steps.npy'
    #all_door_log_dir = 'temp_log.npy'
    check_all_door_log(all_door_log_dir)
    exit()

#model_save_folder = "./model_save/q_learning_minigrid/"
#model_save_title = "exact_q_prediction_fourroomssetconfigs_net_default_2"
#model_save_folder = "./model_save/prediction_models/"
model_save_folder = "/datalake/homes/atefeh/q-learning_generalization_experiments_results/model_save/prediction_models/"

model_save_title = "exact_q_prediction_fourroomssetconfigs_net_default_duplicating_xueyuan_experiments_run5_noL1regular"

#model_save_title = "exact_q_prediction_fourroomssetconfigs_net_default_l2_reg_lambda_1e-6"
model_save_dir = model_save_folder+model_save_title+".pt"
#model_save_dir = "./model_save/q_learning_minigrid/exact_q_prediction_fourroomssetconfigs_net_default_2.pt"
#env_name = 'MiniGrid-Empty-Random-6x6-v0'
env_name = 'MiniGrid-FourRoomsCustom-v0'
model = Net_default()
model_func = Net_default

print("Model Saving Dir:", model_save_dir)

if args.check_all_door_pos:
    #check_model_all_init_pos(model_save_name=model_save_name, env_name=env_name)
    check_all_doors = 1
    multi_thread_proc = 1
    use_ray = 0
    #check all possible door location. Will take a long time to finish.
    #if args.check_one_config: check_all_doors=0

    if check_all_doors:
        if use_ray:
            import threading
            max_thread_num = 2
            all_door_log_save_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_default_l2_reg_lambda_1e-6_action_sampling.npy'
            #all_door_log_save_dir = 'all_door_performance_log_DQN_8_config_trained_gamma09_buffer5e5_2e6steps.npy'
            all_door_log = []
            total_time = 0
            thread_list = []
            for door_a_i in range(4):
              for door_a_j in range(4):
                for door_b_i in range(4):
                    for door_b_j in range(4):
                        door_loc_temp = [[(9,door_a_i*2+1),(9,door_a_j*2+10)], [(door_b_i*2+1,9),(door_b_j*2+10,9)]]
                        if len(thread_list)<max_thread_num:
                            device_id = "cuda:"+str((len(thread_list))%4) #separate the load into all GPUs
                            print("Running on device: ", device_id)
                            thread_temp = threading.Thread(target=ray_multienv_check_model_all_init_pos, args=(model_func, model_save_dir, env_name, door_loc_temp, False, False, True, all_door_log, device_id))
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
                            device_id = "cuda:"+str((len(thread_list))%4) #separate the load into all GPUs
                            print("Running on device: ", device_id)                            
                            thread_temp = threading.Thread(target=ray_multienv_check_model_all_init_pos, args=(model_func, model_save_dir, env_name, door_loc_temp, False, False, True, all_door_log, device_id))
                            thread_temp.start()
                            thread_list.append(thread_temp)  
            print("Saving the dataset to ", all_door_log_save_dir)
            np.save(all_door_log_save_dir, all_door_log)                            

        elif multi_thread_proc:
            import threading
            max_thread_num = 1
            all_door_log_save_dir = 'all_door_performance_log_exact_q_learning_prediction_8_config_trained_net_2_noise_inference_test_test1.npy'
            all_door_log = []
            total_time = 0
            thread_list = []
            for door_a_i in range(4):
              for door_a_j in range(4):
                for door_b_i in range(4):
                    for door_b_j in range(4):
                        door_loc_temp = [[(9,door_a_i*2+1),(9,door_a_j*2+10)], [(door_b_i*2+1,9),(door_b_j*2+10,9)]]#[[(9,door_a_i*2+4),(9,door_a_j*2+11)], [(door_b_i*2+5,9),(door_b_j*2+15,9)]]#
                        if len(thread_list)<max_thread_num:
                            device_id = "cuda:"+str(len(thread_list)%4) #separate the load into all GPUs
                            print("Running on device: ", device_id)
                            thread_temp = threading.Thread(target=check_model_all_init_pos, args=(model_save_dir, env_name, door_loc_temp, True, True, True, all_door_log, device_id, True))
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
                            device_id = "cuda:"+str(len(thread_list)%4) #separate the load into all GPUs
                            print("Running on device: ", device_id)
                            thread_temp = threading.Thread(target=check_model_all_init_pos, args=(model_save_dir, env_name, door_loc_temp, False, False, True, all_door_log, device_id, True))
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
#file_dir = "/data/xushe/rl_dev/data_save/q_learning_fourrooms_exact_q/"
file_dir = "/data/xushe/rl_dev/data_save/Q_learning_dataset_FourRoomsSetConfigs_gamma09_exact_q_table_2/"
#file_dir = "data_save/experiments/Q_learning_dataset_FourRoomsSetConfigs_Random_door_pos_gamma09_exact_q_table_all_16_experiments/"
file_path = get_file_list(file_dir)
#inputs, targets = get_npy_data(file_path)
#if not args.agent_play:
#inputs, targets = select_datapoints(file_path, percentage=100)
inputs, targets = load_all_datapoints(file_path)



#inputs, targets = load_with_uncrucual_cases_processed(file_path, percentage=10, balance_label=True)
#exit()
data_size = len(inputs)
print(data_size)
#print(inputs)
#print(targets)
train_dataset = TensorDataset(inputs, targets)
g_cpu = torch.Generator()
g_cpu.manual_seed(2147483647) #use manual seed to make the data spliting the same for each run
#train_set, test_set = torch.utils.data.random_split(train_dataset, [int(0.90*data_size), (data_size-(int(0.90*data_size)))], generator=g_cpu)
#test_set = train_dataset
#train_set = train_dataset

#load another folder as test set
file_dir = "/data/xushe/rl_dev/data_save/Q_learning_dataset_FourRoomsSetConfigs_gamma09_exact_q_table/" #Q_learning_dataset_FourRoomsSetConfigs_gamma09_exact_q_table has different configuration than Q_learning_dataset_FourRoomsSetConfigs_gamma09_exact_q_table_2, so we use it for the test set.
#file_dir = "data_save/experiments/test_data/Q_learning_dataset_FourRoomsSetConfigs_Random_door_pos_gamma09_exact_q_table_test1/"
file_path = get_file_list(file_dir)
#inputs, targets = load_all_datapoints(file_path)
inputs, targets = select_datapoints(file_path, percentage=30)
test_set = TensorDataset(inputs, targets)
train_set = train_dataset

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024)
#print(train_set)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
#inputs, targets = get_npy_data(file_path[-1:])
#test_dataset = TensorDataset(inputs, targets)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda:1")
#exit()
print("Total Number of Parameters in the model:", count_parameters(model))
trained_model_to_load_path = ""
#load_partial_weight(model, trained_model_to_load_path)
if args.run_test or args.agent_play:
    #model = Net_default()
    #model = Net_0()
    model.load_state_dict(torch.load(model_save_dir))
    print("loading model from:", model_save_dir)

    model.eval()
    #model = model.to(device)

#for para in model.parameters():
#    print(para)
model = model.to(device)    

optimizer = optim.Adadelta(model.parameters(), lr=5e-2)
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,momentum=0.9, weight_decay=5e-4)



if args.agent_play:
    #door_loc_list = [[(9,7),(9,13)], [(2,9),(15,9)]]
    door_loc_list = [[(9,1),(9,12)], [(1,9),(14,9)]]
    print("Checking door location: ", door_loc_list)
    #env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=10)
    #env = VecFrameStack(env, n_stack=4)
    env = gym.make(env_name, door_loc_list=door_loc_list)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    obs = env.reset()
    per_eps_steps = 0
    reset_cnt = 0
    print_fail = 1
    fail_cnt = 0
    rand_noise_injection = True
    log_matrix = np.zeros((19,19,2), dtype=int) #record the number of success and fails in each pos.
    total_reset_to_run = (8*8*4+4)*4 + 5 #make sure it go through one iteration
    first_agent_init_pos = env.get_init_agent_loc()
    agent_init_pos = env.get_init_agent_loc()

    #for i in range(19):
    #    for j in range(19):
    #        print(i,j,env.check_if_door(i,j))
    #exit()
    
    while reset_cnt<=total_reset_to_run:
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
            output = model(obs, add_rand=rand_noise_injection)
        action = output.argmax(dim=1)
        action = action.cpu().item()

        #if action==1: action=2
        #elif action==2: action=3
        action = action
        #print(action, end=" ")
        obs, rewards, dones, info = env.step(action)
        #env.render()
        #if reset_cnt==583 or reset_cnt==711:
        #    env.render()

        per_eps_steps += 1
        if dones:
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] += 1
            #print("Done at loc:", env.get_init_agent_loc())
            #print(log_matrix[:,:,0])
            obs = env.reset()
            per_eps_steps = 0
            reset_cnt += 1
            print_fail = 1
            
            agent_init_pos = env.get_init_agent_loc()
            if first_agent_init_pos==agent_init_pos: 
                #one iteration has been finished.
                break
            
        if per_eps_steps>90 and print_fail:
            log_matrix[agent_init_pos[0], agent_init_pos[1], 1] += 1
            log_matrix[agent_init_pos[0], agent_init_pos[1], 0] -= 1
            fail_cnt += 1
            print_fail = 0
            print("Fail to finish at pos:", agent_init_pos[0], agent_init_pos[1], "reset cnt: ", reset_cnt, ". Total fail cnt: ", fail_cnt)
    #exit()
    print("Finished checking door location: ", door_loc_list)
    print(log_matrix[:,:,0])
    plot_log_matrix(log_matrix)  
    exit()

scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
train_acc = []
test_acc = []
train_loss = []
test_loss = []
for epoch in range(1, 150 + 1):
    
    if args.run_test:
        test(model, device, test_loader)
        break
    else:
        train(model, device, train_loader, optimizer, epoch)
        acc_temp, loss_temp = test(model, device, train_loader)
        train_acc.append(acc_temp)
        train_loss.append(loss_temp)
        acc_temp, loss_temp = test(model, device, test_loader)
        test_acc.append(acc_temp)
        test_loss.append(loss_temp)
        scheduler.step()

train_loss_save_dir = model_save_folder+model_save_title+"_train_loss"+".csv"
with open(train_loss_save_dir, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(train_loss)

test_loss_save_dir = model_save_folder+model_save_title+"_test_loss"+".csv"
with open(test_loss_save_dir, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(test_loss)

train_acc_save_dir = model_save_folder+model_save_title+"_train_acc"+".csv"
with open(train_acc_save_dir, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(train_acc)

test_acc_save_dir = model_save_folder+model_save_title+"_test_acc"+".csv"
with open(test_acc_save_dir, 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(test_acc)
     

if not args.run_test:
    print("Saving model to", model_save_dir, "...")
    torch.save(model.state_dict(), model_save_dir)   

if not args.run_test:
    plot_save_dir = "/datalake/homes/atefeh/q-learning_generalization_experiments_results/plots/"
    plot_title = "exact_q_prediction_fourroomssetconfigs_net_default_duplicating_xueyuan_experiments_run5_noL1regular_ACC"
    plot_save_path = plot_save_dir+plot_title
    plt.figure(figsize=(10,5))
    plt.title("Training and Test Acc")
    plt.plot(test_acc,label="test")
    plt.plot(train_acc,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Acc")
    plt.legend()
    #plt.show()
    plt.savefig(plot_save_path+".png")


 
