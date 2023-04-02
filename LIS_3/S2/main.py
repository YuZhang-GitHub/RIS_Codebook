import os
import torch
import numpy as np
import copy
from train_ddpg import train
from DataPrep import dataPrep
from function_lib import sub_array
import scipy.io as scio
import torch.multiprocessing as mp

if __name__ == '__main__':

    options = {
        'gpu_idx': 3,
        'num_ant': 32,
        'num_bits': 4,
        'num_NNs': 8,  # number of beams
        'sub_array_id': 2,
        'ch_sample_ratio': 1,
        'target_update': 3,
        'pf_print': 10,
        'clustering_mode': 'random',
        'save_freq': 10000,
        'PATH_1': '../S1/pretrained_model/critic_beam',
        'PATH_2': '../S1/pretrained_model/actor_beam',
        'load_iter': '60000'
    }

    LIS_id = os.path.abspath('..')[-1]
    options['path'] = './LIS_' + LIS_id + '_grid1201_1-1400_80.mat'

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 200000,
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 8192,
        'minibatch_size': 1024,
        'gamma': 0
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    if not os.path.exists('pfs/'):
        os.mkdir('pfs/')

    ch_full = dataPrep(options['path'])  # shape: (users, 2*ant)
    ch = sub_array(ch_full, options['num_ant'], options['sub_array_id'])

    user_group = []
    ch_group = []
    label_load = scio.loadmat('./Distributed_LIS_label_1201_1-1400_80-8.mat')['labels']
    for ii in range(label_load.shape[1]):
        label_ = label_load[0, ii].tolist()[0]
        user_group.append(label_)
        ch_group.append(ch[user_group[ii], :])

    with torch.cuda.device(options['gpu_idx']):

        # Quantization settings
        options['num_ph'] = 2 ** options['num_bits']
        options['multi_step'] = torch.from_numpy(
            np.linspace(int(-(options['num_ph'] - 2) / 2),
                        int(options['num_ph'] / 2),
                        num=options['num_ph'],
                        endpoint=True)).type(dtype=torch.float32).reshape(1, -1)
        options['pi'] = torch.tensor(np.pi)
        options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']  # (1, num_phase)
        options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)  # (num_ant, num_phase)

        train_opt_list = []

        for beam_id in range(options['num_NNs']):
            train_opt_list.append(copy.deepcopy(train_opt))

        # ---------- Learning ---------- #
        manager = mp.Manager()
        processes = []
        for beam_id in range(options['num_NNs']):
            p = mp.Process(target=train, args=(ch_group[beam_id],
                                               options,
                                               train_opt_list[beam_id],
                                               beam_id))
            p.start()
            processes.append(p)
        for proc in processes:
            proc.join()

    pp = 1
