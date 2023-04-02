import os
import torch
import numpy as np
import torch.multiprocessing as mp
from train_ddpg import train
from DataPrep import dataPrep
import scipy.io as scio


if __name__ == '__main__':

    options = {
        'gpu_idx': 3,
        'num_ant_all': 256,
        'num_block_ph': 4,  # equal to number of sub-arrays
        'num_NNs': 8,  # number of beams
        'num_bits': 4,
        'path': 'D:/Yu/large_array_RL/paper_simu_new_non/dataset/Distributed_LIS_ULA256.mat'
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    if not os.path.exists('pfs/'):
        os.mkdir('pfs/')

    ch = dataPrep(options['path'])

    num_ant = 64
    num_sub_array = options['num_block_ph']
    num_beam = options['num_NNs']
    phase_pattern = np.zeros((num_sub_array, num_ant, num_beam))

    for ii in range(num_sub_array):
        for kk in range(num_beam):
            fname = '../LIS_' + str(ii) + '/C1/beams/beams_' + str(kk) + '_max.txt'
            with open(fname, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1]
                phase_pattern[ii, :, kk] = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)

    options['phase_pattern'] = phase_pattern
    label_load = scio.loadmat('D:/Yu/large_array_RL/paper_simu_new_non/user_clustering/Distributed_LIS_label_1201_1-1400_80-' + str(options['num_NNs']) + '.mat')['labels']
    ch_group = []
    beams_group = []
    for beam_idx in range(options['num_NNs']):
        label_ = label_load[0, beam_idx].tolist()[0]
        ch_group.append(ch[label_, :])
        beams_group.append(options['phase_pattern'][:, :, beam_idx])

    # Quantization settings
    options['num_ph'] = 2 ** options['num_bits']
    options['multi_step'] = torch.from_numpy(
        np.linspace(int(-(options['num_ph'] - 2) / 2),
                    int(options['num_ph'] / 2),
                    num=options['num_ph'],
                    endpoint=True)).type(dtype=torch.float32).reshape(1, -1)
    options['pi'] = torch.tensor(np.pi)
    options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
    options['ph_table_rep'] = options['ph_table'].repeat(options['num_block_ph'], 1)

    # ---------- Learning ---------- #
    manager = mp.Manager()
    processes = []
    for beam_id in range(options['num_NNs']):
        p = mp.Process(target=train, args=(ch_group[beam_id],
                                           beams_group[beam_id],
                                           options,
                                           beam_id))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()

    pp = 1
