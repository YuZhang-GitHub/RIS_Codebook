import os
import torch
import numpy as np


class envCB:

    def __init__(self, ch, num_inp, num_bits, idx, options):

        self.idx = idx
        self.num_block_ph = num_inp
        self.num_bits = num_bits
        self.options = options
        self.cb_size = 2 ** self.num_bits
        self.codebook = self.codebook_gen()
        self.ch = torch.from_numpy(ch).float().cuda()
        self.state = torch.zeros((1, self.num_block_ph)).float().cuda()
        self.bf_vec = self.init_bf_vec()
        self.previous_gain = 0
        self.previous_gain_pred = 0
        self.th_step = 0.01
        self.threshold = torch.tensor([0]).float().cuda()
        self.count = 1
        self.record_freq = 10
        self.record_decay_th = 1000
        self.achievement = torch.tensor([0]).float().cuda()
        self.gain_record = [np.array(0)]
        self.N_count = 1
        self.best_bf_vec = self.init_best()
        self.best_bf_gain = 0
        self.opt_bf_gain()
        self.base_beams = torch.from_numpy(options['sub_phase_pattern']).float().cuda()
        self.ph_comb = torch.zeros_like(self.base_beams).float().cuda()
        self.ph_vec = torch.zeros((1, self.options['num_ant_all'])).float().cuda()

    def step(self, input_action):
        self.state = input_action
        self.ph_comb = self.base_beams + torch.t(input_action)
        self.bf_vec, self.ph_vec = self.phase2bf(self.ph_comb)
        reward, bf_gain = self.reward_fn()
        terminal = 0
        return input_action.clone(), reward, bf_gain, terminal

    def reward_fn(self):
        bf_gain = self.bf_gain_cal()
        if bf_gain > self.previous_gain:
            if bf_gain > self.threshold:
                reward = np.array([1]).reshape((1, 1))
                self.threshold = self.threshold_modif(bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = np.array([1]).reshape((1, 1))
        else:
            if bf_gain > self.threshold:
                reward = np.array([1]).reshape((1, 1))
                self.threshold = self.threshold_modif(bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = np.array([-1]).reshape((1, 1))
        self.previous_gain = self.previous_gain_pred
        return reward, bf_gain

    def get_reward(self, input_action):
        inner_state = input_action

        # Quantization Processing
        # self.options['ph_table_rep'].cuda()
        mat_dist = torch.abs(input_action.reshape(self.num_block_ph, 1) - self.options['ph_table_rep'])
        action_quant = self.options['ph_table_rep'][range(self.num_block_ph), torch.argmin(mat_dist, dim=1)].reshape(1, -1)
        inner_ph = self.base_beams + torch.t(action_quant)
        inner_bf, inner_ph_vec = self.phase2bf(inner_ph)
        bf_gain = self.bf_gain_cal_only(inner_bf)
        if bf_gain > self.previous_gain:
            if bf_gain > self.threshold:
                reward = np.array([1]).reshape((1, 1))
                self.threshold = self.threshold_modif_get_reward(inner_ph, bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = np.array([1]).reshape((1, 1))
        else:
            if bf_gain > self.threshold:
                reward = np.array([1]).reshape((1, 1))
                self.threshold = self.threshold_modif_get_reward(inner_ph, bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = np.array([-1]).reshape((1, 1))
        # if self.count % self.record_freq == 0:
        #     self.gain_vs_iter()
        #     if self.count == self.record_decay_th:
        #         self.record_freq = 1000
        self.previous_gain_pred = bf_gain + 0.1
        self.count += 1
        return reward, bf_gain, inner_state.clone(), inner_state.clone()

    def threshold_modif(self, bf_gain):
        self.achievement = bf_gain
        self.gain_recording(self.ph_vec)
        self.threshold = bf_gain
        return self.threshold

    def threshold_modif_get_reward(self, inner_bf, bf_gain):
        self.achievement = bf_gain
        self.gain_recording(inner_bf)
        self.threshold = bf_gain
        return self.threshold

    def opt_bf_gain(self):
        ch_r = torch.Tensor.cpu(self.ch.clone()).numpy()[:, :self.options['num_ant_all']]
        ch_i = torch.Tensor.cpu(self.ch.clone()).numpy()[:, self.options['num_ant_all']:]
        radius = np.sqrt(np.square(ch_r) + np.square(ch_i))
        gain_opt = np.mean(np.square(np.sum(radius, axis=1)))
        print('EGC bf gain: ', gain_opt)
        # return gain_opt

    def phase2bf(self, ph_comb):  # ph_comb: num_sub_array, sub_array_ant
        ph_vec = torch.reshape(ph_comb, (1, -1))
        bf_vec = torch.zeros((1, 2 * self.options['num_ant_all'])).float().cuda()
        for kk in range(self.options['num_ant_all']):
            bf_vec[0, 2 * kk] = torch.cos(ph_vec[0, kk])
            bf_vec[0, 2 * kk + 1] = torch.sin(ph_vec[0, kk])
        return bf_vec, ph_vec

    def bf_gain_cal(self): # used in self.reward_fn
        bf_r = self.bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = self.bf_vec[0, 1::2].clone().reshape(1, -1)
        ch_r = torch.squeeze(self.ch[:, :self.options['num_ant_all']].clone())
        ch_i = torch.squeeze(self.ch[:, self.options['num_ant_all']:].clone())
        bf_gain_1 = torch.matmul(bf_r, torch.t(ch_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(ch_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(ch_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(ch_r))

        bf_gain_r = (bf_gain_1+bf_gain_2)**2
        bf_gain_i = (bf_gain_3-bf_gain_4)**2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain = torch.mean(bf_gain_pattern)
        return bf_gain

    def bf_gain_cal_only(self, bf_vec): # used in self.get_reward
        bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)
        ch_r = torch.squeeze(self.ch[:, :self.options['num_ant_all']].clone())
        ch_i = torch.squeeze(self.ch[:, self.options['num_ant_all']:].clone())
        bf_gain_1 = torch.matmul(bf_r, torch.t(ch_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(ch_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(ch_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(ch_r))

        bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
        bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain = torch.mean(bf_gain_pattern)
        return bf_gain

    # def gain_vs_iter(self):
    #     gain_best = max(self.gain_record).reshape((1, 1))
    #     if os.path.exists('performance.txt'):
    #         with open('performance.txt', 'ab') as pf:
    #             np.savetxt(pf, np.array(self.count).reshape((1, 1)), fmt='%.2f', delimiter='\n')
    #         with open('performance.txt', 'ab') as pf:
    #             np.savetxt(pf, gain_best, fmt='%.2f', delimiter='\n')
    #     else:
    #         np.savetxt('performance.txt', np.array(self.count).reshape((1, 1)), fmt='%.2f', delimiter='\n')
    #         with open('performance.txt', 'ab') as pf:
    #             np.savetxt(pf, gain_best, fmt='%.2f', delimiter='\n')

    def gain_recording(self, bf_vec):
        new_gain = torch.Tensor.cpu(self.achievement).detach().numpy().reshape((1, 1))
        bf_print = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
        if new_gain > max(self.gain_record):
            self.gain_record.append(new_gain)
            self.best_bf_vec = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
            self.best_bf_gain = new_gain
            if os.path.exists('beams/beams_' + str(self.idx) + '_max.txt'):
                with open('beams/beams_' + str(self.idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams/beams_' + str(self.idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            else:
                np.savetxt('beams/beams_' + str(self.idx) + '_max.txt', new_gain, fmt='%.2f', delimiter='\n')
                # with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                #     np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams/beams_' + str(self.idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')

    def codebook_gen(self):
        angles = np.linspace(0, 2 * np.pi, self.cb_size, endpoint=False)
        cb = np.exp(1j * angles)
        codebook = torch.zeros((self.cb_size, 2)) # shape of the codebook
        for ii in range(cb.shape[0]):
            codebook[ii, 0] = torch.tensor(np.real(cb[ii]))
            codebook[ii, 1] = torch.tensor(np.imag(cb[ii]))
        return codebook

    def init_bf_vec(self):
        bf_vec = torch.empty((1, 2 * self.num_block_ph))
        bf_vec[0, ::2] = torch.tensor([1])
        bf_vec[0, 1::2] = torch.tensor([0])
        bf_vec = bf_vec.float().cuda()
        return bf_vec

    def init_best(self):
        ph_book = np.linspace(-np.pi, np.pi, 2 ** self.num_bits, endpoint=False)
        ph_vec = np.array([[ph_book[np.random.randint(0, len(ph_book))] for ii in range(self.num_block_ph)]])
        bf_complex = np.exp(1j * ph_vec)
        bf_vec = np.empty((1, 2 * self.num_block_ph))
        for kk in range(self.num_block_ph):
            bf_vec[0, 2*kk] = np.real(bf_complex[0, kk])
            bf_vec[0, 2*kk+1] = np.imag(bf_complex[0, kk])
        return bf_vec
