import numpy as np
import operator as op
from functools import reduce


def phases_gen(q_bits):
    angles = np.linspace(0, 2 * np.pi, 2 ** q_bits, endpoint=False)
    cb = np.exp(1j * angles)
    codebook = np.zeros((2 ** q_bits, 2))  # shape of the codebook
    for idx in range(cb.shape[0]):
        codebook[idx, 0] = np.real(cb[idx])
        codebook[idx, 1] = np.imag(cb[idx])
    return codebook


def bf_gain_cal(cb, H):
    bf_r = cb[:, ::2]
    bf_i = cb[:, 1::2]
    ch_r = H[:, :32]
    ch_i = H[:, 32:]
    bf_gain_1 = np.matmul(bf_r, np.transpose(ch_r))
    bf_gain_2 = np.matmul(bf_i, np.transpose(ch_i))
    bf_gain_3 = np.matmul(bf_r, np.transpose(ch_i))
    bf_gain_4 = np.matmul(bf_i, np.transpose(ch_r))
    bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
    bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
    bf_gain_pattern = bf_gain_r + bf_gain_i  # BF gain pattern matrix
    return bf_gain_pattern


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)


def corr_mining(bf_gain_matrix):
    # norm_factor = np.sqrt(np.mean(bf_gain_matrix, axis=0))
    # feature_mat = np.zeros(bf_gain_matrix.shape)
    # for u_id in range(bf_gain_matrix.shape[1]):
    #     feature_mat[:, u_id] = bf_gain_matrix[:, u_id] / norm_factor[u_id]
    num_raw_feature = bf_gain_matrix.shape[0]
    num_user = bf_gain_matrix.shape[1]
    num_feature = ncr(num_raw_feature, 2)
    norm_factor = np.mean(bf_gain_matrix, axis=0)
    feature_mat = np.zeros((num_feature, num_user))
    for u_id in range(num_user):
        feature_count = 0
        for idx_1 in range(num_raw_feature - 1):
            for idx_2 in range(idx_1 + 1, num_raw_feature):
                feature_mat[feature_count, u_id] = (bf_gain_matrix[idx_1, u_id] - bf_gain_matrix[idx_2, u_id]) / \
                                                   norm_factor[u_id]
                feature_count = feature_count + 1
        if feature_count != num_feature:
            print('error...')
    return feature_mat


def proj_pattern_cal(best_beam, sensing_beam):
    best_beam_r = best_beam[:, ::2]
    best_beam_i = best_beam[:, 1::2]
    sensing_beam_r = sensing_beam[:, ::2]
    sensing_beam_i = sensing_beam[:, 1::2]
    bf_gain_1 = np.matmul(best_beam_r, np.transpose(sensing_beam_r))
    bf_gain_2 = np.matmul(best_beam_i, np.transpose(sensing_beam_i))
    bf_gain_3 = np.matmul(best_beam_r, np.transpose(sensing_beam_i))
    bf_gain_4 = np.matmul(best_beam_i, np.transpose(sensing_beam_r))
    bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
    bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
    proj_pattern = bf_gain_r + bf_gain_i
    return proj_pattern


def real2ph(real_vec):
    num_ph = int(real_vec.shape[1] / 2)
    ph_vec = np.zeros((real_vec.shape[0], num_ph))
    for jj in range(real_vec.shape[0]):
        for ii in range(num_ph):
            if real_vec[jj, ii] <= 0 and real_vec[jj, ii + num_ph] <= 0:
                ph_vec[jj, ii] = np.arctan(np.divide(real_vec[jj, ii + num_ph], real_vec[jj, ii])) - np.pi
            elif real_vec[jj, ii] <= 0 and real_vec[jj, ii + num_ph] >= 0:
                ph_vec[jj, ii] = np.arctan(np.divide(real_vec[jj, ii + num_ph], real_vec[jj, ii])) + np.pi
            else:
                ph_vec[jj, ii] = np.arctan(np.divide(real_vec[jj, ii + num_ph], real_vec[jj, ii]))
    return ph_vec


def radius_mat(real_vec):
    num_element = int(real_vec.shape[1] / 2)
    radius_matrix = np.zeros((real_vec.shape[0], num_element))
    for jj in range(real_vec.shape[0]):
        for ii in range(num_element):
            radius_matrix[jj, ii] = np.sqrt(np.power(real_vec[jj, ii], 2) + np.power(real_vec[jj, ii+num_element], 2))
    return radius_matrix


def apply_phase(ch_orig, ph_pattern):
    ph_orig = real2ph(ch_orig)
    ph_rot = ph_orig - ph_pattern
    radius = radius_mat(ch_orig)
    ch_rot_complex = np.multiply(radius, np.exp(1j * ph_rot))
    ch_rot = np.concatenate((np.real(ch_rot_complex), np.imag(ch_rot_complex)), axis=1)
    return ch_rot


def sub_array(ch, sub_len, idx):
    num_ant = int(ch.shape[1] / 2)
    sub_ch = np.concatenate((ch[:, sub_len * (idx - 1):sub_len * idx],
                             ch[:, num_ant + sub_len * (idx - 1):num_ant + sub_len * idx]), axis=1)

    return sub_ch
