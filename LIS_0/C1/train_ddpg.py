import os
import numpy as np
import torch
from env_ddpg import envCB


def train(ch, base_beams, options, beam_id):
    with torch.cuda.device(options['gpu_idx']):

        options['ph_table_rep'] = options['ph_table_rep'].cuda()
        options['multi_step'] = options['multi_step'].cuda()
        options['ph_table'] = options['ph_table'].cuda()

        options['sub_phase_pattern'] = base_beams

        CB_Env = envCB(ch, options['num_block_ph'], options['num_bits'], beam_id, options)

        num_ph = 2 ** options['num_bits']
        angles = np.linspace(0, 2 * np.pi, num_ph, endpoint=False)

        # -------------- Exhaustive Search -------------- #
        iteration = 0
        for loop_idx_1 in range(num_ph):
            for loop_idx_2 in range(num_ph):
                action = torch.from_numpy(np.array([[angles[loop_idx_1], angles[loop_idx_2]]])).float().cuda()
                CB_Env.get_reward(action)
                iteration = iteration + 1

                iter_id = np.array(iteration).reshape(1, 1)
                best_state = CB_Env.best_bf_vec.reshape(1, -1)
                best_gain = CB_Env.best_bf_gain
                if os.path.exists('pfs/pf_' + str(beam_id) + '.txt'):
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, iter_id, fmt='%d', delimiter='\n')
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, best_gain, fmt='%.3f', delimiter='\n')
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, best_state, fmt='%.5f', delimiter=',')
                else:
                    np.savetxt('pfs/pf_' + str(beam_id) + '.txt', iter_id, fmt='%d', delimiter='\n')
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, best_gain, fmt='%.3f', delimiter='\n')
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, best_state, fmt='%.5f', delimiter=',')

                print('Beam: %d, iter: %d.' % (beam_id, iteration))

    return 0
