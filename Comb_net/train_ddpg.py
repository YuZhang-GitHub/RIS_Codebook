import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DDPG_classes import Actor, Critic, OUNoise, init_weights
from env_ddpg import envCB


def train(ch, base_beams, options, train_options, beam_id):

    with torch.cuda.device(options['gpu_idx']):

        options['sub_phase_pattern'] = base_beams

        options['ph_table_rep'] = options['ph_table_rep'].cuda()
        options['multi_step'] = options['multi_step'].cuda()
        options['ph_table'] = options['ph_table'].cuda()

        actor_net = Actor(options['num_block_ph'], options['num_block_ph'])
        critic_net = Critic(2 * options['num_block_ph'], 1)
        actor_net_t = Actor(options['num_block_ph'], options['num_block_ph'])
        critic_net_t = Critic(2 * options['num_block_ph'], 1)
        ounoise = OUNoise((1, options['num_block_ph']))
        CB_Env = envCB(ch, options['num_block_ph'], options['num_bits'], beam_id, options)

        actor_net.apply(init_weights)
        actor_net_t.load_state_dict(actor_net.state_dict())
        critic_net.apply(init_weights)
        critic_net_t.load_state_dict(critic_net.state_dict())
        actor_net.cuda()
        critic_net.cuda()
        actor_net_t.cuda()
        critic_net_t.cuda()

        critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-3, weight_decay=1e-3)
        actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3, weight_decay=1e-2)
        critic_criterion = nn.MSELoss()

        if train_options['overall_iter'] == 1:
            state = torch.zeros((1, options['num_block_ph'])).float().cuda()  # vector of phases
            print('Initial State Activated.')
        else:
            state = train_options['state']

        # -------------- training -------------- #
        replay_memory = train_options['replay_memory']
        iteration = 0
        num_of_iter = train_options['num_iter']
        while iteration < num_of_iter:

            # Proto-action
            action_pred = actor_net(state)
            reward_pred, bf_gain_pred, action_quant_pred, state_1_pred = CB_Env.get_reward(action_pred)
            reward_pred = torch.from_numpy(reward_pred).float().cuda()

            # Exploration and Quantization Processing
            action_pred_noisy = ounoise.get_action(action_pred,
                                                   t=train_options['overall_iter'])  # torch.Size([1, action_dim])
            mat_dist = torch.abs(action_pred_noisy.reshape(options['num_block_ph'], 1) - options['ph_table_rep'])
            action_quant = options['ph_table_rep'][range(options['num_block_ph']), torch.argmin(mat_dist, dim=1)].reshape(1,
                                                                                                                     -1)

            # action_quant = action_pred

            state_1, reward, bf_gain, terminal = CB_Env.step(action_quant)  # get next state and reward
            reward = torch.from_numpy(reward).float().cuda()
            action = action_quant.reshape((1, -1)).float().cuda()  # best action accordingly

            # save transition to replay memory and if replay memory is full, remove the oldest transition
            replay_memory.append((state, action, reward, state_1, terminal))
            replay_memory.append((state, action_quant_pred, reward_pred, state_1_pred, terminal))
            if len(replay_memory) > train_options['replay_memory_size']:
                replay_memory.pop(0)  # clear the oldest memory
                replay_memory.pop(0)

            # -------------- Experience Replay -------------- #
            # sample random minibatch
            minibatch = random.sample(replay_memory, min(len(replay_memory), train_options['minibatch_size']))

            # unpack minibatch, since torch.cat is by default dim=0, which is the dimension of batch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))  # torch.Size([*, state_dim])
            action_batch = torch.cat(tuple(d[1] for d in minibatch))  # torch.Size([*, action_dim])
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))  # torch.Size([*, 1])
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))  # torch.Size([*, state_dim])

            state_batch = state_batch.detach()
            action_batch = action_batch.detach()
            reward_batch = reward_batch.detach()
            state_1_batch = state_1_batch.detach()

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # -------------- Core of Learning (Brain) -------------- #
            # loss calculation for Critic Network
            next_actions = actor_net_t(state_1_batch)
            next_Q = critic_net_t(state_1_batch, next_actions)
            Q_prime = reward_batch + train_options['gamma'] * next_Q
            Q_pred = critic_net(state_batch, action_batch)
            critic_loss = critic_criterion(Q_pred, Q_prime.detach())

            # Update Critic Network
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # loss calculation for Actor Network
            actor_loss = torch.mean(-critic_net(state_batch, actor_net(state_batch)))

            # Update Actor Network
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # -------------- Naive Loop Related Data Processing -------------- #
            # UPDATE state, epsilon, target network, etc.
            state = state_1_pred
            iteration += 1
            train_options['overall_iter'] += 1  # global counter

            # update: target network
            if train_options['overall_iter'] % options['target_update'] == 0:
                actor_params = actor_net.state_dict()
                critic_params = critic_net.state_dict()
                actor_t_params = actor_net_t.state_dict()
                critic_t_params = critic_net_t.state_dict()

                for name in critic_params:
                    critic_params[name] = train_options['tau'] * critic_params[name].clone() + \
                                          (1 - train_options['tau']) * critic_t_params[name].clone()

                critic_net_t.load_state_dict(critic_params)

                for name in actor_params:
                    actor_params[name] = train_options['tau'] * actor_params[name].clone() + \
                                         (1 - train_options['tau']) * actor_t_params[name].clone()

                actor_net_t.load_state_dict(actor_params)

            # store: best beamforming vector so far
            if train_options['overall_iter'] % options['pf_print'] == 0:
                iter_id = np.array(train_options['overall_iter']).reshape(1, 1)
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

            print(
                "Beam: %d, Iteration: %d, Q value: %.4f, Reward: %d, BF Gain pred: %.2f, BF Gain: %.2f, Critic Loss: %.2f, Policy Loss: %.2f" % \
                (beam_id, train_options['overall_iter'],
                 np.max(torch.Tensor.cpu(Q_pred.detach()).numpy().squeeze()),
                 int(torch.Tensor.cpu(reward).numpy().squeeze()),
                 torch.Tensor.cpu(bf_gain_pred.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(bf_gain.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(critic_loss.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(actor_loss.detach()).numpy().squeeze()))

        # Training Communication Interface
        train_options['replay_memory'] = replay_memory  # used for the next loop
        train_options['state'] = state  # used for the next loop
        train_options['best_state'] = CB_Env.best_bf_vec  # used for clustering and assignment

    return train_options
