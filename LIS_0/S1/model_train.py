import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from environment import envCB
# from dqnAgent import dqnAgent, init_weights


def train(model, model_t, env, options, train_options, beam_id):

    CB_Env = env

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_of_iter = train_options['num_iter']
    if train_options['overall_iter'] == 0:
        state = torch.empty(1, 2 * options['num_ant'])
        state[0, ::2] = torch.tensor([1])
        state[0, 1::2] = torch.tensor([0])
        state = state.cuda()
    else:
        state = train_options['state']

    # -------------- training -------------- #
    replay_memory = train_options['replay_memory']
    iteration = 0
    while iteration < num_of_iter:

        # predicted action
        action_pred = model(state)  # action_pred is a torch.Tensor with torch.Size([64]), representing Q values
        random_action = random.random() <= train_options['epsilon']
        if random_action:
            # print('Select best action from random actions.')
            action_set = torch.zeros(model.action_dim, dtype=torch.float32)
            action_set[random.randint(0, model.action_dim - 1)] = 1
        else:
            action_set = action_pred
        # up to this point, we get action_set, either random one or predicted one

        state_1, reward, terminal = CB_Env.step(action_set)  # get next state and reward
        reward = torch.from_numpy(reward).float().cuda()
        action = action_set.reshape((1, -1)).float().cuda()  # best action accordingly

        # s, reward_pred, t = CB_Env.step(action_pred)
        # reward_pred = torch.from_numpy(reward_pred).float().cuda()

        # save transition to replay memory and if replay memory is full, remove the oldest transition
        replay_memory.append((state, action, reward, state_1, terminal))
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # -------------- experience replay -------------- #
        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch, since torch.cat is by default dim=0
        state_batch = torch.cat(tuple(d[0] for d in minibatch))  # torch.Size([*, 64])
        action_batch = torch.cat(tuple(d[1] for d in minibatch))  # torch.Size([*, 64])
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))  # torch.Size([*, 1])
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))  # torch.Size([*, 64])

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # -------------- Core of Learning (Brain) -------------- #
        # get action_pred for the next state: batch training
        # important: state_1_batch is the state_{t+1}
        # output_1_batch = model(state_1_batch)  # torch.Size([*, 64])
        # action_1_batch = torch.Tensor.cpu(torch.argmax(output_1_batch, dim=1)).numpy().tolist()

        output_1_batch_t = model_t(state_1_batch)  # action_{t+1} from target network
        action_1_batch_t = torch.Tensor.cpu(torch.argmax(output_1_batch_t, dim=1)).numpy().tolist()
        index_1 = range(0, len(minibatch))
        est_values_t = output_1_batch_t[index_1, action_1_batch_t].reshape((len(minibatch), 1))

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        # y_batch stands for the "true" q values
        max_pos = torch.argmax(action_batch, dim=1).reshape((action_batch.shape[0], 1))
        # y_batch = torch.zeros((len(minibatch), model.action_dim)).float().cuda()
        y_batch = output_1_batch_t
        for ii in range(len(minibatch)):
            if minibatch[ii][4]:
                y_batch[ii, max_pos[ii]] = reward_batch[ii, :]
            else:
                y_batch[ii, max_pos[ii]] = reward_batch[ii, :] + model.gamma * est_values_t[ii, :]

        # for idx in range(action_batch.shape[0]):
        #     action_idx_batch[idx, max_pos[idx]] = 1
        q_value = model(state_batch)

        # -------------- optimize the network -------------- #
        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()
        y_batch = y_batch.detach()  # returns a new Tensor, detached from the current graph, no gradient pathway
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        # -------------- naive loop related processing -------------- #
        # UPDATE state, epsilon, target network regularly by previous settings
        state = state_1
        iteration += 1
        train_options['overall_iter'] += 1

        if iteration % options['target_update'] == 0:
            model_t.load_state_dict(model.state_dict())

        if train_options['overall_iter'] % model.iteration_epsilon == 0:
            train_options['epsilon'] = train_options['epsilon'] * model.discount_epsilon
            if train_options['epsilon'] <= model.final_epsilon:
                train_options['epsilon'] = model.final_epsilon

        if train_options['overall_iter'] % options['pf_print'] == 0:
            iter_id = np.array(train_options['overall_iter']).reshape(1, 1)
            best_state = CB_Env.best_state.reshape(1, -1)
            if os.path.exists('pfs/pf_' + str(beam_id) + '.txt'):
                with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                    np.savetxt(bm, iter_id, fmt='%d', delimiter='\n')
                with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                    np.savetxt(bm, best_state, fmt='%.2f', delimiter=',')
            else:
                np.savetxt('pfs/pf_' + str(beam_id) + '.txt', iter_id, fmt='%d', delimiter='\n')
                with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                    np.savetxt(bm, best_state, fmt='%.2f', delimiter=',')

        # print("Beam ID:", beam_id,
        #       "Iteration:", train_options['overall_iter'],
        #       # "Q max:", np.max(action_pred.cpu().detach().numpy()),
        #       # "Replay memory size:", len(replay_memory),
        #       # "Action predicted by agent:", torch.Tensor.cpu(torch.argmax(action_pred)).numpy(),
        #       # "Reward by agent:", np.int(reward_pred),
        #       )

    train_options['state'] = state
    train_options['best_state'] = CB_Env.best_state
    return train_options
