import logging
import pickle
import random
from collections import deque
from datetime import datetime
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import model
import utils
import agents

# env_small: 9x9, env_regular: 15x15
# 11x11 전용 환경 사용
from env import env_regular as game

logging.basicConfig(
    filename='logs/log_{}.txt'.format(datetime.now().strftime('%y%m%d')),
    level=logging.WARNING)

# Game
# env_regular.py에 맞게 BOARD_SIZE를 11로 설정했습니다.
BOARD_SIZE = 11
N_MCTS = 400
TAU_THRES = 6
SEED = 0
PRINT_SELFPLAY = True

# Net
N_BLOCKS = 10
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 128

# Training
USE_TENSORBOARD = False
N_SELFPLAY = 100
TOTAL_ITER = 10000000
MEMORY_SIZE = 30000
N_EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-4
L2 = 0

# Hyperparameter sharing
agents.PRINT_MCTS = PRINT_SELFPLAY

# Set gpu or cpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('cuda:', use_cuda)

# Numpy printing style
np.set_printoptions(suppress=True)

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)

# Global variables
rep_memory = deque(maxlen=MEMORY_SIZE)
cur_memory = deque()
step = 0
start_iter = 0
total_epoch = 0
result = {'Black': 0, 'White': 0, 'Draw': 0}
if USE_TENSORBOARD:
    from tensorboardX import SummaryWriter

    Writer = SummaryWriter()

# Initialize agent & model
Agent = agents.ZeroAgent(BOARD_SIZE,
                         N_MCTS,
                         IN_PLANES,
                         noise=True)
Agent.model = model.PVNet(N_BLOCKS,
                          IN_PLANES,
                          OUT_PLANES,
                          BOARD_SIZE).to(device)
optimizer = optim.Adam(Agent.model.parameters(), lr=LR, weight_decay=L2, eps=1e-6)

logging.warning(
    '\nCUDA: {}'
    '\nAGENT: {}'
    '\nMODEL: {}'
    '\nSEED: {}'
    '\nBOARD_SIZE: {}'
    '\nN_MCTS: {}'
    '\nTAU_THRES: {}'
    '\nN_BLOCKS: {}'
    '\nIN_PLANES: {}'
    '\nOUT_PLANES: {}'
    '\nN_SELFPLAY: {}'
    '\nMEMORY_SIZE: {}'
    '\nN_EPOCHS: {}'
    '\nBATCH_SIZE: {}'
    '\nLR: {}'
    '\nL2: {}'.format(
        use_cuda,
        type(Agent).__name__,
        type(Agent.model).__name__,
        SEED,
        BOARD_SIZE,
        N_MCTS,
        TAU_THRES,
        N_BLOCKS,
        IN_PLANES,
        OUT_PLANES,
        N_SELFPLAY,
        MEMORY_SIZE,
        N_EPOCHS,
        BATCH_SIZE,
        LR,
        L2))


def self_play(n_selfplay):
    global cur_memory, rep_memory
    global Agent

    Agent.model.eval()
    state_black = deque()
    state_white = deque()
    pi_black = deque()
    pi_white = deque()

    for episode in range(n_selfplay):
        # 에피소드 번호 출력
        print(f"\n[ Self-Play Episode {episode + 1}/{n_selfplay} ]")

        if (episode + 1) % 10 == 0:
            logging.warning('Playing Episode {:3}'.format(episode + 1))

        env = game.GameState('text')
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), 'float')
        turn = 0
        root_id = (0,)
        win_index = 0
        time_steps = 0
        action_index = None

        while win_index == 0:
            if PRINT_SELFPLAY:
                utils.render_str(board, BOARD_SIZE, action_index)

            tau = 1 if time_steps < TAU_THRES else 0
            pi = Agent.get_pi(root_id, tau)

            state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)
            if turn == 0:
                state_black.appendleft(state)
                pi_black.appendleft(pi)
            else:
                state_white.appendleft(state)
                pi_white.appendleft(pi)

            action, action_index = utils.get_action(pi)
            root_id += (action_index,)

            row = action_index // BOARD_SIZE
            col = action_index % BOARD_SIZE

            if PRINT_SELFPLAY:
                with torch.no_grad():
                    state_input = torch.tensor([state]).to(device).float()
                    p, v = Agent.model(state_input)
                    p = p.cpu().numpy()[0]
                    v = v.item()

                    print('\nPi:\n{}'.format(
                        pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2)
                    ))
                    print('\nPolicy:\n{}'.format(
                        p.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2)
                    ))

                if turn == 0:
                    print("\nBlack's win%: {:.2f}%".format((v + 1) / 2 * 100))
                else:
                    print("\nWhite's win%: {:.2f}%".format((v + 1) / 2 * 100))

            board, _, win_index, turn, _ = env.step((row, col))
            time_steps += 1

            if win_index != 0:
                if win_index == 1:
                    reward_black = 1.
                    reward_white = -1.
                    result['Black'] += 1
                elif win_index == 2:
                    reward_black = -1.
                    reward_white = 1.
                    result['White'] += 1
                else:
                    reward_black = 0.
                    reward_white = 0.
                    result['Draw'] += 1

                while state_black or state_white:
                    if state_black:
                        cur_memory.append((state_black.pop(),
                                           pi_black.pop(),
                                           reward_black))
                    if state_white:
                        cur_memory.append((state_white.pop(),
                                           pi_white.pop(),
                                           reward_white))

                if PRINT_SELFPLAY:
                    utils.render_str(board, BOARD_SIZE, action_index)
                    bw, ww, dr = result['Black'], result['White'], result['Draw']
                    win_rate = (bw + 0.5 * dr) / (bw + ww + dr) * 100 if (bw + ww + dr) > 0 else 0
                    print('')
                    print('=' * 20,
                          " {:3} Game End   ".format(episode + 1),
                          '=' * 20)
                    print('Black Win: {:3}   '
                          'White Win: {:3}   '
                          'Draw: {:2}   '
                          'Win%: {:.2f}%'.format(
                        bw, ww, dr, win_rate))
                    print('current memory size:', len(cur_memory))

                Agent.reset()

    rep_memory.extend(utils.augment_dataset(cur_memory, BOARD_SIZE))


def train(n_epochs, n_iter):
    global step, total_epoch
    global Agent, optimizer, Writer
    global rep_memory, cur_memory

    Agent.model.train()
    loss_all, loss_v, loss_p = [], [], []

    # 학습할 데이터가 부족하면 건너뜁니다.
    if len(rep_memory) < BATCH_SIZE:
        print("Not enough data to train. Skipping.")
        return

    # 이 부분이 나중에 데이터가 쌓이면 느려지는 원인이 됩니다.
    train_memory = []
    sample_size = BATCH_SIZE * len(cur_memory)
    if len(rep_memory) < sample_size:
        sample_size = len(rep_memory)

    train_memory.extend(random.sample(list(rep_memory), sample_size))

    dataloader = DataLoader(train_memory,
                            batch_size=BATCH_SIZE,
                            shuffle=False,  # True로 바꾸는 것을 권장합니다.
                            pin_memory=use_cuda)

    print('=' * 58)
    print(' ' * 20 + ' Start Learning ' + ' ' * 20)
    print('=' * 58)
    print('current memory size:', len(cur_memory))
    print('replay memory size:', len(rep_memory))
    print('train memory size:', len(train_memory))
    print('optimizer: {}'.format(optimizer))

    for epoch in range(n_epochs):
        for i, (s, pi, z) in enumerate(dataloader):
            s_batch = s.to(device).float()
            pi_batch = pi.to(device).float()
            z_batch = z.to(device).float()

            p_batch, v_batch = Agent.model(s_batch)

            v_loss = (v_batch - z_batch).pow(2).mean()
            p_loss = -(pi_batch * p_batch.log()).sum(dim=-1).mean()
            loss = v_loss + p_loss

            if PRINT_SELFPLAY:
                loss_v.append(v_loss.item())
                loss_p.append(p_loss.item())
                loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if USE_TENSORBOARD:
                Writer.add_scalar('Loss', loss.item(), step)
                Writer.add_scalar('Loss V', v_loss.item(), step)
                Writer.add_scalar('Loss P', p_loss.item(), step)

            if PRINT_SELFPLAY:
                print('{:4} Step Loss: {:.4f}   Loss V: {:.4f}   Loss P: {:.4f}'
                      .format(step, loss.item(), v_loss.item(), p_loss.item()))
        total_epoch += 1

        if PRINT_SELFPLAY:
            print('-' * 58)
            print('{:2} Epoch Loss: {:.4f}   Loss V: {:.4f}   Loss P: {:.4f}'
                  .format(total_epoch,
                          np.mean(loss_all),
                          np.mean(loss_v),
                          np.mean(loss_p)))


def save_model(agent, n_iter, step):
    datetime_now = datetime.now().strftime('%y%m%d')
    if not os.path.exists('data'):
        os.makedirs('data')
    torch.save(
        agent.model.state_dict(),
        'data/{}_{}_{}_step_model.pickle'.format(datetime_now, n_iter, step))


def save_dataset(memory, n_iter, step):
    datetime_now = datetime.now().strftime('%y%m%d')
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/{}_{}_{}_step_dataset.pickle'.format(
            datetime_now, n_iter, step), 'wb') as f:
        pickle.dump(list(memory), f, pickle.HIGHEST_PROTOCOL)


def load_data(model_path, dataset_path):
    global rep_memory, step, start_iter
    if model_path:
        print('load model: {}'.format(model_path))
        state = Agent.model.state_dict()
        state.update(torch.load(model_path))
        Agent.model.load_state_dict(state)
        step = int(model_path.split('_')[2])
        start_iter = int(model_path.split('_')[1]) + 1
    if dataset_path:
        print('load dataset: {}'.format(dataset_path))
        with open(dataset_path, 'rb') as f:
            rep_memory = deque(pickle.load(f), maxlen=MEMORY_SIZE)


def reset_iter(result, cur_memory):
    global total_epoch
    result['Black'] = 0
    result['White'] = 0
    result['Draw'] = 0
    total_epoch = 0
    cur_memory.clear()


if __name__ == '__main__':
    # 처음부터 시작하기 위해 model_path와 dataset_path를 None으로 설정합니다.
    model_path = None
    dataset_path = None
    load_data(model_path, dataset_path)

    for n_iter in range(start_iter, TOTAL_ITER):
        print('=' * 58)
        print(' ' * 20 + '  {:2} Iteration  '.format(n_iter) + ' ' * 20)
        print('=' * 58)

        datetime_now = datetime.now().strftime('%y%m%d')
        if n_iter > 0:
            # 이 부분 때문에 두 번째 이터레이션부터는 셀프플레이를 1번만 합니다.
            N_SELFPLAY_ITER = 1
            self_play(N_SELFPLAY_ITER)
            train(N_EPOCHS, n_iter)
        else:
            # 첫 이터레이션에서는 100번 셀프플레이를 합니다.
            self_play(N_SELFPLAY)

        # 100 이터레이션마다 모델과 데이터셋을 저장합니다.
        if n_iter % 100 == 0:
            save_model(Agent, n_iter + 100, step)
            save_dataset(rep_memory, n_iter + 100, step)

        reset_iter(result, cur_memory)