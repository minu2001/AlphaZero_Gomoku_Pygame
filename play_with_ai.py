# play_with_ai.py (콘솔 출력 자릿수 수정)

import sys
import threading
import queue
import numpy as np
import torch
import pygame

import agents
import model
import utils
from pygame_env import PygameEnv, BOARD_SIZE

# NEW: Numpy 배열 출력 시 소수점 2자리로 고정하고, 과학적 표기법 끄기
np.set_printoptions(precision=2, suppress=True)

# --- 설정 ---
MODEL_PATH = 'data/251013_11500_382997_step_model.pickle'
N_BLOCKS = 10
IN_PLANES = 5
OUT_PLANES = 128
N_MCTS = 400
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")


def load_ai_agent(model_path):
    ai_agent = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES, noise=False)
    ai_agent.model = model.PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, BOARD_SIZE).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        ai_agent.model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}");
        sys.exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}");
        sys.exit()
    ai_agent.model.eval()
    return ai_agent


def ai_worker(ai_agent, root_id, result_queue):
    print("AI가 수를 생각하고 있습니다...")
    _ = ai_agent.get_pi(root_id, tau=0)

    visit_counts = ai_agent.visit
    action_index = np.argmax(visit_counts)
    policy_map = visit_counts / (visit_counts.sum() + 1e-8)
    value = ai_agent.get_q_value(root_id)

    result_queue.put((action_index, policy_map, value))


def print_ai_info(policy, value, turn):
    player = "White" if turn == 1 else "Black"
    win_rate = (value + 1) / 2 * 100

    print("\n-----------------------------")
    print("Policy:")
    print(policy.reshape(BOARD_SIZE, BOARD_SIZE))
    print(f"\n{player}'s win%: {win_rate:.2f}%")
    print("-----------------------------")


def main():
    env = PygameEnv()
    ai_agent = load_ai_agent(MODEL_PATH)

    root_id = (0,)
    ai_thread = None
    result_queue = queue.Queue()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()

            if event.type == pygame.MOUSEBUTTONUP:
                mouse_pos = pygame.mouse.get_pos()
                if env.is_restart_button_clicked(mouse_pos):
                    env.reset();
                    root_id = (0,);
                    ai_thread = None
                    ai_agent.reset()
                    continue

                if env.win_index == 0 and env.turn == 0:
                    action = env.get_mouse_click(mouse_pos)
                    if action:
                        _, valid, _, _ = env.step(action)
                        if valid:
                            row, col = action
                            action_index = row * BOARD_SIZE + col
                            root_id += (action_index,)

        # --- AI 턴 처리 ---
        if env.win_index == 0 and env.turn == 1:
            if ai_thread is None or not ai_thread.is_alive():
                ai_thread = threading.Thread(target=ai_worker, args=(ai_agent, root_id, result_queue))
                ai_thread.start()

        # --- AI 계산 결과 확인 ---
        try:
            action_index, policy_map, value = result_queue.get_nowait()
            print("AI가 수를 결정했습니다.")

            print_ai_info(policy_map, value, env.turn)

            row = action_index // BOARD_SIZE
            col = action_index % BOARD_SIZE

            _, valid, _, _ = env.step((row, col))
            if valid:
                root_id += (action_index,)

            ai_thread = None
        except queue.Empty:
            pass

        env.render()


if __name__ == '__main__':
    main()