import numpy as np


class GameState:
    def __init__(self, render_mode='text'):
        self.size = 11  # 11x11 보드
        self.render_mode = render_mode
        self.reset()

    def reset(self):
        self.game_board = np.zeros((self.size, self.size), dtype=int)
        self.curr_turn = 0  # 0 = 흑, 1 = 백
        self.win_index = 0
        self.message = ""
        self.action_index = None
        return self.game_board, self.curr_turn

    def step(self, action):
        """
        action: (row, col) 좌표
        return: (board, check_valid_pos, win_index, next_turn, action_index)
        """
        row, col = action
        check_valid_pos = False
        action_index = None

        # 범위 밖
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            self.message = "잘못된 위치입니다."
            return self.game_board, check_valid_pos, self.win_index, self.curr_turn, action_index

        # 이미 돌이 있는 칸
        if self.game_board[row][col] != 0:
            self.message = "이미 놓인 자리입니다."
            return self.game_board, check_valid_pos, self.win_index, self.curr_turn, action_index

        # 돌 두기
        stone = 1 if self.curr_turn == 0 else -1
        self.game_board[row][col] = stone
        check_valid_pos = True
        action_index = row * self.size + col

        # 승리 판정
        self.win_index = self._check_win(row, col, stone)

        # 턴 교체
        if self.win_index == 0:
            self.curr_turn = 1 - self.curr_turn

        return self.game_board, check_valid_pos, self.win_index, self.curr_turn, action_index

    def _check_win(self, row, col, stone):
        """ 오목 규칙: 연속 5개 확인 """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 가로, 세로, 대각선
        for dr, dc in directions:
            count = 1
            # 앞으로
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.game_board[r][c] == stone:
                count += 1
                r += dr
                c += dc
            # 뒤로
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.game_board[r][c] == stone:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return 1 if stone == 1 else 2

        # 무승부 (보드가 가득 참)
        if np.all(self.game_board != 0):
            return 3

        return 0


def Return_BoardParams():
    """
    (board_size, win_mark)
    """
    return (11, 5)
