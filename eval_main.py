import sys
import math
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import pygame
import pygame.gfxdraw

# ==============================
# 설정
# ==============================
BOARD_SIZE = 11          # 11x11
CELL = 48                # 칸 크기 (px)
MARGIN = 60              # 바둑판 외곽 여백
LINE_W = 2               # 격자선 두께
BORDER_W = 4             # 외곽 테두리 두께
STAR_R = 5               # 성화점(별점) 반지름
FPS = 60

# 색상 팔레트
WOOD_LIGHT = (219, 184, 136)
WOOD_DARK = (196, 160, 118)
GRID = (70, 50, 20)
BORDER = (60, 45, 18)
BLACK = (15, 15, 15)
WHITE = (245, 245, 245)
HUD = (30, 30, 30)
GOLD = (255, 210, 60)
RED = (220, 60, 60)
WIN_HALO = (255, 120, 120, 60)

# 키 안내
HELP_TEXT = "[Click] 착수  |  [A] 사람 vs AI 토글  |  [R] 리셋  |  [H] 좌표표시  |  [ESC] 종료"

# ==============================
# 유틸: 효과음 생성 (별도 파일 불필요)
# ==============================
def make_click_sound(sample_rate=22050, ms=55, freq=220):
    """간단한 파형으로 '탁' 소리 생성."""
    try:
        import pygame.sndarray
        t = np.linspace(0, ms / 1000.0, int(sample_rate * (ms / 1000.0)), endpoint=False)
        # 지수 감쇠가 있는 짧은 사인파 + 클립 노이즈
        wave = (0.7 * np.sin(2 * np.pi * freq * t) * np.exp(-8 * t)).astype(np.float32)
        # 2채널 스테레오
        stereo = np.repeat(wave[:, None], 2, axis=1)
        sound = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
        sound.set_volume(0.25)
        return sound
    except Exception:
        return None

# ==============================
# 보드/게임 로직
# ==============================
class Omok:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.reset()

    def reset(self):
        self.board = [[0] * self.size for _ in range(self.size)]  # 0: empty, 1: 흑, -1: 백
        self.turn = 1  # 흑 선공
        self.last_move: Optional[Tuple[int, int]] = None
        self.winner = 0
        self.win_seq: List[Tuple[int, int]] = []

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def place(self, x, y) -> bool:
        if self.winner != 0:
            return False
        if not self.in_bounds(x, y):
            return False
        if self.board[y][x] != 0:
            return False
        self.board[y][x] = self.turn
        self.last_move = (x, y)
        if self.check_win_from(x, y):
            self.winner = self.turn
        else:
            self.turn *= -1
        return True

    def check_win_from(self, x, y) -> bool:
        """방향 4개(ㅡ, |, \, /)로 5목 확인"""
        who = self.board[y][x]
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in dirs:
            seq = [(x, y)]
            # 양방향 확장
            for sgn in (1, -1):
                cx, cy = x, y
                while True:
                    cx += dx * sgn
                    cy += dy * sgn
                    if not self.in_bounds(cx, cy) or self.board[cy][cx] != who:
                        break
                    seq.append((cx, cy))
            if len(seq) >= 5:
                # 5개 이상이면 5개만 강조 (시각적 일관성)
                seq.sort()
                self.win_seq = self._pick_five_in_line(seq, dx, dy)
                return True
        return False

    def _pick_five_in_line(self, seq, dx, dy):
        """한 방향 직선에서 연속된 5개 추출 (여러 개면 마지막 5개)"""
        seq = sorted(seq, key=lambda p: (p[0], p[1]))
        if len(seq) == 5:
            return seq
        # 연속 구간 중 5개를 리턴
        # 간단히 끝에서 5개만 선택
        return seq[-5:]

    def empty_cells(self):
        return [(x, y) for y in range(self.size) for x in range(self.size) if self.board[y][x] == 0]


# ==============================
# 간단 AI: 랜덤 + 중앙 선호
# (전시용 데모: 렌더/이펙트가 주역이므로 경량)
# ==============================
def ai_move(game: Omok) -> Optional[Tuple[int, int]]:
    empties = game.empty_cells()
    if not empties:
        return None
    # 중앙 선호 점수
    cx = cy = (game.size - 1) / 2.0
    pts = []
    for (x, y) in empties:
        d = math.hypot(x - cx, y - cy)
        pts.append((d + random.random()*0.15, (x, y)))  # 약간의 랜덤성
    pts.sort(key=lambda t: t[0])
    return pts[0][1]


# ==============================
# 렌더링
# ==============================
def board_rect():
    W = 2 * MARGIN + CELL * (BOARD_SIZE - 1)
    H = 2 * MARGIN + CELL * (BOARD_SIZE - 1)
    return W, H

def draw_wood(surface: pygame.Surface):
    """나무결 느낌 배경(그라데이션 + 가는 세로 결)"""
    W, H = surface.get_size()
    # 기본 그라데이션
    for y in range(H):
        r = WOOD_LIGHT[0] + int((WOOD_DARK[0] - WOOD_LIGHT[0]) * y / H)
        g = WOOD_LIGHT[1] + int((WOOD_DARK[1] - WOOD_LIGHT[1]) * y / H)
        b = WOOD_LIGHT[2] + int((WOOD_DARK[2] - WOOD_LIGHT[2]) * y / H)
        pygame.draw.line(surface, (r, g, b), (0, y), (W, y))
    # 가는 결 (투명한 어두운 라인)
    grain = pygame.Surface((W, H), pygame.SRCALPHA)
    for x in range(0, W, 4):
        pygame.draw.line(grain, (0, 0, 0, 12), (x, 0), (x, H))
    surface.blit(grain, (0, 0))

def grid_to_px(ix: int, iy: int) -> Tuple[int, int]:
    """격자 좌표(0~10)를 화면 좌표로 변환"""
    px = MARGIN + ix * CELL
    py = MARGIN + iy * CELL
    return px, py

def px_to_grid(px: int, py: int) -> Optional[Tuple[int, int]]:
    """화면 좌표를 가장 가까운 격자 점으로 스냅"""
    # 각 축에서 스냅
    gx = round((px - MARGIN) / CELL)
    gy = round((py - MARGIN) / CELL)
    if 0 <= gx < BOARD_SIZE and 0 <= gy < BOARD_SIZE:
        # 실제 점과의 거리 확인(클릭이 너무 멀면 무시)
        sx, sy = grid_to_px(gx, gy)
        if math.hypot(px - sx, py - sy) <= CELL * 0.45:
            return gx, gy
    return None

def draw_grid(surface: pygame.Surface):
    # 외곽 테두리
    sx, sy = grid_to_px(0, 0)
    ex, ey = grid_to_px(BOARD_SIZE - 1, BOARD_SIZE - 1)
    pygame.draw.rect(surface, BORDER, (sx - BORDER_W, sy - BORDER_W,
                                       (ex - sx) + BORDER_W * 2, (ey - sy) + BORDER_W * 2), BORDER_W)

    # 격자
    for i in range(BOARD_SIZE):
        x1, y = grid_to_px(0, i)
        x2, _ = grid_to_px(BOARD_SIZE - 1, i)
        pygame.draw.line(surface, GRID, (x1, y), (x2, y), LINE_W)

        x, y1 = grid_to_px(i, 0)
        _, y2 = grid_to_px(i, BOARD_SIZE - 1)
        pygame.draw.line(surface, GRID, (x, y1), (x, y2), LINE_W)

    # 성화점(별점) — 11x11 기준: (3,3), (3,7), (7,3), (7,7), (5,5)
    stars = [(3, 3), (3, 7), (7, 3), (7, 7), (5, 5)]
    for (ix, iy) in stars:
        cx, cy = grid_to_px(ix, iy)
        pygame.gfxdraw.filled_circle(surface, cx, cy, STAR_R, GRID)

def draw_stone(surface: pygame.Surface, ix: int, iy: int, who: int, glow: bool, t_now: float, place_time_map: dict):
    """돌 렌더링: 그림자 + 반사광 + 착수 glow"""
    cx, cy = grid_to_px(ix, iy)
    r = CELL // 2 - 3

    # 그림자
    pygame.gfxdraw.filled_circle(surface, cx + 2, cy + 2, r, (0, 0, 0))
    pygame.gfxdraw.filled_circle(surface, cx + 2, cy + 2, r - 1, (0, 0, 0, 180))

    # 본체
    color = BLACK if who == 1 else WHITE
    pygame.gfxdraw.filled_circle(surface, cx, cy, r, color)
    pygame.gfxdraw.aacircle(surface, cx, cy, r, color)

    # 반사광
    hl_color = (255, 255, 255) if who == 1 else (110, 110, 110)
    pygame.gfxdraw.filled_circle(surface, cx - r // 3, cy - r // 3, max(2, r // 5), hl_color)

    # 착수 Glow 애니메이션 (최근 착수만)
    if glow and (ix, iy) in place_time_map:
        t0 = place_time_map[(ix, iy)]
        dt = t_now - t0
        # 0.5초 동안 반경이 커지며 투명해짐
        dur = 0.5
        if 0 <= dt <= dur:
            prog = dt / dur
            gr = r + int(10 + 18 * prog)
            alpha = int(max(0, 180 * (1.0 - prog)))
            ring = pygame.Surface((gr * 2 + 4, gr * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring, (255, 215, 0, alpha), (gr + 2, gr + 2), gr, width=3)
            surface.blit(ring, (cx - gr - 2, cy - gr - 2))

def draw_win_effect(surface: pygame.Surface, win_seq: List[Tuple[int, int]], t_now: float):
    """승리 5목 강조: 라인 + 반짝 하이라이트"""
    if not win_seq:
        return
    # 라인
    pts = [grid_to_px(ix, iy) for (ix, iy) in win_seq]
    # 반짝이는 효과 (alpha 요동)
    alpha = int(100 + 60 * (1 + math.sin(t_now * 6)) / 2)
    line_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.lines(line_surf, (255, 100, 100, alpha), False, pts, 6)
    surface.blit(line_surf, (0, 0))

    # 각 돌 하이라이트
    halo = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for (ix, iy) in win_seq:
        cx, cy = grid_to_px(ix, iy)
        pygame.draw.circle(halo, WIN_HALO, (cx, cy), CELL // 2 + 6)
    surface.blit(halo, (0, 0))

def draw_hud(surface: pygame.Surface, game: Omok, human_vs_ai: bool, show_coords: bool, font):
    W, H = surface.get_size()
    pad = 10
    # 상태
    mode = "사람 vs AI" if human_vs_ai else "사람 vs 사람"
    turn = "흑(●)" if game.turn == 1 else "백(○)"
    if game.winner != 0:
        turn = "흑(●) 승리!" if game.winner == 1 else "백(○) 승리!"
    msg = f"{mode}   |   현재: {turn}"
    text = font.render(msg, True, HUD)
    surface.blit(text, (pad, pad))

    # 도움말
    help_font = pygame.font.SysFont(None, 18)
    help_text = help_font.render(HELP_TEXT, True, (60, 60, 60))
    surface.blit(help_text, (pad, H - help_text.get_height() - pad))

    # 좌표 표시 옵션
    if show_coords:
        small = pygame.font.SysFont(None, 18)
        # 위/왼쪽 좌표
        for i in range(BOARD_SIZE):
            x, y = grid_to_px(i, 0)
            lab = small.render(str(i), True, (50, 50, 50))
            surface.blit(lab, (x - lab.get_width() // 2, y - 24))
            x, y = grid_to_px(0, i)
            lab = small.render(str(i), True, (50, 50, 50))
            surface.blit(lab, (x - 24, y - lab.get_height() // 2))

def main():
    pygame.init()
    pygame.display.set_caption("Omok 11x11 — AlphaZero 전시용 보드 (렌주룰 미적용)")
    W, H = board_rect()
    # HUD 공간 조금 더
    screen = pygame.display.set_mode((W, H + 40))
    clock = pygame.time.Clock()

    # 오디오(효과음)
    have_sound = False
    click_sound = None
    try:
        pygame.mixer.pre_init(22050, -16, 2, 256)
        pygame.mixer.init()
        click_sound = make_click_sound()
        have_sound = click_sound is not None
    except Exception:
        have_sound = False

    # 폰트
    font = pygame.font.SysFont(None, 26)

    game = Omok(BOARD_SIZE)
    human_vs_ai = True
    show_coords = False

    # 최근 착수 시간 기록(Glow 애니메이션용)
    place_time_map = {}

    # AI가 백(두 번째)로 시작
    waiting_ai = False

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t_now = time.time()

        # 이벤트 처리
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    game.reset()
                    place_time_map.clear()
                    waiting_ai = False
                elif e.key == pygame.K_a:
                    human_vs_ai = not human_vs_ai
                    game.reset()
                    place_time_map.clear()
                    waiting_ai = False
                elif e.key == pygame.K_h:
                    show_coords = not show_coords
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if game.winner == 0:
                    pos = pygame.mouse.get_pos()
                    gx_gy = px_to_grid(*pos)
                    if gx_gy is not None:
                        x, y = gx_gy
                        if game.place(x, y):
                            place_time_map[(x, y)] = t_now
                            if have_sound:
                                click_sound.play()
                            # AI 차례면 플래그
                            if human_vs_ai and game.winner == 0 and game.turn == -1:
                                waiting_ai = True

        # AI 착수
        if running and human_vs_ai and waiting_ai and game.winner == 0 and game.turn == -1:
            move = ai_move(game)
            if move:
                x, y = move
                if game.place(x, y):
                    place_time_map[(x, y)] = time.time()
                    if have_sound:
                        click_sound.play()
            waiting_ai = False

        # 렌더
        screen.fill(WOOD_LIGHT)
        draw_wood(screen)
        draw_grid(screen)

        # 돌
        for iy in range(BOARD_SIZE):
            for ix in range(BOARD_SIZE):
                who = game.board[iy][ix]
                if who != 0:
                    is_last = (game.last_move == (ix, iy))
                    draw_stone(screen, ix, iy, who, glow=is_last, t_now=time.time(), place_time_map=place_time_map)

        # 승리 이펙트
        if game.winner != 0 and game.win_seq:
            draw_win_effect(screen, game.win_seq, t_now=time.time())

        # HUD
        draw_hud(screen, game, human_vs_ai, show_coords, font)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
