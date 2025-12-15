# pygame_env.py v10 — 착수 시 날아오는 애니메이션 추가
import os
import math
import time
from typing import Optional, Tuple, List

import numpy as np
import pygame
import pygame.gfxdraw

# ===================== 사용자 커스텀 =====================
CREATOR_ME_NAME = "이용민"
CREATOR_PARTNER_NAME = "이민우"

# 예) 절대경로 / 또는 프로젝트 상대경로 둘 중 하나만 사용
AVATAR_ME_PATH = r"C:\Users\user\PycharmProjects\Woongwon\photo1.jpg"
AVATAR_PARTNER_PATH = r"C:\Users\user\PycharmProjects\Woongwon\photo2.jpg"
# AVATAR_ME_PATH = os.path.join("assets", "photo1.jpg")
# AVATAR_PARTNER_PATH = os.path.join("assets", "photo2.jpg")


# ===================== 디자인 & 레이아웃 상수 =====================
# --- 보드 ---
BOARD_SIZE = 11
CELL = 60
MARGIN = 80
LINE_W = 2
BORDER_W = 4
STAR_R = 5

# --- HUD ---
HUD_H = 130

# --- 창 크기 (자동 계산) ---
WINDOW_WIDTH = 2 * MARGIN + CELL * (BOARD_SIZE - 1)
WINDOW_HEIGHT = 2 * MARGIN + CELL * (BOARD_SIZE - 1) + HUD_H + 20

# --- 색상 (테마) ---
ACCENT = (60, 120, 255)
ACCENT_DARK = (40, 80, 170)
WOOD_LIGHT = (219, 184, 136)
WOOD_DARK = (196, 160, 118)
GRID = (70, 50, 20)
BORDER = (60, 45, 18)
BLACK = (15, 15, 15)
WHITE = (245, 245, 245)
HUD_BG = (30, 30, 30)
HUD_TEXT = (230, 230, 230)
HELP_GRAY = (120, 120, 120)
WIN_HALO = (255, 120, 120, 60)

# --- 기타 ---
FPS = 60
HELP_TEXT = "[Click] 착수  |  [R] 재시작  |  [ESC] 종료"


# ===================== 유틸리티 함수 =====================
def _load_font_path():
    """시스템에서 사용 가능한 한글 폰트 경로를 찾습니다."""
    for p in [
        os.path.join("fonts", "NanumSquareRoundR.ttf"), os.path.join("fonts", "NanumGothic.ttf"),
        os.path.join("fonts", "NotoSansKR-Regular.otf"), os.path.join("fonts", "NotoSansKR-Regular.ttf"),
    ]:
        if os.path.exists(p): return p
    try:
        found = pygame.font.match_font(
            ["NanumSquareRound", "NanumGothic", "Noto Sans KR", "Malgun Gothic", "AppleGothic"])
        if found: return found
    except Exception:
        pass
    return None


def load_font(size):
    """지정된 크기의 폰트를 로드합니다."""
    path = _load_font_path()
    try:
        return pygame.font.Font(path, size) if path else pygame.font.SysFont("Malgun Gothic", size)
    except Exception:
        return pygame.font.SysFont("Malgun Gothic", size)


def draw_shadowed_rect(surface, rect, fill, radius=14, shadow=(0, 0, 0, 70), offset=(4, 6)):
    """그림자가 있는 둥근 사각형을 그립니다."""
    x, y, w, h = rect
    shadow_surf = pygame.Surface((w + 12, h + 12), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, shadow, (6, 6, w, h), border_radius=radius + 2)
    surface.blit(shadow_surf, (x + offset[0] - 6, y + offset[1] - 6))
    pygame.draw.rect(surface, fill, (x, y, w, h), border_radius=radius)


def _load_avatar(path: str, size: int) -> Optional[pygame.Surface]:
    """경로에서 이미지를 로드하여 원형으로 자릅니다."""
    try:
        img = pygame.image.load(path).convert_alpha()
    except Exception:
        return None
    img = pygame.transform.smoothscale(img, (size, size))
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255), (size // 2, size // 2), size // 2)
    circ = pygame.Surface((size, size), pygame.SRCALPHA)
    circ.blit(img, (0, 0))
    circ.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return circ


def _avatar_or_initials(name: str, size: int, path: Optional[str]):
    """사진이 있으면 로드하고, 없으면 이름 이니셜로 아바타를 생성합니다."""
    surf = _load_avatar(path, size)
    if surf: return surf
    circle = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.gfxdraw.filled_circle(circle, size // 2, size // 2, size // 2, (235, 235, 240))
    pygame.gfxdraw.aacircle(circle, size // 2, size // 2, size // 2, (210, 210, 215))
    ini = ""
    if name.strip():
        parts = name.strip().split()
        ini = parts[0][0] + parts[-1][0] if len(parts) >= 2 else name[:2]
    ini = ini.upper()
    font = load_font(max(18, size // 3))
    text = font.render(ini, True, (80, 80, 90))
    circle.blit(text, ((size - text.get_width()) // 2, (size - text.get_height()) // 2))
    return circle


# ===================== 메인 환경 클래스 =====================
class PygameEnv:
    """오목 게임의 렌더링 및 상태 관리를 담당하는 클래스"""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Gomoku 11x11 AI Match")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.board: List[List[int]] = []
        self.turn: int = 0
        self.win_index: int = 0
        self.last_move: Optional[Tuple[int, int]] = None
        self.win_seq: List[Tuple[int, int]] = []
        self.placed_at: dict = {}
        self.restart_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.animations = []  # NEW: 애니메이션 관리 리스트
        self.reset()
        self._load_assets()

    def _load_assets(self):
        """게임에 필요한 폰트, 사운드, 아바타를 미리 로드합니다."""
        self.fonts = {
            'title': load_font(32), 'status': load_font(20), 'help': load_font(18),
            'name': load_font(22), 'made_by': load_font(16),
        }
        try:
            pygame.mixer.pre_init(22050, -16, 2, 256)
            pygame.mixer.init()
            self.click_sound = self._make_click_sound()
            self.have_sound = self.click_sound is not None
        except Exception:
            self.click_sound, self.have_sound = None, False

        self.creators = [
            {'name': CREATOR_ME_NAME, 'path': AVATAR_ME_PATH},
            {'name': CREATOR_PARTNER_NAME, 'path': AVATAR_PARTNER_PATH}
        ]
        self.avatars = [_avatar_or_initials(p['name'], 64, p['path']) for p in self.creators]

    def _grid_to_px(self, ix: int, iy: int) -> Tuple[int, int]:
        return MARGIN + ix * CELL, HUD_H + MARGIN + iy * CELL

    def _px_to_grid(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        if not (MARGIN // 2 < px < WINDOW_WIDTH - MARGIN // 2 and
                HUD_H + MARGIN // 2 < py < WINDOW_HEIGHT - MARGIN // 2):
            return None
        gy = round((py - (HUD_H + MARGIN)) / CELL)
        gx = round((px - MARGIN) / CELL)
        if 0 <= gx < BOARD_SIZE and 0 <= gy < BOARD_SIZE:
            sx, sy = self._grid_to_px(gx, gy)
            if math.hypot(px - sx, py - sy) <= CELL * 0.45:
                return gy, gx
        return None

    def _draw_board_base(self):
        self.screen.fill(WOOD_LIGHT)
        W, H = self.screen.get_size()
        for y in range(H):
            r = WOOD_LIGHT[0] + int((WOOD_DARK[0] - WOOD_LIGHT[0]) * y / H)
            g = WOOD_LIGHT[1] + int((WOOD_DARK[1] - WOOD_LIGHT[1]) * y / H)
            b = WOOD_LIGHT[2] + int((WOOD_DARK[2] - WOOD_LIGHT[2]) * y / H)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (W, y))
        grain = pygame.Surface((W, H), pygame.SRCALPHA)
        for x in range(0, W, 4):
            pygame.draw.line(grain, (0, 0, 0, 12), (x, 0), (x, H))
        self.screen.blit(grain, (0, 0))
        sx, sy = self._grid_to_px(0, 0)
        ex, ey = self._grid_to_px(BOARD_SIZE - 1, BOARD_SIZE - 1)
        pygame.draw.rect(self.screen, BORDER, (sx - BORDER_W, sy - BORDER_W,
                                               (ex - sx) + BORDER_W * 2, (ey - sy) + BORDER_W * 2), BORDER_W, 4)
        for i in range(BOARD_SIZE):
            x1, y_ = self._grid_to_px(0, i);
            x2, _ = self._grid_to_px(BOARD_SIZE - 1, i)
            pygame.draw.line(self.screen, GRID, (x1, y_), (x2, y_), LINE_W)
            x_, y1 = self._grid_to_px(i, 0);
            _, y2 = self._grid_to_px(i, BOARD_SIZE - 1)
            pygame.draw.line(self.screen, GRID, (x_, y1), (x_, y2), LINE_W)
        for (ix, iy) in [(3, 3), (3, 7), (7, 3), (7, 7), (BOARD_SIZE // 2, BOARD_SIZE // 2)]:
            cx, cy = self._grid_to_px(ix, iy)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, STAR_R, GRID)

    def _draw_stones_and_effects(self):
        t_now = time.time()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] != 0:
                    self._draw_stone_at_grid(r, c, self.board[r][c], t_now)
        if self.win_index != 0 and self.win_seq:
            self._draw_win_effect(t_now)
        self._draw_hover_hint()

    def _draw_stone_at_px(self, cx, cy, who):  # CHANGED: 픽셀 좌표로 돌을 그리는 함수
        r = CELL // 2 - 3
        pygame.gfxdraw.filled_circle(self.screen, int(cx) + 2, int(cy) + 2, r, (0, 0, 0, 180))
        color = BLACK if who == 1 else WHITE
        pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), r, color)
        pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), r, color)
        hl_color = (255, 255, 255) if who == 1 else (110, 110, 110)
        pygame.gfxdraw.filled_circle(self.screen, int(cx) - r // 3, int(cy) - r // 3, max(2, r // 5), hl_color)

    def _draw_stone_at_grid(self, row, col, who, t_now):  # CHANGED: 기존 돌 그리기 함수 이름 변경
        cx, cy = self._grid_to_px(col, row)
        self._draw_stone_at_px(cx, cy, who)
        if self.last_move and (row, col) == self.last_move and (row, col) in self.placed_at:
            dt = t_now - self.placed_at[(row, col)]
            if 0 <= dt <= 0.5:
                prog, r = dt / 0.5, CELL // 2 - 3
                gr = r + int(10 + 18 * prog)
                alpha = int(max(0, 180 * (1.0 - prog)))
                ring = pygame.Surface((gr * 2 + 4, gr * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(ring, (255, 215, 0, alpha), (gr + 2, gr + 2), gr, width=3)
                self.screen.blit(ring, (cx - gr - 2, cy - gr - 2))

    def _draw_win_effect(self, t_now):
        if not self.win_seq: return
        pts = [self._grid_to_px(c, r) for (r, c) in self.win_seq]
        alpha = int(100 + 60 * (1 + math.sin(t_now * 6)) / 2)
        line_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(line_surf, (255, 100, 100, alpha), False, pts, 8)
        self.screen.blit(line_surf, (0, 0))
        halo = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for (r, c) in self.win_seq:
            cx, cy = self._grid_to_px(c, r)
            pygame.draw.circle(halo, WIN_HALO, (cx, cy), CELL // 2 + 6)
        self.screen.blit(halo, (0, 0))

    def _draw_hover_hint(self):
        if self.animations: return  # 애니메이션 중에는 호버 표시 안 함
        hit = self._px_to_grid(*pygame.mouse.get_pos())
        if hit and self.board[hit[0]][hit[1]] == 0:
            cx, cy = self._grid_to_px(hit[1], hit[0])
            color = (0, 0, 0, 50) if self.turn == 0 else (255, 255, 255, 80)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, CELL // 2 - 4, color)

    def _draw_hud_and_controls(self):
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, WINDOW_WIDTH, HUD_H))
        title = self.fonts['title'].render("GOMOKU", True, WHITE)
        self.screen.blit(title, (30, (HUD_H // 2) - title.get_height() + 10))
        if self.win_index == 1:
            status_text = "흑(●) 승리!"
        elif self.win_index == -1:
            status_text = "백(○) 승리!"
        elif self.turn == 0:
            status_text = "사람(흑)의 차례입니다."
        else:
            status_text = "AI(백)의 차례입니다."
        status = self.fonts['status'].render(status_text, True, HUD_TEXT)
        self.screen.blit(status, (32, (HUD_H // 2) + 10))
        self._draw_creator_info()
        help_surf = self.fonts['help'].render(HELP_TEXT, True, HELP_GRAY)
        self.screen.blit(help_surf, (20, WINDOW_HEIGHT - help_surf.get_height() - 10))
        btn_w, btn_h = 130, 44
        btn_x, btn_y = WINDOW_WIDTH - btn_w - 20, WINDOW_HEIGHT - btn_h - 15
        self.restart_btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        hover = self.restart_btn_rect.collidepoint(pygame.mouse.get_pos())
        bg = [min(255, c + 15) for c in ACCENT] if hover else ACCENT
        draw_shadowed_rect(self.screen, self.restart_btn_rect, bg, radius=10, offset=(2, 3), shadow=(0, 0, 0, 100))
        pygame.draw.rect(self.screen, ACCENT_DARK, self.restart_btn_rect, 2, 10)
        label = self.fonts['status'].render("재시작", True, WHITE)
        self.screen.blit(label, (btn_x + (btn_w - label.get_width()) // 2, btn_y + (btn_h - label.get_height()) // 2))

    def _draw_creator_info(self):
        title_surf = self.fonts['made_by'].render("MADE BY", True, (150, 150, 160))
        avatar1, name1 = self.avatars[0], self.fonts['name'].render(self.creators[0]['name'], True, WHITE)
        avatar2, name2 = self.avatars[1], self.fonts['name'].render(self.creators[1]['name'], True, WHITE)
        chip1_width, chip2_width = avatar1.get_width() + 10 + name1.get_width(), avatar2.get_width() + 10 + name2.get_width()
        # 수정된 코드
        gap = 30  # 칩 사이의 간격
        total_width = chip1_width + gap + chip2_width
        block_start_x = WINDOW_WIDTH - total_width - 30
        title_x, title_y = block_start_x, (HUD_H // 2) - 35
        self.screen.blit(title_surf, (title_x, title_y))
        chips_y = title_y + title_surf.get_height() - 5
        current_x = block_start_x
        self.screen.blit(avatar1, (current_x, chips_y))
        self.screen.blit(name1, (current_x + avatar1.get_width() + 10,
                                 chips_y + (avatar1.get_height() - name1.get_height()) // 2))
        current_x += chip1_width + gap
        self.screen.blit(avatar2, (current_x, chips_y))
        self.screen.blit(name2, (current_x + avatar2.get_width() + 10,
                                 chips_y + (avatar2.get_height() - name2.get_height()) // 2))

    # NEW: 애니메이션 그리기 및 관리 함수
    def _draw_animations(self):
        if not self.animations: return

        t_now = time.time()
        remaining_animations = []

        for anim in self.animations:
            elapsed = t_now - anim['start_time']
            progress = min(1.0, elapsed / anim['duration'])

            # 부드러운 도착을 위한 이징(easing) 함수
            ease_progress = 1 - (1 - progress) ** 3

            px = anim['start_pos'][0] + (anim['end_pos'][0] - anim['start_pos'][0]) * ease_progress
            py = anim['start_pos'][1] + (anim['end_pos'][1] - anim['start_pos'][1]) * ease_progress

            self._draw_stone_at_px(px, py, anim['who'])

            if progress < 1.0:
                remaining_animations.append(anim)
            else:  # 애니메이션 완료
                row, col = anim['target_cell']
                self.board[row][col] = anim['who']
                self.last_move = (row, col)
                self.placed_at[(row, col)] = t_now
                if self.have_sound: self.click_sound.play()
                if self._check_win_from(row, col):
                    self.win_index = anim['who']

        self.animations = remaining_animations

    def _make_click_sound(self, sample_rate=22050, ms=55, freq=220):
        try:
            import pygame.sndarray
            t = np.linspace(0, ms / 1000.0, int(sample_rate * (ms / 1000.0)), endpoint=False)
            wave = (0.7 * np.sin(2 * np.pi * freq * t) * np.exp(-8 * t)).astype(np.float32)
            stereo = np.repeat(wave[:, None], 2, axis=1)
            snd = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
            snd.set_volume(0.25)
            return snd
        except (ImportError, pygame.error):
            return None

    def reset(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.turn, self.win_index = 0, 0
        self.last_move, self.win_seq, self.placed_at = None, [], {}
        self.animations = []

    def get_mouse_click(self, pos) -> Optional[Tuple[int, int]]:
        return self._px_to_grid(*pos)

    def is_restart_button_clicked(self, pos) -> bool:
        return self.restart_btn_rect and self.restart_btn_rect.collidepoint(pos)

    def step(self, action: Tuple[int, int]):  # CHANGED: 애니메이션 시작 로직으로 변경
        if self.win_index != 0 or self.animations:  # 게임이 끝났거나 애니메이션 중이면 수 비활성화
            return self.board, False, self.win_index, self.turn

        row, col = action
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and self.board[row][col] == 0):
            return self.board, False, self.win_index, self.turn

        who = 1 if self.turn == 0 else -1

        # 애니메이션 정보 생성
        end_pos = self._grid_to_px(col, row)
        start_pos = (end_pos[0], HUD_H)  # 화면 위쪽에서 날아오도록 설정

        anim = {
            'who': who,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'target_cell': (row, col),
            'start_time': time.time(),
            'duration': 0.25,  # 애니메이션 시간 (초)
        }
        self.animations.append(anim)

        # 턴은 즉시 넘겨서 AI가 생각할 시간을 줌
        self.turn = 1 - self.turn

        # 아직 보드 상태는 바뀌지 않았지만, 유효한 수로 처리
        return self.board, True, self.win_index, self.turn

    def _check_win_from(self, row, col) -> bool:
        who = self.board[row][col]
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            seq = [(row, col)]
            for sgn in (1, -1):
                r, c = row, col
                for _ in range(BOARD_SIZE):
                    r, c = r + dr * sgn, c + dc * sgn
                    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == who): break
                    seq.append((r, c))
            if len(seq) >= 5:
                self.win_seq = self._find_five_in_a_row(sorted(seq))
                return True
        return False

    def _find_five_in_a_row(self, seq: List[Tuple[int, int]]):
        if not seq: return []
        for i in range(len(seq) - 4):
            is_continuous = all(
                (seq[i + j + 1][0] - seq[i + j][0]) ** 2 + (seq[i + j + 1][1] - seq[i + j][1]) ** 2 <= 2
                for j in range(4)
            )
            if is_continuous: return seq[i:i + 5]
        return seq[:5]

    def render(self):  # CHANGED: 애니메이션 그리기 호출 추가
        self.clock.tick(FPS)
        self._draw_board_base()
        self._draw_stones_and_effects()
        self._draw_animations()  # NEW: 날아오는 돌 그리기
        self._draw_hud_and_controls()
        pygame.display.flip()