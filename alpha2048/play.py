"""Small human-playable desktop game, independent of the model runtime."""
import tkinter as tk
from time import perf_counter
import math
import random

from .core import Action, slide
from .env import Game2048
from .appearance import apply_typography, rounded_family, BG, INK, BOARD, COLORS

KEYS = {'Up': Action.UP, 'w': Action.UP, 'Right': Action.RIGHT, 'd': Action.RIGHT,
        'Down': Action.DOWN, 's': Action.DOWN, 'Left': Action.LEFT, 'a': Action.LEFT}


TEXT = {
    'en': dict(score='SCORE', best='SESSION BEST', over='Game over',
               final='Score: {score:,} · R to restart', help='How to play', close='Got it',
               guide='Move with the arrow keys or W, A, S, D.\n\n'
                     'Tiles slide as far as possible. Two matching tiles merge into one; '
                     'each tile can merge once per move. Merges add to your score.\n\n'
                     'A new 2 or 4 appears after a valid move. Reach 2048, then keep going! '
                     'The game ends when no moves remain.\n\n'
                     '↻ or R starts a new game. Session best lasts until you close the window.\n\n'
                     '中 / EN switches the language without restarting your game.',
               win='2048 reached! Keep going.', blocked='No moves remain.'),
    'zh': dict(score='得分', best='本次最佳', over='游戏结束',
               final='得分：{score:,} · 按 R 重开', help='操作方法', close='知道了',
               guide='使用方向键或 W、A、S、D 移动。\n\n'
                     '方块会滑向所选方向；相同数字相遇时合并，每个方块每步最多合并一次。合并后的数字计入得分。\n\n'
                     '每次有效移动后会出现一个新的 2 或 4。达到 2048 后可以继续；没有合法移动时游戏结束。\n\n'
                     '点击 ↻ 或按 R 开始新一局。本次最佳分数保留到关闭窗口。\n\n'
                     '点击 中 / EN 切换语言，不会重开当前棋局。',
               win='达成 2048！可以继续挑战。', blocked='没有可以移动的方向。'),
}


def tile_paths(board, action):
    """Original tiles and destinations for animation; never modifies game state."""
    action = Action(action)
    paths, merged = [], set()
    for line in range(4):
        if action == Action.LEFT:
            cells = [(line, c) for c in range(4)]
        elif action == Action.RIGHT:
            cells = [(line, c) for c in range(3, -1, -1)]
        elif action == Action.UP:
            cells = [(r, line) for r in range(4)]
        else:
            cells = [(r, line) for r in range(3, -1, -1)]
        occupied = [(pos, int(board[pos])) for pos in cells if board[pos]]
        source, target = 0, 0
        while source < len(occupied):
            pos, value = occupied[source]
            destination = cells[target]
            paths.append((value, pos, destination))
            if source + 1 < len(occupied) and occupied[source + 1][1] == value:
                paths.append((value, occupied[source + 1][0], destination))
                merged.add(destination)
                source += 2
            else:
                source += 1
            target += 1
    return paths, merged


class PlayWindow:
    def __init__(self, root, game=None):
        self.root = root
        self.rounded_family = rounded_family(root)
        self.game = game if game is not None else Game2048()
        self.best = self.game.score
        self.moves = 0
        self.reached_2048 = False
        self.language = 'en'
        self.animation = None
        self.animation_scale = 1.
        self.animation_job = None
        self.pending_action = None
        self.help_window = None
        self.advisor = None
        self.ai_executor = None
        self.ai_provider = None
        self.revision = 0
        self.shake = None
        self.shake_job = None
        self.confetti = None
        self.confetti_job = None
        self.status = tk.StringVar()  # Internal state; no instructional text on the board.
        root.title('2048')
        root.configure(bg=BG)
        root.geometry('780x790')
        root.minsize(780, 530)
        body = tk.Frame(root, bg=BG, padx=20, pady=20)
        body.pack(fill='both', expand=True)
        header = tk.Frame(body, bg=BG)
        header.pack(fill='x', pady=(0, 12))
        self.score_text, self.score_label = self.score_card(header)
        self.best_text, self.best_label = self.score_card(header)
        self.help_button = self.button(header, '?', self.show_help)
        self.help_button.pack(side='right', padx=(6, 0))
        self.ai_button = self.button(header, 'AI', self.show_ai)
        self.ai_button.pack(side='right', padx=(6, 0))
        self.new_button = self.button(header, '↻', self.restart)
        self.new_button.pack(side='right', padx=(6, 0))
        self.language_button = self.button(header, '中', self.toggle_language)
        self.language_button.pack(side='right', padx=(6, 0))
        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0, height=1, width=1,
                                takefocus=True)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _: self.draw())
        self.canvas.bind('<Button-1>', lambda _: self.canvas.focus_set())
        root.bind('<KeyPress>', self.on_key)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.translate()
        self.refresh()
        self.canvas.focus_set()

    @staticmethod
    def button(parent, text, command):
        return tk.Button(parent, text=text, command=command, bg='#8f7a66', fg='white',
                         activebackground='#776452', activeforeground='white',
                         relief='flat', bd=0, width=4, padx=9, pady=8, cursor='hand2',
                         font=('Microsoft YaHei UI', 13, 'bold'))

    @staticmethod
    def score_card(parent):
        card = tk.Frame(parent, bg=BOARD, padx=10, pady=5)
        card.pack(side='left', padx=(0, 8))
        label = tk.Label(card, bg=BOARD, fg='#f9f6f2', font=('Microsoft YaHei UI', 9, 'bold'))
        label.pack(side='left', padx=(0, 8))
        value = tk.StringVar(value='0')
        tk.Label(card, textvariable=value, bg=BOARD, fg='white',
                 font=('Segoe UI', 16, 'bold')).pack(side='left')
        return value, label

    def translate(self):
        text = TEXT[self.language]
        self.score_label.configure(text=text['score'])
        self.best_label.configure(text=text['best'])
        self.language_button.configure(text='中' if self.language == 'en' else 'EN')
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.title(text['help'])
            self.help_heading.configure(text=text['help'])
            self.help_body.configure(text=text['guide'])
            self.help_close.configure(text=text['close'])
        if self.advisor is not None:
            self.advisor.translate()
        apply_typography(self.root, self.language)
        self.draw()

    def show_ai(self):
        if self.advisor is not None:
            self.advisor.window.lift()
            return
        from .advisor import AdvisorWindow
        self.advisor = AdvisorWindow(self)

    def toggle_language(self):
        self.language = 'zh' if self.language == 'en' else 'en'
        self.translate()
        self.canvas.focus_set()

    def show_help(self):
        if self.advisor is not None:
            self.advisor.pause()
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.lift()
            return
        window = self.help_window = tk.Toplevel(self.root)
        window.configure(bg=BG)
        window.transient(self.root)
        window.resizable(False, False)
        self.help_heading = tk.Label(window, bg=BG, fg=INK,
                                      font=('Microsoft YaHei UI', 18, 'bold'))
        self.help_heading.pack(anchor='w', padx=24, pady=(22, 12))
        self.help_body = tk.Label(window, bg=BG, fg=INK, justify='left', wraplength=390,
                                 font=('Microsoft YaHei UI', 11))
        self.help_body.pack(padx=24)
        self.help_close = tk.Button(window, command=self.close_help, bg='#8f7a66',
                                    fg='white', relief='flat', padx=18, pady=8,
                                    font=('Microsoft YaHei UI', 10))
        self.help_close.pack(anchor='e', padx=24, pady=20)
        window.protocol('WM_DELETE_WINDOW', self.close_help)
        window.bind('<Escape>', lambda _: self.close_help())
        self.translate()
        window.update_idletasks()
        window.geometry(f'+{max(0, self.root.winfo_rootx() + 30)}+{max(0, self.root.winfo_rooty() + 30)}')
        window.grab_set()
        self.help_close.focus_set()

    def close_help(self):
        if self.help_window is not None:
            self.help_window.grab_release()
            self.help_window.destroy()
            self.help_window = None
        self.canvas.focus_set()

    def on_key(self, event):
        if self.help_window is not None:
            return
        key = event.keysym
        action = KEYS.get(key, KEYS.get(key.lower()))
        if action is not None:
            self.move(action)
            return 'break'
        if key.lower() == 'r':
            self.restart()
            return 'break'
        if key in ('question', 'F1'):
            self.show_help()
            return 'break'

    def move(self, action, from_ai=False):
        if not from_ai and self.advisor is not None:
            self.advisor.pause()
        if self.animation is not None:
            self.pending_action = action  # Buffer one direction, never a long delayed queue.
            return
        before = self.game.board
        result = self.game.step(action)
        self.cancel_shake()
        if result.changed:
            self.revision += 1
            self.moves += 1
            self.best = max(self.best, result.score)
            paths, merged = tile_paths(before, action)
            after, _, _ = slide(before, action)
            spawned = {(r, c) for r in range(4) for c in range(4)
                       if after[r, c] == 0 and result.board[r, c] != 0}
            if self.animation_scale > 0:
                self.animation = dict(start=perf_counter(), paths=paths, merged=merged,
                                      spawned=spawned, scale=self.animation_scale)
            if self.advisor is not None:
                self.advisor.invalidate()
        else:
            self.shake = dict(start=perf_counter(), action=Action(action))
            self.shake_tick()
        if result.changed and not self.reached_2048 and result.board.max() >= 2048:
            self.reached_2048 = True
            self.status.set(TEXT[self.language]['win'])
            if self.advisor is not None:
                self.advisor.pause()
            self.start_confetti()
        if result.terminated:
            self.status.set(TEXT[self.language]['blocked'])
        self.refresh()
        if self.animation is not None:
            self.tick()

    def tick(self):
        self.animation_job = None
        if self.animation is None:
            return
        if perf_counter() - self.animation['start'] >= .23 * self.animation['scale']:
            self.animation = None
            self.draw()
            action, self.pending_action = self.pending_action, None
            if action is not None:
                self.move(action)
            return
        self.draw()
        self.animation_job = self.root.after(8 if self.animation['scale'] < 1 else 12, self.tick)

    def set_animation_scale(self, scale):
        self.animation_scale = scale
        if self.animation is not None:
            if scale == 0:
                self.cancel_animation()
                self.draw()
            else:
                now = perf_counter()
                elapsed = (now - self.animation['start']) / self.animation['scale']
                self.animation['start'] = now - elapsed * scale
                self.animation['scale'] = scale

    def cancel_animation(self):
        if self.animation_job is not None:
            self.root.after_cancel(self.animation_job)
        self.animation_job = None
        self.animation = None
        self.pending_action = None

    def shake_tick(self):
        self.shake_job = None
        if self.shake is None:
            return
        if perf_counter() - self.shake['start'] >= .22:
            self.shake = None
            self.draw()
            return
        self.draw()
        self.shake_job = self.root.after(12, self.shake_tick)

    def cancel_shake(self):
        if self.shake_job is not None:
            self.root.after_cancel(self.shake_job)
        self.shake_job = None
        self.shake = None

    def start_confetti(self):
        self.cancel_confetti()
        # Cosmetic randomness must never consume the game's tile-spawn RNG.
        rng = random.Random()
        colors = ('#edc22e', '#f59563', '#e96f74', '#69b7a2', '#79a9d1', '#b195ce')
        particles = [dict(x=rng.random(), y=rng.uniform(-.65, -.02),
                          speed=rng.uniform(.42, .70), drift=rng.uniform(-.08, .08),
                          phase=rng.uniform(0, math.tau), spin=rng.uniform(-8, 8),
                          color=rng.choice(colors), size=rng.uniform(4, 7)) for _ in range(110)]
        self.confetti = dict(start=perf_counter(), particles=particles)
        self.confetti_tick()

    def confetti_tick(self):
        self.confetti_job = None
        if self.confetti is None:
            return
        if perf_counter() - self.confetti['start'] >= 2.8:
            self.confetti = None
            self.draw()
            return
        self.draw()
        self.confetti_job = self.root.after(16, self.confetti_tick)

    def cancel_confetti(self):
        if self.confetti_job is not None:
            self.root.after_cancel(self.confetti_job)
        self.confetti_job = None
        self.confetti = None

    def draw_confetti(self):
        if self.confetti is None:
            return
        elapsed = perf_counter() - self.confetti['start']
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        for particle in self.confetti['particles']:
            angle = particle['phase'] + particle['spin'] * elapsed
            x = width * (particle['x'] + particle['drift'] * elapsed) + 12 * math.sin(angle)
            y = height * (particle['y'] + particle['speed'] * elapsed)
            if not (-15 < x < width + 15 and -15 < y < height + 15):
                continue
            size = particle['size']
            points = []
            for dx, dy in ((-size, -size/2), (size, -size/2), (size, size/2), (-size, size/2)):
                points.extend((x + dx * math.cos(angle) - dy * math.sin(angle),
                               y + dx * math.sin(angle) + dy * math.cos(angle)))
            self.canvas.create_polygon(points, fill=particle['color'], outline='', tags='confetti')

    def restart(self):
        if self.advisor is not None:
            self.advisor.pause()
        self.cancel_animation()
        self.cancel_shake()
        self.cancel_confetti()
        self.game.reset()
        self.revision += 1
        if self.advisor is not None:
            self.advisor.invalidate()
        self.moves = 0
        self.reached_2048 = False
        self.status.set('')
        self.refresh()
        self.canvas.focus_set()

    def close(self):
        if self.advisor is not None:
            self.advisor.close()
        if self.ai_provider is not None:
            self.ai_provider.cancelled.set()
        if self.ai_executor is not None:
            self.ai_executor.shutdown(wait=False, cancel_futures=True)
        self.cancel_animation()
        self.cancel_shake()
        self.cancel_confetti()
        self.root.destroy()

    def refresh(self):
        self.score_text.set(f'{self.game.score:,}')
        self.best_text.set(f'{self.best:,}')
        self.draw()

    def draw(self):
        canvas = self.canvas
        canvas.delete('all')
        side = max(1, min(canvas.winfo_width(), canvas.winfo_height() - 40) - 16)
        left = (canvas.winfo_width() - side) / 2
        top = 8
        if self.shake is not None:
            progress = min(1, max(0, (perf_counter() - self.shake['start']) / .22))
            offset = 6 * math.sin(progress * math.pi * 6) * (1 - progress)
            if self.shake['action'] in (Action.LEFT, Action.RIGHT):
                left += offset
            else:
                top += offset
        gap = max(4, side * .022)
        cell = max(1, (side - gap * 5) / 4)
        canvas.create_rectangle(left, top, left + side, top + side, fill=BOARD, outline='')

        def tile(value, row, col, scale=1):
            x = left + gap + col * (cell + gap) + cell / 2
            y = top + gap + row * (cell + gap) + cell / 2
            half = cell * scale / 2
            canvas.create_rectangle(x - half, y - half, x + half, y + half,
                                    fill=COLORS.get(value, '#3c3a32'), outline='')
            if value:
                size = max(1, int(cell * (.34 if len(str(value)) <= 3 else .25) * scale))
                canvas.create_text(x, y, text=str(value), fill=INK if value <= 4 else '#f9f6f2',
                                   font=(self.rounded_family, size, 'bold'))

        for row in range(4):
            for col in range(4):
                tile(0, row, col)
        animation = self.animation
        elapsed = (perf_counter() - animation['start']) / animation['scale'] if animation else 1
        if animation and elapsed < .13:
            progress = min(1, elapsed / .13)
            progress = 1 - (1 - progress) ** 3
            for value, (r0, c0), (r1, c1) in animation['paths']:
                tile(value, r0 + (r1 - r0) * progress, c0 + (c1 - c0) * progress)
        else:
            progress = min(1, max(0, (elapsed - .13) / .10))
            for row, values in enumerate(self.game.board):
                for col, value in enumerate(values):
                    if not value:
                        continue
                    scale = 1
                    if animation and (row, col) in animation['spawned']:
                        scale = .35 + .65 * progress
                    elif animation and (row, col) in animation['merged']:
                        scale = 1 + .10 * math.sin(math.pi * progress)
                    tile(int(value), row, col, scale)
        if not animation and self.game.terminated:
            text = TEXT[self.language]
            canvas.create_text(left + side / 2, top + side + 24, text=text['over'],
                               fill=INK, font=('Microsoft YaHei UI', 17, 'bold'), tags='game-over')
        self.draw_confetti()


def main():
    root = tk.Tk()
    from .runtime import install_error_handler
    install_error_handler(root, 'game')
    PlayWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
