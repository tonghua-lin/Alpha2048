"""Model playback workspace; training tools are archived."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import traceback
import tkinter as tk
from tkinter import filedialog, ttk
from .run_io import read_json, write_json
PROJECT = Path(__file__).resolve().parents[1]
BG, PANEL, INK, MUTED, ACCENT = ('#f1f5f9', '#ffffff', '#102a43', '#627d98', '#008c82')
TILE_COLORS = {0: '#e5eaf0', 2: '#e0f2f1', 4: '#b2dfdb', 8: '#80cbc4', 16: '#4db6ac', 32: '#26a69a', 64: '#009688', 128: '#00897b', 256: '#00796b', 512: '#00695c', 1024: '#00574b', 2048: '#ffbd59'}

class Board(tk.Canvas):

    def __init__(self, master):
        super().__init__(master, bg=PANEL, height=390, highlightthickness=0)
        self.board = [[0] * 4 for _ in range(4)]
        self.bind('<Configure>', lambda _: self.draw())

    def draw(self, board=None):
        if board is not None:
            self.board = board
        self.delete('all')
        side = min(self.winfo_width() - 24, self.winfo_height() - 24)
        cell = max(20, side / 4)
        left = (self.winfo_width() - 4 * cell) / 2
        top = (self.winfo_height() - 4 * cell) / 2
        for r in range(4):
            for c in range(4):
                value = self.board[r][c]
                x, y = (left + c * cell + 4, top + r * cell + 4)
                self.create_rectangle(x, y, x + cell - 8, y + cell - 8, fill=TILE_COLORS.get(value, '#e9a23b'), outline='')
                if value:
                    self.create_text(x + (cell - 8) / 2, y + (cell - 8) / 2, text=str(value), fill=INK if value <= 16 or value >= 2048 else 'white', font=('Segoe UI', 24 if value < 10000 else 19, 'bold'))

class Workspace:

    def __init__(self, root, view='search', output_root=None):
        if view != 'search':
            raise ValueError('Training workspace has been archived')
        self.root, self.view = (root, view)
        self.output_root = Path(output_root) if output_root else PROJECT / 'runs'
        self.process, self.log_handle, self.output = (None, None, None)
        self.closing = self.stop_pending = False
        self.history = []
        self.control = dict(paused=False, step_token=0, delay_ms=120)
        self.inputs = []
        root.title('Alpha2048 · ' + 'Value model 对局测试')
        root.geometry('1120x820')
        root.minsize(980, 740)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=INK)
        style.configure('TButton', padding=(12, 7))
        style.configure('Accent.TButton', background=ACCENT, foreground='white')
        style.map('Accent.TButton', background=[('active', '#00766d'), ('disabled', '#a6c5c1')])
        style.configure('TProgressbar', troughcolor='#dce6ed', background=ACCENT)
        header = tk.Frame(root, bg=INK)
        header.pack(fill='x')
        title = 'Play / Value model 对局测试'
        tk.Label(header, text=title, fg='white', bg=INK, font=('Microsoft YaHei UI', 20, 'bold')).pack(anchor='w', padx=26, pady=(17, 3))
        subtitle = '模型评分 + expectimax 搜索 · 单步或自动播放'
        tk.Label(header, text=subtitle, fg='#bcccdc', bg=INK).pack(anchor='w', padx=26, pady=(0, 16))
        self.body = ttk.Frame(root, padding=(24, 14))
        self.body.pack(fill='both', expand=True)
        self.settings = ttk.Frame(self.body)
        self.settings.pack(fill='x')
        self.settings.columnconfigure(1, weight=1)
        self.build_search()
        self.status = tk.StringVar(value='准备就绪')
        ttk.Label(self.body, textvariable=self.status, wraplength=1010).pack(fill='x', pady=(10, 3))
        self.location = tk.StringVar(value='输出和错误日志会自动保存到 runs 文件夹。')
        ttk.Label(self.body, textvariable=self.location, foreground=MUTED, wraplength=1010).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_id = root.after(250, self.poll)
        root.report_callback_exception = self.callback_error

    def field(self, row, label, value, browse=None):
        variable = tk.StringVar(value=value)
        ttk.Label(self.settings, text=label).grid(row=row, column=0, sticky='w', padx=(0, 16), pady=5)
        entry = ttk.Entry(self.settings, textvariable=variable)
        entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.inputs.append(entry)
        if browse:
            button = ttk.Button(self.settings, text='选择…', command=lambda: browse(variable))
            button.grid(row=row, column=2, padx=(10, 0))
            self.inputs.append(button)
        return variable

    def buttons(self):
        row = ttk.Frame(self.body)
        row.pack(fill='x', pady=(12, 10))
        self.start_button = ttk.Button(row, text='开始新对局', style='Accent.TButton', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(row, text='停止并保存', state='disabled', command=self.stop)
        self.stop_button.pack(side='left', padx=8)
        ttk.Button(row, text='打开输出文件夹', command=self.open_output).pack(side='right')
        return row

    def cards(self, labels):
        row = ttk.Frame(self.body)
        row.pack(fill='x', pady=(0, 12))
        values = []
        for i, title in enumerate(labels):
            row.columnconfigure(i, weight=1)
            card = tk.Frame(row, bg=PANEL)
            card.grid(row=0, column=i, sticky='ew', padx=(0 if i == 0 else 10, 0))
            tk.Label(card, text=title, fg=MUTED, bg=PANEL).pack(anchor='w', padx=15, pady=(10, 0))
            var = tk.StringVar(value='—')
            tk.Label(card, textvariable=var, fg=INK, bg=PANEL, font=('Segoe UI', 19, 'bold')).pack(anchor='w', padx=15, pady=(2, 10))
            values.append(var)
        return values

    def build_search(self):
        latest = PROJECT / 'models' / 'latest'
        self.checkpoint = self.field(0, '模型文件', str(latest / 'best.pt'), lambda v: self.browse(v, False))
        options = ttk.Frame(self.settings)
        options.grid(row=1, column=0, columnspan=3, sticky='w', pady=7)
        self.seed, self.games = (tk.StringVar(value='9200000'), tk.StringVar(value='1'))
        for label, var in [('起始 seed', self.seed), ('对局数', self.games)]:
            ttk.Label(options, text=label).pack(side='left', padx=(0, 8))
            entry = ttk.Entry(options, textvariable=var, width=12)
            entry.pack(side='left', padx=(0, 20))
            self.inputs.append(entry)
        ttk.Label(options, text='播放间隔（毫秒）').pack(side='left')
        self.delay = tk.DoubleVar(value=120)
        ttk.Scale(options, from_=0, to=1000, variable=self.delay, command=self.speed_changed, length=180).pack(side='left', padx=10)
        self.depth = tk.StringVar(value='1')
        depth_row = ttk.Frame(self.settings)
        depth_row.grid(row=2, column=0, columnspan=3, sticky='w', pady=5)
        ttk.Label(depth_row, text='搜索深度').pack(side='left', padx=(0, 8))
        selector = ttk.Combobox(depth_row, textvariable=self.depth, values=('0', '1', '2', '3', '4'), width=5, state='readonly')
        selector.pack(side='left')
        self.inputs.append(selector)
        ttk.Label(depth_row, text='  0 = 直接预测；1 = 随机落子后再走一步，末端由模型评分。深度越大越慢。', foreground=MUTED).pack(side='left')
        row = self.buttons()
        self.pause_button = ttk.Button(row, text='暂停', command=self.toggle_pause, state='disabled')
        self.pause_button.pack(side='left')
        self.step_button = ttk.Button(row, text='单步', command=self.single_step)
        self.step_button.pack(side='left', padx=8)
        self.score_card, self.tile_card, self.step_card, self.game_card = self.cards(['得分', '最大方块', '步数', '对局'])
        content = tk.Frame(self.body, bg=PANEL)
        content.pack(fill='both', expand=True)
        self.board = Board(content)
        self.board.pack(side='left', fill='both', expand=True)
        aside = tk.Frame(content, bg=PANEL, width=270)
        aside.pack(side='right', fill='y', padx=24, pady=20)
        tk.Label(aside, text='上一步 · 候选动作评分', bg=PANEL, fg=INK, font=('Microsoft YaHei UI', 12, 'bold')).pack(anchor='w', pady=(0, 14))
        self.q_vars = []
        for title in ('↑ 上', '→ 右', '↓ 下', '← 左'):
            var = tk.StringVar(value=title + '    —')
            tk.Label(aside, textvariable=var, bg=PANEL, fg=INK, font=('Microsoft YaHei UI', 14), anchor='w').pack(fill='x', pady=8)
            self.q_vars.append(var)
        self.action_text = tk.StringVar(value='棋盘显示落子后的状态。')
        tk.Label(aside, textvariable=self.action_text, wraplength=250, justify='left', bg=PANEL, fg=ACCENT).pack(anchor='w', pady=18)
        tk.Label(aside, text='评分单位是结构评分，\n不代表未来游戏得分。', bg=PANEL, fg=MUTED, justify='left').pack(anchor='w')

    def browse(self, variable, directory):
        selected = filedialog.askdirectory(parent=self.root) if directory else filedialog.askopenfilename(parent=self.root, filetypes=[('模型', '*.pt')])
        if selected:
            variable.set(selected)

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def set_busy(self, busy):
        for widget in self.inputs + [self.start_button]:
            widget.configure(state='disabled' if busy else 'readonly' if isinstance(widget, ttk.Combobox) else 'normal')
        self.stop_button.configure(state='normal' if busy else 'disabled')
        self.pause_button.configure(state='normal' if busy else 'disabled')

    def start(self, paused=False):
        if self.alive():
            return
        try:
            prefix = 'value-games'
            output = self.output_root / (prefix + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            checkpoint, seed, games = (Path(self.checkpoint.get()), int(self.seed.get()), int(self.games.get()))
            if not checkpoint.is_file() or seed < 0 or games < 1:
                raise ValueError('请选择模型文件，seed ≥ 0，对局数 ≥ 1。')
            arguments = ['alpha2048.test_value_games', '--checkpoint', str(checkpoint), '--seed', str(seed), '--games', str(games), '--depth', self.depth.get(), '--delay-ms', str(int(self.delay.get()))]
            if paused:
                arguments.append('--paused')
            self.control = dict(paused=paused, step_token=1 if paused else 0, delay_ms=int(self.delay.get()))
            output.parent.mkdir(exist_ok=True)
            self.log_handle = output.with_suffix('.log').open('w', encoding='utf-8')
            python = PROJECT / '.venv/Scripts/python.exe'
            env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
            self.process = subprocess.Popen([str(python), '-u', '-m', *arguments, '--output', str(output)], cwd=PROJECT, stdout=self.log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=env)
            self.output, self.stop_pending = (output, False)
            self.history = []
            self.pause_button.configure(text='继续' if paused else '暂停')
            self.board.draw([[0] * 4 for _ in range(4)])
            self.location.set(str(output))
            self.status.set('正在启动，请稍候…')
            self.set_busy(True)
        except Exception as exc:
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None
            self.status.set('无法启动：' + str(exc))

    def stop(self):
        if self.alive():
            self.stop_pending = True
            self.status.set('正在停止并保存，请稍候…')

    def speed_changed(self, *_):
        if hasattr(self, 'delay'):
            self.control['delay_ms'] = int(self.delay.get())

    def toggle_pause(self):
        self.control['paused'] = not self.control['paused']
        self.pause_button.configure(text='继续' if self.control['paused'] else '暂停')

    def single_step(self):
        if not self.alive():
            self.start(paused=True)
        else:
            self.control['paused'] = True
            self.control['step_token'] += 1
            self.pause_button.configure(text='继续')

    def open_output(self):
        destination = self.output if self.output and self.output.exists() else PROJECT / 'runs'
        os.startfile(str(destination))

    def update_search(self, state):
        if 'board' not in state:
            return
        self.board.draw(state['board'])
        self.score_card.set(f"{state.get('score', 0):,}")
        self.tile_card.set(str(state.get('max_tile', 0)))
        self.step_card.set(str(state.get('step', 0)))
        self.game_card.set(f"{state.get('game', 1)} / {state.get('games', 1)}")
        for i, title in enumerate(('↑ 上', '→ 右', '↓ 下', '← 左')):
            value = state.get('q', [None] * 4)[i]
            text = '—' if value is None else f'{value:.4f}'
            selected = '  ✓' if state.get('action') == i else ''
            self.q_vars[i].set(f'{title}    {text}{selected}')
        if state.get('action') is not None:
            action = ('上', '右', '下', '左')[state['action']]
            self.action_text.set(f"上一步选择：{action}，合并得分 +{state.get('reward', 0)}\n搜索 {state.get('inference_ms', 0):.1f} ms · 深度 {state.get('depth', 0)}\n棋盘显示落子后的状态。")
        if state.get('status') == 'searching':
            self.status.set(f"正在搜索 · 深度 {state.get('depth', 1)} · 停止请求将在搜索批次边界保存。")
        if state.get('status') == 'running':
            self.status.set(f"{state.get('model_type', 'value model')} 自动对局进行中；评分对应上一步棋盘，棋盘显示落子后的状态。")

    def poll(self):
        if self.process is not None:
            if self.output.exists():
                if self.alive():
                    write_json(self.output / 'control.json', self.control)
                if self.stop_pending:
                    (self.output / 'stop.request').touch()
                state = read_json(self.output / 'status.json', {})
                self.update_search(state)
                names = {'loading': '正在加载数据或模型…', 'paused': '已暂停，可单步或继续。', 'stopped': '已停止，当前进度已保存。', 'complete': '已完成，结果已保存。', 'error': '运行失败：' + state.get('message', '')}
                if state.get('status') in names:
                    self.status.set(names[state['status']])
                if state.get('status') == 'complete' and state.get('results'):
                    results = state['results']
                    complete = [r for r in results if r['terminated']]
                    if complete:
                        mean = sum((r['score'] for r in complete)) / len(complete)
                        scores = '、'.join((str(r['score']) for r in complete[:20]))
                        self.status.set(f'已完成 {len(complete)} 局 · 平均得分 {mean:,.0f} · 各局得分：{scores}')
            if not self.alive():
                code = self.process.returncode
                state = read_json(self.output / 'status.json', {})
                if code or state.get('status') not in ('complete', 'stopped'):
                    self.status.set(f"运行未完成：{state.get('message', '请查看日志')} · 日志：{self.output.with_suffix('.log').name}")
                self.process = None
                if self.log_handle:
                    self.log_handle.close()
                    self.log_handle = None
                self.set_busy(False)
                if self.closing:
                    self.root.destroy()
                    return
        self.poll_id = self.root.after(300, self.poll)

    def close(self):
        if self.alive():
            self.closing = True
            self.stop()
            self.status.set('正在保存并关闭窗口…')
        else:
            self.root.after_cancel(self.poll_id)
            self.root.destroy()

    def callback_error(self, kind, value, tb):
        path = PROJECT / 'runs/desktop-errors.log'
        path.parent.mkdir(exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            traceback.print_exception(kind, value, tb, file=handle)
        self.status.set(f'界面错误：{value}。详情已保存到 {path.name}')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', choices=('search',), default='search')
    args = parser.parse_args(argv)
    root = tk.Tk()
    Workspace(root, args.view)
    root.mainloop()
if __name__ == '__main__':
    main()
