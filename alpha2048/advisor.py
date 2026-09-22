"""Non-modal AI controls; inference is lazy and never touches Tk from a worker."""
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import tkinter as tk
from threading import Event
import logging

import numpy as np
from .appearance import apply_typography, BG, INK, ACCENT
from .inference import AIError, Inference, recommendations, best_action, displayed_percentages

SPEEDS = {'instant': (0., 0.), 'fast': (.25, .5), 'normal': (.5, 1.)}
from .ui_text import WORDS



class AdvisorWindow:
    def __init__(self, app, provider=None):
        self.app = app
        if app.ai_executor is None:
            app.ai_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='2048-AI')
        if provider is None and app.ai_provider is None:
            app.ai_provider = Inference()
        self.provider = provider if provider is not None else app.ai_provider
        self.window = tk.Toplevel(app.root)
        self.window.configure(bg=BG)
        self.window.geometry('520x680')
        self.window.minsize(500, 650)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.closed = self.auto = self.want_step = self.inflight = False
        self.results = Queue()
        self.cached_key = self.failed_key = None
        self.scores = None
        self.seconds = self.ready_at = self.requested_at = 0
        self.job = None
        self.generation = 0
        self.error = None
        self.depth = tk.IntVar(value=1)
        self.device = tk.StringVar(value='cpu')
        self.speed = tk.StringVar(value='normal')
        self.app.set_animation_scale(1.)
        body = tk.Frame(self.window, bg=BG, padx=22, pady=20)
        body.pack(fill='both', expand=True)
        self.labels, self.choices = {}, {}
        for name, variable, values in [('depth', self.depth, (0, 1, 2)),
                                       ('speed', self.speed, ('instant', 'fast', 'normal')),
                                       ('device', self.device, ('cpu', 'cuda'))]:
            row = tk.Frame(body, bg=BG)
            row.pack(fill='x', pady=(0, 12))
            label = tk.Label(row, bg=BG, fg=INK, width=9, anchor='w', font=('Microsoft YaHei UI', 10, 'bold'))
            label.pack(side='left')
            self.labels[name] = label
            self.choices[name] = []
            for value in values:
                button = tk.Button(row, relief='flat', bd=0, padx=10, pady=8,
                                   font=('Microsoft YaHei UI', 10), cursor='hand2',
                                   command=lambda n=name, v=value: self.choose(n, v))
                button.pack(side='left', padx=(0, 6))
                self.choices[name].append((button, value))
        cards = tk.Frame(body, bg=BG)
        cards.pack(fill='both', expand=True, pady=(4, 16))
        self.cards = []
        positions = ((0, 1), (1, 2), (2, 1), (1, 0))
        for index in range(3):
            cards.columnconfigure(index, weight=1, uniform='cards')
            cards.rowconfigure(index, weight=1, uniform='cards')
        for i in range(4):
            row, col = positions[i]
            card = tk.Frame(cards, bg='#eee4da', padx=8, pady=8)
            card.grid(row=row, column=col, sticky='nsew', padx=4, pady=4)
            direction = tk.Label(card, bg='#eee4da', fg=INK, font=('Microsoft YaHei UI', 11))
            direction.pack()
            weight = tk.Label(card, bg='#eee4da', fg=INK, font=('Segoe UI', 21, 'bold'))
            weight.pack(pady=(3, 0))
            self.cards.append((card, direction, weight))
        buttons = tk.Frame(body, bg=BG)
        buttons.pack(fill='x')
        self.buttons = {}
        for i, (name, command) in enumerate([('analyze', self.request_advice), ('step', self.single_step),
                                            ('auto', self.start_auto), ('stop', self.pause)]):
            buttons.columnconfigure(i, weight=1, uniform='buttons')
            button = tk.Button(buttons, command=command, bg=ACCENT, fg='white',
                               activebackground='#776452', activeforeground='white', disabledforeground='#cfc5ba',
                               relief='flat', bd=0, pady=9, cursor='hand2', font=('Microsoft YaHei UI', 10))
            button.grid(row=0, column=i, sticky='ew', padx=(0, 5) if i < 3 else (0, 0))
            self.buttons[name] = button
        self.status = tk.StringVar()
        self.status_label = tk.Label(body, textvariable=self.status, bg=BG, fg=INK,
                                     anchor='w', justify='left', wraplength=430, font=('Microsoft YaHei UI', 9))
        self.status_label.pack(fill='x', pady=(12, 0))
        self.translate()
        self.invalidate()
        self.poll()

    def key(self):
        return (self.generation, self.app.revision, self.app.game.board.tobytes(), self.device.get(), self.depth.get())

    def choose(self, name, value):
        getattr(self, name).set(value)
        if name == 'speed':
            self.app.set_animation_scale(SPEEDS[value][1])
        else:
            self.settings_changed()
        self.style_choices()

    def style_choices(self):
        text = WORDS[self.app.language]
        for name, choices in self.choices.items():
            for button, value in choices:
                selected = getattr(self, name).get() == value
                label = str(value) if name == 'depth' else (('GPU' if value == 'cuda' else 'CPU') if name == 'device' else text[value])
                button.configure(text=('✓ ' if selected else '   ') + label,
                                 bg=ACCENT if selected else '#ede0c8', fg='white' if selected else INK,
                                 activebackground='#bbada0', activeforeground='white')

    def translate(self):
        text = WORDS[self.app.language]
        self.window.title(text['title'])
        for name, label in self.labels.items():
            label.configure(text=text[name])
        for name, button in self.buttons.items():
            button.configure(text=text[name])
        self.style_choices()
        self.render()
        if self.error:
            self.show_error()
        else:
            self.status.set(text['loading'] if self.inflight or self.scores is None else text['ready'])
        apply_typography(self.window, self.app.language)

    def render(self):
        text = WORDS[self.app.language]
        scores = self.scores if self.cached_key == self.key() else None
        probabilities = recommendations(scores) if scores is not None else None
        best = best_action(scores) if scores is not None else None
        displayed = displayed_percentages(scores) if probabilities is not None else None
        for i, (card, direction, weight) in enumerate(self.cards):
            color = '#edcf72' if i == best else '#eee4da'
            for widget in (card, direction, weight):
                widget.configure(bg=color)
            direction.configure(text=text['directions'][i] + (' ✓' if i == best else ''))
            weight.configure(text=f'{displayed[i] / 10:.1f}%' if scores is not None and np.isfinite(scores[i]) else '—')

    def invalidate(self):
        self.generation += 1
        if hasattr(self, 'work_cancelled'):
            self.work_cancelled.set()
        if hasattr(self.provider, 'cancelled'):
            self.provider.cancelled.set()
        self.cached_key = self.failed_key = None
        self.error = None
        self.scores = None
        self.want_step = False
        self.render()
        self.status_label.configure(fg=INK)
        self.status.set(WORDS[self.app.language]['loading'])

    def settings_changed(self, _=None):
        self.pause()
        self.invalidate()

    def request_advice(self):
        self.pause()
        self.invalidate()

    def single_step(self):
        self.auto = False
        self.want_step = True
        self.failed_key = None
        self.requested_at = perf_counter()

    def start_auto(self):
        self.auto = True
        self.want_step = False
        self.failed_key = None
        self.requested_at = perf_counter()

    def pause(self):
        self.auto = False
        self.want_step = False
        if not self.closed:
            self.status.set(WORDS[self.app.language]['stopped'])

    def show_error(self):
        code, detail = self.error
        message = WORDS[self.app.language].get(code, WORDS[self.app.language]['inference_error'])
        if code == 'inference_error' and detail:
            message += ' ' + detail
        self.status_label.configure(fg='#b34f36')
        self.status.set(message)

    def submit(self, key):
        board = self.app.game.board
        device, depth = self.device.get(), self.depth.get()
        cancelled = self.work_cancelled = Event()
        provider, results = self.provider, self.results
        self.inflight = True
        self.status.set(WORDS[self.app.language]['loading'])
        def worker():
            try:
                if cancelled.is_set():
                    results.put((key, None, 0, ('cancelled', ''), perf_counter()))
                    return
                if hasattr(provider, 'cancelled'):
                    provider.cancelled.clear()
                    if cancelled.is_set():
                        provider.cancelled.set()
                scores, seconds = provider(board, device, depth)
                recommendations(scores)
                results.put((key, np.asarray(scores).copy(), seconds, None, perf_counter()))
            except Exception as exc:
                if not cancelled.is_set():
                    logging.exception('Game AI inference failed')
                results.put((key, None, 0, (getattr(exc, 'code', 'inference_error'), str(exc)), perf_counter()))
        self.app.ai_executor.submit(worker)

    def poll(self):
        if self.closed:
            return
        self.job = None
        try:
            while True:
                key, scores, seconds, error, completed_at = self.results.get_nowait()
                self.inflight = False
                if key != self.key():
                    continue
                if error:
                    self.pause()
                    self.failed_key = key
                    self.error = error
                    self.show_error()
                else:
                    self.error = None
                    self.cached_key, self.scores, self.seconds = key, scores, seconds
                    self.ready_at = completed_at
                    self.status_label.configure(fg=INK)
                    self.status.set(f"{WORDS[self.app.language]['ready']} · {seconds * 1000:.1f} ms")
                    self.render()
        except Empty:
            pass
        idle = self.app.animation is None and self.app.help_window is None
        if self.app.game.terminated:
            self.pause()
            self.status.set(WORDS[self.app.language]['terminal'])
            self.cached_key = self.key()
            self.scores = np.full(4, -np.inf)
            self.render()
        elif idle:
            key = self.key()
            if self.cached_key != key and self.failed_key != key and not self.inflight:
                self.submit(key)
            due = max(self.ready_at, self.requested_at) + SPEEDS[self.speed.get()][0]
            if self.cached_key == key and (self.want_step or self.auto) and perf_counter() >= due:
                action = best_action(self.scores)
                self.want_step = False
                if action is not None:
                    self.app.move(action, from_ai=True)
        self.buttons['auto'].configure(state='disabled' if self.auto else 'normal')
        self.buttons['stop'].configure(state='normal' if self.auto or self.want_step else 'disabled')
        self.job = self.app.root.after(10 if self.speed.get() == 'instant' else 16, self.poll)

    def close(self):
        if self.closed:
            return
        self.pause()
        self.closed = True
        if hasattr(self, 'work_cancelled'):
            self.work_cancelled.set()
        if hasattr(self.provider, 'cancelled'):
            self.provider.cancelled.set()
        if self.job is not None:
            self.app.root.after_cancel(self.job)
        self.window.destroy()
        self.app.advisor = None
