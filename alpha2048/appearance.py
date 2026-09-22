"""Shared desktop typography with local-font fallbacks."""
import tkinter as tk
from tkinter import font


def rounded_family(root):
    available = set(font.families(root))
    return next((name for name in ('Arial Rounded MT Bold', 'Segoe UI', 'Arial')
                 if name in available), 'TkDefaultFont')


def apply_typography(root, language='en'):
    family = 'Microsoft YaHei UI' if language == 'zh' else rounded_family(root)

    def visit(widget):
        if isinstance(widget, (tk.Label, tk.Button)):
            current = font.Font(root=root, font=widget.cget('font'))
            widget.configure(font=(family, current.actual('size'), 'bold'))
        for child in widget.winfo_children():
            visit(child)

    visit(root)


BG = '#faf8ef'
INK = '#776e65'
BOARD = '#bbada0'
COLORS = {0: '#cdc1b4', 2: '#eee4da', 4: '#ede0c8', 8: '#f2b179',
          16: '#f59563', 32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72',
          256: '#edcc61', 512: '#edc850', 1024: '#edc53f', 2048: '#edc22e'}
ACCENT = '#8f7a66'
