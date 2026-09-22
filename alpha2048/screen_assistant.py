"""Read-only screen advisor. Screen capture, OCR and search share one worker."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from time import perf_counter
import ctypes
import logging
import sys
import tkinter as tk
from PIL import ImageGrab, ImageTk
import numpy as np

from .ui_text import WORDS as AI_WORDS
from .inference import Inference, AIError, recommendations, best_action, displayed_percentages
from .appearance import apply_typography, rounded_family, BG, INK, BOARD, COLORS, ACCENT
from .screen_ocr import BoardReader, RecognitionError, fingerprint, same_frame

WORDS = {
 'en': dict(title='2048 · Screen assistant', select='Select board', refresh='Refresh', watch='Live', stop='Stop',
            depth='Depth', device='Device', ready='Select the outer edge of the 4 × 4 board.',
            selecting='Drag around the board. Esc cancels.', waiting='Waiting for a stable board…',
            reading='Reading and thinking…', paused='Paused', done='Recommendation ready', monitoring='Live · Monitoring board',
            uncertain='Board unclear. Check the selection or wait for the animation to finish.',
            error='Unable to capture or analyze. Please retry.', overlap='Move this assistant away from the selected board.',
            ocr_missing='OCR model files are missing. Reinstall the application.',
            ocr_invalid='OCR model files are damaged or incompatible. Reinstall the application.',
            region='Selected: {w} × {h}', over='Game over', help='Select the whole board, including its outer border. Keep it visible and uncovered.\n\nLive refreshes after the board settles. This assistant only recommends moves.\n\nReselect after moving the browser, scrolling, or changing zoom. Esc cancels selection.'),
 'zh': dict(title='2048 · 屏幕助手', select='框选棋盘', refresh='刷新', watch='实时推荐', stop='停止',
            depth='深度', device='计算设备', ready='沿 4 × 4 棋盘的外边缘框选。',
            selecting='拖动框选棋盘，Esc 取消。', waiting='等待棋盘稳定…', reading='正在识别和思考…',
            paused='已暂停', done='推荐已更新', monitoring='实时推荐中 · 持续监测棋盘', uncertain='棋盘不清晰，请检查框选范围或等待动画结束。',
            error='截图或分析失败，请重试。', overlap='请将助手窗口移到选定棋盘之外。',
            ocr_missing='数字识别模型文件缺失，请重新安装程序。',
            ocr_invalid='数字识别模型文件损坏或不兼容，请重新安装程序。',
            region='已框选：{w} × {h}', over='游戏结束', help='框选整个棋盘，包含外边框。棋盘需保持可见且无遮挡。\n\n实时推荐会在棋盘稳定后刷新，仅提供建议，不会操作游戏。\n\n移动浏览器、滚动或改变缩放后请重新框选。Esc 取消框选。')}


def desktop_bounds():
    if sys.platform == 'win32':
        user = ctypes.windll.user32
        return tuple(user.GetSystemMetrics(i) for i in (76,77,78,79))
    image = ImageGrab.grab()
    return 0,0,*image.size


def overlaps(a,b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def absolute_geometry(x,y,width,height):
    # A bare '-267' is a Tk bottom/right offset. '+-267' is an absolute
    # negative origin, needed for monitors above or left of the primary one.
    return f'{width}x{height}+{x}+{y}'


class Selector:
    def __init__(self, root, callback, text):
        self.callback = callback
        self.origin = None
        self.x,self.y,w,h = desktop_bounds()
        screenshot = ImageGrab.grab(all_screens=True)
        self.window = tk.Toplevel(root)
        try:
            self.build(screenshot, text, w, h)
        except Exception:
            self.window.destroy()
            raise

    def build(self, screenshot, text, w, h):
        self.window.overrideredirect(True)
        self.window.attributes('-topmost',True)
        self.window.geometry(absolute_geometry(self.x,self.y,w,h))
        self.canvas = tk.Canvas(self.window, highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both',expand=True)
        self.photo = ImageTk.PhotoImage(screenshot)
        self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.canvas.create_rectangle(0,0,w,44,fill=ACCENT,outline='')
        self.canvas.create_text(w/2,22,text=text,fill='white',font=('Segoe UI',12,'bold'))
        self.rectangle = None
        self.canvas.bind('<ButtonPress-1>', self.start)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.finish)
        self.window.bind('<Escape>',lambda event:self.close(None))
        self.window.protocol('WM_DELETE_WINDOW',lambda:self.close(None))
        self.window.grab_set()
        self.window.focus_force()

    def start(self,event):
        self.origin = event.x,event.y
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(event.x,event.y,event.x,event.y,outline='#edc22e',width=4)

    def drag(self,event):
        if self.origin:
            self.canvas.coords(self.rectangle,*self.origin,event.x,event.y)

    def finish(self,event):
        if self.origin is None:
            return
        x0,y0 = self.origin
        box = (min(x0,event.x)+self.x,min(y0,event.y)+self.y,max(x0,event.x)+self.x,max(y0,event.y)+self.y)
        if min(box[2]-box[0],box[3]-box[1]) >= 160 and abs((box[2]-box[0])/(box[3]-box[1])-1) <= .12:
            self.close(box)

    def close(self,box):
        self.window.grab_release()
        self.window.destroy()
        self.callback(box)


class ScreenEngine:
    def __init__(self, capture=None, reader=None, provider=None):
        self.capture = capture or (lambda box:ImageGrab.grab(bbox=box,all_screens=True))
        self.reader = reader
        self.provider = provider or Inference()
        self.cancelled = self.provider.cancelled
        self.cache = None

    def analyze(self, box, device, depth):
        if self.cancelled.is_set():
            return 'cancelled',None,None
        image = self.capture(box)
        first = fingerprint(image)
        # Bounded, cancellable settling interval; never hold the Tk event loop.
        if self.cancelled.wait(.18):
            return 'cancelled',None,None
        image = self.capture(box)
        if not same_frame(first,fingerprint(image)):
            return 'waiting',None,None
        if self.reader is None:
            self.reader = BoardReader()
        board = self.reader.read(image)
        key = (board.tobytes(),device,depth)
        if self.cancelled.is_set():
            return 'cancelled',None,None
        if self.cache and self.cache[0] == key:
            scores = self.cache[1]
        else:
            scores,_ = self.provider(board,device,depth)
            recommendations(scores)
            self.cache = key,scores
        # Do not publish a recommendation for a board that moved during search.
        if self.cancelled.is_set():
            return 'cancelled',None,None
        if not same_frame(fingerprint(image),fingerprint(self.capture(box))):
            return 'waiting',None,None
        return 'done',board,scores


class ScreenAssistant:
    def __init__(self, root, engine=None):
        self.root = root
        self.language = 'en'
        self.engine = engine or ScreenEngine()
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='2048-screen')
        self.results = Queue()
        self.closed = self.inflight = self.live = False
        self.generation = 0
        self.box = None
        self.selector = None
        self.selection_job = None
        self.board = self.scores = None
        self.state = 'ready'
        self.error = None
        self.depth = tk.IntVar(value=1)
        self.device = tk.StringVar(value='cpu')
        self.next_run = 0
        self.pending = False
        self.font_family = rounded_family(root)
        self.styled_language = None
        available_height = max(600, root.winfo_screenheight()-100)
        self.compact = available_height < 800
        self.preview_side = 128 if self.compact else 232
        root.configure(bg=BG)
        root.geometry(f'550x{min(850,available_height)}')
        root.minsize(520,min(700,available_height))
        root.protocol('WM_DELETE_WINDOW',self.close)
        body = tk.Frame(root,bg=BG,padx=20,pady=16)
        body.pack(fill='both',expand=True)
        toolbar = tk.Frame(body,bg=BG)
        toolbar.pack(fill='x')
        self.select_button = self.button(toolbar,self.select)
        self.select_button.pack(side='left',fill='x',expand=True)
        self.language_button = self.button(toolbar,self.toggle_language,'中')
        self.language_button.pack(side='right',padx=(6,0))
        self.button(toolbar,self.help,'?').pack(side='right',padx=(6,0))
        self.region = tk.Label(body,bg=BG,fg=INK,font=('Segoe UI',9),anchor='w')
        self.region.pack(fill='x',pady=(10,6))
        self.preview = tk.Canvas(body,width=self.preview_side,height=self.preview_side,bg=BG,highlightthickness=0)
        self.preview.pack()
        self.choices = {}
        self.labels = {}
        for name,values in [('depth',(0,1,2)),('device',('cpu','cuda'))]:
            row = tk.Frame(body,bg=BG)
            row.pack(fill='x',pady=(10,0))
            label = tk.Label(row,width=10,anchor='w',bg=BG,fg=INK,font=('Segoe UI',10))
            label.pack(side='left')
            self.labels[name] = label
            self.choices[name] = []
            for value in values:
                button = self.button(row,lambda n=name,v=value:self.choose(n,v))
                button.pack(side='left',padx=(0,6))
                self.choices[name].append((button,value))
        cards = tk.Frame(body,bg=BG)
        cards.pack(fill='both',expand=True,pady=12)
        self.cards = []
        for i in range(3):
            cards.columnconfigure(i,weight=1,uniform='cards')
            cards.rowconfigure(i,weight=1,uniform='cards')
        for row,col in ((0,1),(1,2),(2,1),(1,0)):
            card = tk.Frame(cards,bg='#eee4da')
            card.grid(row=row,column=col,sticky='nsew',padx=3,pady=3)
            title = tk.Label(card,bg='#eee4da',fg=INK,font=('Segoe UI',9 if self.compact else 10))
            title.pack(pady=(6,0))
            value = tk.Label(card,bg='#eee4da',fg=INK,font=('Segoe UI',13 if self.compact else 17))
            value.pack(pady=(0,6))
            self.cards.append((card,title,value))
        row = tk.Frame(body,bg=BG)
        row.pack(fill='x')
        self.buttons = {}
        for i,(name,command) in enumerate((('refresh',self.refresh),('watch',self.watch),('stop',self.stop))):
            row.columnconfigure(i,weight=1,uniform='actions')
            button = self.button(row,command)
            button.grid(row=0,column=i,sticky='ew',padx=3)
            self.buttons[name] = button
        self.status = tk.Label(body,bg=BG,fg=INK,anchor='w',justify='left',wraplength=465,font=('Segoe UI',8 if self.compact else 9),height=3)
        self.status.pack(fill='x',pady=(10,0))
        self.render()
        self.job = root.after(50,self.poll)

    def button(self,parent,command,text=''):
        return tk.Button(parent,text=text,command=command,bg=ACCENT,fg='white',relief='flat',bd=0,
                         padx=12,pady=6 if self.compact else 8,cursor='hand2',font=('Segoe UI',9 if self.compact else 10,'bold'))

    def invalidate(self):
        self.generation += 1
        self.engine.cancelled.set()
        self.board = self.scores = None
        self.error = None

    def select(self):
        if self.selector is not None or self.selection_job is not None:
            return
        self.stop()
        self.root.withdraw()
        self.selection_job = self.root.after(160,self.open_selector)

    def open_selector(self):
        self.selection_job = None
        if self.closed:
            return
        try:
            self.selector = Selector(self.root,self.selected,WORDS[self.language]['selecting'])
        except Exception as exc:
            logging.exception('Screen selection failed')
            self.root.deiconify()
            self.state,self.error = 'error',str(exc)
            self.render()

    def selected(self,box):
        self.selector = None
        if self.closed:
            return
        self.root.deiconify()
        if box is not None:
            self.box = box
            self.watch()

    def choose(self,name,value):
        getattr(self,name).set(value)
        self.invalidate()
        self.pending = self.box is not None
        self.state = 'reading' if self.box else 'ready'
        self.render()

    def refresh(self):
        self.invalidate()
        self.pending = self.box is not None
        self.state = 'reading' if self.box else 'ready'
        self.render()

    def watch(self):
        self.live = True
        self.refresh()

    def stop(self):
        self.live = self.pending = False
        self.invalidate()
        self.state = 'paused'
        self.render()

    def toggle_language(self):
        self.language = 'zh' if self.language=='en' else 'en'
        self.render()

    def help(self):
        from tkinter import messagebox
        self.stop()
        messagebox.showinfo(WORDS[self.language]['title'],WORDS[self.language]['help'],parent=self.root)

    def submit(self):
        bounds = (self.root.winfo_rootx(),self.root.winfo_rooty(),
                  self.root.winfo_rootx()+self.root.winfo_width(),self.root.winfo_rooty()+self.root.winfo_height())
        if overlaps(bounds,self.box):
            self.board = self.scores = None
            self.state = 'overlap'
            self.next_run = perf_counter()+.5
            self.render()
            return
        self.pending = False
        self.inflight = True
        generation,box,device,depth = self.generation,self.box,self.device.get(),self.depth.get()
        self.engine.cancelled.clear()
        engine,results = self.engine,self.results
        def work():
            try:
                result = engine.analyze(box,device,depth)
                results.put((generation,result,None))
            except RecognitionError as exc:
                results.put((generation,('uncertain',None,None),str(exc)))
            except AIError as exc:
                logging.exception('Screen assistant inference failed')
                results.put((generation,(exc.code,None,None),str(exc)))
            except Exception as exc:
                if not engine.cancelled.is_set():
                    logging.exception('Screen assistant failed')
                results.put((generation,('error',None,None),str(exc)))
        self.executor.submit(work)

    def poll(self):
        if self.closed:
            return
        try:
            generation,result,error = self.results.get_nowait()
            self.inflight = False
            if generation == self.generation:
                self.state,self.board,self.scores = result
                self.error = error
                if self.state not in ('done','waiting','uncertain','cancelled'):
                    self.live = False
                self.next_run = perf_counter()+.25
                self.render()
        except Empty:
            pass
        if self.box and not self.inflight and (self.pending or self.live) and perf_counter() >= self.next_run:
            self.submit()
        self.job = self.root.after(50,self.poll)

    def render(self):
        text = WORDS[self.language]
        self.root.title(text['title'])
        self.select_button.configure(text=text['select'])
        self.language_button.configure(text='中' if self.language=='en' else 'EN')
        self.region.configure(text=text['region'].format(w=self.box[2]-self.box[0],h=self.box[3]-self.box[1]) if self.box else text['ready'])
        for name,choices in self.choices.items():
            self.labels[name].configure(text=text[name])
            for button,value in choices:
                selected = getattr(self,name).get()==value
                label = str(value) if name=='depth' else ('CPU' if value=='cpu' else 'GPU')
                button.configure(text=('✓ ' if selected else '  ')+label,bg=ACCENT if selected else '#ede0c8',fg='white' if selected else INK)
        for name,button in self.buttons.items():
            button.configure(text=('✓ ' if name=='watch' and self.live else '')+text[name],
                             state='normal' if self.box or name=='stop' else 'disabled')
        self.preview.delete('all')
        side = self.preview_side
        cell = (side-4)/4
        self.preview.create_rectangle(0,0,side,side,fill=BOARD,outline='')
        for r in range(4):
            for c in range(4):
                value = int(self.board[r,c]) if self.board is not None else None
                x,y = c*cell+4,r*cell+4
                self.preview.create_rectangle(x,y,x+cell-4,y+cell-4,fill=COLORS.get(value,'#cdc1b4'),outline='')
                font_size = (8 if value and value>=1024 else 10) if self.compact else 12
                self.preview.create_text(x+(cell-4)/2,y+(cell-4)/2,text='?' if value is None else (str(value) if value else ''),
                                         fill=INK if value is None or value<8 else 'white',font=(self.font_family,font_size,'bold'))
        probabilities = recommendations(self.scores) if self.scores is not None else None
        displayed = displayed_percentages(self.scores) if self.scores is not None else None
        best = best_action(self.scores) if self.scores is not None else None
        for i,(card,title,value) in enumerate(self.cards):
            color = '#edcf72' if i==best else '#eee4da'
            card.configure(bg=color)
            title.configure(text=AI_WORDS[self.language]['directions'][i],bg=color)
            value.configure(text=f'{displayed[i]/10:.1f}%' if probabilities is not None and np.isfinite(self.scores[i]) else '—',bg=color)
        state = 'over' if self.state=='done' and best is None else self.state
        message = text.get(state,AI_WORDS[self.language].get(state,text['error']))
        if self.live and state == 'done':
            message = text['monitoring']
        self.status.configure(text=message,fg='#c45d42' if self.error or state=='overlap' else INK)
        if self.styled_language != self.language:
            apply_typography(self.root,self.language)
            self.styled_language = self.language

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.engine.cancelled.set()
        if self.selection_job is not None:
            self.root.after_cancel(self.selection_job)
        if self.selector is not None:
            self.selector.close(None)
        self.root.after_cancel(self.job)
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.root.destroy()


def main():
    if sys.platform == 'win32':
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (OSError,AttributeError):
            pass
    root = tk.Tk()
    from .runtime import install_error_handler
    install_error_handler(root, 'screen-assistant')
    ScreenAssistant(root)
    root.mainloop()


if __name__ == '__main__':
    main()
