from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

root = Path(SPECPATH).parent
stage = Path(os.environ['ALPHA2048_STAGE'])
datas = [(str(stage/'models/latest/best.pt'),'models/latest'),
         (str(root/'assets/ocr/en_PP-OCRv4_rec.onnx'),'assets/ocr'),
         (str(root/'assets/ocr/dict.txt'),'assets/ocr')]
a = Analysis([str(root/'packaging/launcher.py')],pathex=[str(root)],
             binaries=collect_dynamic_libs('onnxruntime'),datas=datas,
             hiddenimports=[],hookspath=[],runtime_hooks=[],
             excludes=['alpha2048.desktop','alpha2048.test_value_games','IPython','pytest',
                       'matplotlib','pandas','scipy','tensorflow','torchvision','torchaudio'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,[],exclude_binaries=True,name='2048',debug=False,
          bootloader_ignore_signals=False,strip=False,upx=False,console=False,
          disable_windowed_traceback=False)
coll = COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='Alpha2048-CPU')
