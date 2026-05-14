# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['src\\gui.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('tesseract/tesseract.exe', 'tesseract'),
        ('tesseract/tessdata',      'tesseract/tessdata'),
    ],
    hiddenimports=[
        'pyperclip',
        'pystray',
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        'pynput._util',
        'pynput._util.win32',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'sklearn',
        'scipy',
        'numpy',
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScreenshotOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['tesseract.exe'],
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['tesseract.exe'],
    name='ScreenshotOCR',
)
