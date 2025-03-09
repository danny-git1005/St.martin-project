# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['weight_detect.py'],
    pathex=[],
    binaries=[],
    datas=[('onnx_model', 'onnx_model'), 
            ('tool', 'tool'), 
            ('C:/Users/user/Desktop/St.martin/St.martin_venv/Lib/site-packages/onnxruntime/capi', 'onnxruntime/capi')],
    hiddenimports=['requests'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='weight_detect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
