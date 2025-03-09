### Some tips in using Onnx

1. pyinstaller 需要打包 Onnxruntime 相關包時，datas的部分需要將 onnxruntime/capi 包進去
```
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
    optimize=0,
)
```

2. 執行打包完的exe遇到的問題
```
Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!
```

solution: 再將 cudnn 安裝包裡的LIB 放到 CUDA 裡時，要放cudnn_ops_infer64_8.dl

3. 版本
```
driver : 528.24
CUDA : 11.8
Cudnn : 8.9.7
onnxruntime-gpu : 1.19.2
```
