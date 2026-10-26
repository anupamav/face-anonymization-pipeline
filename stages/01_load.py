"""Load one RGB-depth pair from the project images folder or a ZIP."""
import io, zipfile, json
from pathlib import Path
import numpy as np, cv2, matplotlib.pyplot as plt, pandas as pd
import torch

rgb_name = RGB_FILENAME or f"rgb_{PAIR_ID}.npy"
depth_name = DEPTH_FILENAME or f"depth_{PAIR_ID}.npy"
def load_image_bytes(data, name, is_depth=False):
    if name.lower().endswith('.npy'):
        return np.load(io.BytesIO(data), allow_pickle=False)
    decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR)
    if decoded is None:raise ValueError(f'Cannot decode {name}')
    return decoded if is_depth else cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
if INPUT_ZIP:
    with zipfile.ZipFile(IMAGE_DIR / INPUT_ZIP) as archive:
        def zipped(name,is_depth=False):
            names=[v for v in archive.namelist() if v.rsplit('/',1)[-1]==name and not v.startswith('__MACOSX/')]
            if len(names)!=1:raise ValueError(f'Expected one {name} in ZIP, found {len(names)}')
            return load_image_bytes(archive.read(names[0]),name,is_depth)
        rgb=zipped(rgb_name);raw=zipped(depth_name,True)
else:
    rgb=load_image_bytes((IMAGE_DIR/rgb_name).read_bytes(),rgb_name)
    raw=load_image_bytes((IMAGE_DIR/depth_name).read_bytes(),depth_name,True)
assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
assert raw.ndim==2 and raw.shape==rgb.shape[:2], 'Need depth registered to RGB, same pixel grid'
if CALIBRATION_SHAPE is not None:
    assert raw.shape==tuple(CALIBRATION_SHAPE), 'Image dimensions differ from supplied calibration'
assert all(np.isfinite(v) for v in [FX,FY,CX,CY]) and FX>0 and FY>0
bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
height_px,width_px=raw.shape
K=np.array([[FX,0,CX],[0,FY,CY],[0,0,1]],float)
OUT=OUTPUT_DIR/f'pair_{PAIR_ID:02d}';OUT.mkdir(parents=True,exist_ok=True)
depth=raw.astype(float)*DEPTH_SCALE
valid=np.isfinite(depth)&(depth>=MIN_DEPTH_M)&(depth<=MAX_DEPTH_M)
assert valid.any(), 'No valid metric depth'
depth[~valid]=np.nan
print(f'Pair {PAIR_ID}: RGB {rgb.shape}; depth {depth.shape}; missing {100*(~valid).mean():.2f}%')
print('Intrinsics:',K)
