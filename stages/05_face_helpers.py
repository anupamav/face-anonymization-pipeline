# 6. Reference face helper snapshot — original coordinate/intersection and pose fallback logic.
"""Face anonymization with face-coordinate and pose-based fallback."""
import ast
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

POSE_THRESHOLD = 0.90
PERSON_CONF_THRESHOLD = 0.75
POSE_COMPLETENESS_THRESHOLD = 0.75
KEYPOINT_VALID_THRESHOLD = 0.50
INTERSECTION_THRESHOLD = 0.75
POSE_FACE_SCALE = 1.0  # Use original pose-derived face region without additional scaling


def parse_bbox(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none'):
        return None
    # Supports existing CSV entries such as [np.int64(120), np.int64(28), ...].
    text = re.sub(r'np\.(?:int\d+|float\d+)\(\s*([-+]?\d+(?:\.\d+)?)\s*\)', r'\1', text)
    try:
        box = ast.literal_eval(text)
        if len(box) != 4:
            return None
        box = [float(v) for v in box]
        if not np.all(np.isfinite(box)):
            return None
        return box
    except (ValueError, SyntaxError, TypeError):
        return None


def resolve(root, path):
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    path = str(path).strip()
    if not path or path.lower() == 'nan':
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def box_mask(box, h, w):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((int(np.clip(round(x1), 0, w)), int(np.clip(round(x2), 0, w))))
    y1, y2 = sorted((int(np.clip(round(y1), 0, h)), int(np.clip(round(y2), 0, h))))
    mask = np.zeros((h, w), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def intersection_logic(box, child_mask, h, w):
    face = box_mask(box, h, w)
    if not face.any():
        return None, 'MANUAL_INVALID_FACE_BOX', np.nan
    if child_mask is None:
        return face, 'FULL_FACE_BOX_NO_SEGMENTATION', np.nan
    intersect = face & child_mask
    ratio = float(intersect.sum() / face.sum())
    if ratio < INTERSECTION_THRESHOLD:
        return face, 'FULL_FACE_BOX_LOW_INTERSECTION', ratio
    return intersect, 'FACE_BOX_INTERSECTED_WITH_CHILD_MASK', ratio


def make_pose_model(weights='yolo11n-pose.pt'):
    from ultralytics import YOLO
    return YOLO(weights)


def pose_face_mask(rgb, child_mask, pose_model):
    """Build a face mask from a tilted eye line and the nose landmark.

    With a child segmentation mask, clip the face band to the child.
    Without segmentation, create a wider polygon around the landmarks.
    Pose output is provisional and always requires visual review.
    """
    h, w = rgb.shape[:2]
    pose_input = rgb.copy()
    if child_mask is not None:
        pose_input[~child_mask] = 0
    result = pose_model.predict(source=pose_input, conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None, 'POSE_NOT_DETECTED', pose_input

    boxes = result.boxes.xyxy.cpu().numpy()
    detections = result.boxes.conf.cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf
    if conf is None or len(xy) != len(boxes) or xy.shape[1] < 17:
        return None, 'POSE_KEYPOINTS_UNAVAILABLE', pose_input
    conf = conf.cpu().numpy()
    if conf.shape != (len(boxes), 17):
        return None, 'POSE_KEYPOINTS_UNAVAILABLE', pose_input

    completeness = (conf >= KEYPOINT_VALID_THRESHOLD).mean(axis=1)
    eligible = np.flatnonzero((detections >= PERSON_CONF_THRESHOLD) &
                              (completeness >= POSE_COMPLETENESS_THRESHOLD))
    if len(eligible) == 0:
        return None, 'NO_PERSON_WITH_CONFIDENT_COMPLETE_POSE', pose_input

    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    idx = int(eligible[np.argmin(areas[eligible])])
    globals()['pose_debug']={'source':'YOLO','xy':xy[idx].copy(),'scores':conf[idx].copy(),'threshold':0.5}
    # COCO pose: nose=0, left_eye=1, right_eye=2
    if np.max(conf[idx, [0, 1, 2]]) <= POSE_THRESHOLD:
        return None, 'POSE_FACE_KEYPOINTS_LOW_CONFIDENCE', pose_input
    if np.min(conf[idx, [0, 1, 2]]) < KEYPOINT_VALID_THRESHOLD:
        return None, 'POSE_FACE_LANDMARKS_INCOMPLETE', pose_input
    nose, left, right = xy[idx, 0], xy[idx, 1], xy[idx, 2]
    if not np.all(np.isfinite([*nose, *left, *right])):
        return None, 'POSE_INVALID_FACE_LANDMARKS', pose_input
    if np.linalg.norm(left - right) < 5:
        return None, 'POSE_EYE_SPAN_TOO_SMALL', pose_input

    # Unit direction along the line connecting the eyes; normal points toward nose.
    midpoint = (left + right) / 2
    eye_vector = right - left
    eye_width = float(np.linalg.norm(eye_vector))
    tangent = eye_vector / eye_width
    normal = np.array([-tangent[1], tangent[0]])
    if np.dot(nose - midpoint, normal) < 0:
        normal = -normal
    nose_depth = float(np.dot(nose - midpoint, normal))
    if nose_depth < max(2.0, eye_width * 0.08):
        return None, 'POSE_NOSE_EYE_GEOMETRY_UNRELIABLE', pose_input

    # The upper cut follows the tilted eye line, shifted 20% of eye-to-nose
    # distance away from the nose. Lower cut is 30% beyond the nose.
    top_offset = -0.20 * nose_depth
    bottom_offset = 1.30 * nose_depth

    # Scale the pose-generated face region around its center (1.0 = original).
    # This changes width and height; detected face-coordinate boxes are unchanged.
    face_center_offset = (top_offset + bottom_offset) / 2.0
    half_height = (bottom_offset - top_offset) / 2.0 * POSE_FACE_SCALE
    top_offset = face_center_offset - half_height
    bottom_offset = face_center_offset + half_height
    half_width = (1.35 if child_mask is not None else 2.0) * eye_width * POSE_FACE_SCALE
    left_edge = midpoint - tangent * half_width
    right_edge = midpoint + tangent * half_width
    corners = np.array([
        left_edge + normal * top_offset,
        right_edge + normal * top_offset,
        right_edge + normal * bottom_offset,
        left_edge + normal * bottom_offset,
    ], dtype=np.float32)
    polygon = np.round(corners).astype(np.int32)
    band = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(band, polygon, 255)
    candidate = band > 0

    if child_mask is not None:
        clipped = candidate & child_mask
        if not clipped.any():
            return None, 'POSE_FACE_REGION_EMPTY_AFTER_CHILD_CLIP', pose_input
        # Require meaningful coverage; otherwise segmentation may be incorrect.
        overlap = clipped.sum() / max(candidate.sum(), 1)
        if overlap < 0.30:
            return None, 'POSE_FACE_REGION_LOW_CHILD_OVERLAP', pose_input
        candidate = clipped
        status = 'POSE_EYE_NOSE_REGION_CLIPPED_TO_CHILD_MASK'
    else:
        status = 'POSE_EYE_NOSE_FACE_POLYGON_NO_CHILD_MASK'
    if len(boxes) > 1:
        status += '__SMALLEST_QUALIFIED_PERSON_REVIEW'
    return candidate, status, pose_input




_sapiens_pose_cache=None

def get_sapiens2_pose_1b():
    global _sapiens_pose_cache
    if _sapiens_pose_cache is None:
        import sys,runpy
        sys.path.insert(0,'/content/sapiens2')
        # Match the official Space's import guard against unrelated BLIP/API drift.
        sys.modules['mmpretrain']=None
        from huggingface_hub import hf_hub_download,snapshot_download
        from sapiens.pose.datasets import UDPHeatmap
        from sapiens.pose.models import init_model
        root=Path(snapshot_download('facebook/sapiens2-pose',repo_type='space',allow_patterns=['assets/configs/**']))
        cfg=root/'assets/configs/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py'
        checkpoint=hf_hub_download('facebook/sapiens2-pose-1b','sapiens2_1b_pose.safetensors')
        model=init_model(str(cfg),checkpoint,device='cuda')
        codec_cfg=dict(model.cfg.codec)
        assert codec_cfg.pop('type')=='UDPHeatmap'
        model.codec=UDPHeatmap(**codec_cfg)
        info=runpy.run_path(str(root/'assets/configs/_base_/keypoints308.py'))['dataset_info']
        name2id={v['name']:int(k) for k,v in info['keypoint_info'].items()}
        _sapiens_pose_cache=(model,name2id)
    return _sapiens_pose_cache

def sapiens2_face_mask(rgb,child_mask):
    """Local Sapiens2 1B pose -> same tilted eye/nose facial-band geometry."""
    model,name2id=get_sapiens2_pose_1b()
    pose_input=rgb.copy()
    if child_mask is not None:pose_input[~child_mask]=0
    h,w=rgb.shape[:2]
    # Already cropped to selected child: no second person detector is required.
    box=np.array([[0,0,w-1,h-1]],np.float32)
    data=model.pipeline(dict(img=cv2.cvtColor(pose_input,cv2.COLOR_RGB2BGR),
                             bbox=box,bbox_score=np.ones(1,np.float32)))
    data=model.data_preprocessor(data)
    with torch.inference_mode():
        heatmaps=model(data['inputs']).cpu().numpy()[0]
    keypoints,scores=model.codec.decode(heatmaps)
    meta=data['data_samples']['meta']
    keypoints=(keypoints/meta['input_size']*meta['bbox_scale']+
               meta['bbox_center']-0.5*meta['bbox_scale'])[0]
    scores=np.asarray(scores)[0]
    globals()['pose_debug']={'source':'Sapiens2 1B','xy':keypoints.copy(),'scores':scores.copy(),'threshold':0.3,'name2id':name2id}
    # Resolve names from the 308-keypoint metadata, never COCO hardcoded indices.
    ids=[name2id[name] for name in ['nose','left_eye','right_eye']]
    confidence=scores[ids]
    # Sapiens heatmap scores are model-specific, not calibrated YOLO probabilities.
    SAPIENS_FACE_SCORE_THRESHOLD=0.3
    if not np.isfinite(confidence).all() or np.min(confidence)<SAPIENS_FACE_SCORE_THRESHOLD:
        return None,'SAPIENS2_1B_FACE_KEYPOINTS_LOW_CONFIDENCE',pose_input
    nose,left,right=keypoints[ids]
    if not np.isfinite([*nose,*left,*right]).all():
        return None,'SAPIENS2_1B_INVALID_FACE_LANDMARKS',pose_input
    if any(not (0<=pt[0]<w and 0<=pt[1]<h) for pt in [nose,left,right]):
        return None,'SAPIENS2_1B_FACE_LANDMARKS_OUTSIDE_IMAGE',pose_input
    if np.linalg.norm(left-right)<5:
        return None,'SAPIENS2_1B_EYE_SPAN_TOO_SMALL',pose_input
    midpoint = (left + right) / 2
    eye_vector = right - left
    eye_width = float(np.linalg.norm(eye_vector))
    tangent = eye_vector / eye_width
    normal = np.array([-tangent[1], tangent[0]])
    if np.dot(nose - midpoint, normal) < 0:
        normal = -normal
    nose_depth = float(np.dot(nose - midpoint, normal))
    if nose_depth < max(2.0, eye_width * 0.08):
        return None, 'POSE_NOSE_EYE_GEOMETRY_UNRELIABLE', pose_input

    # The upper cut follows the tilted eye line, shifted 20% of eye-to-nose
    # distance away from the nose. Lower cut is 30% beyond the nose.
    top_offset = -0.20 * nose_depth
    bottom_offset = 1.30 * nose_depth

    # Scale the pose-generated face region around its center (1.0 = original).
    # This changes width and height; detected face-coordinate boxes are unchanged.
    face_center_offset = (top_offset + bottom_offset) / 2.0
    half_height = (bottom_offset - top_offset) / 2.0 * POSE_FACE_SCALE
    top_offset = face_center_offset - half_height
    bottom_offset = face_center_offset + half_height
    half_width = (1.35 if child_mask is not None else 2.0) * eye_width * POSE_FACE_SCALE
    left_edge = midpoint - tangent * half_width
    right_edge = midpoint + tangent * half_width
    corners = np.array([
        left_edge + normal * top_offset,
        right_edge + normal * top_offset,
        right_edge + normal * bottom_offset,
        left_edge + normal * bottom_offset,
    ], dtype=np.float32)
    polygon = np.round(corners).astype(np.int32)
    band = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(band, polygon, 255)
    candidate = band > 0

    if child_mask is not None:
        clipped = candidate & child_mask
        if not clipped.any():
            return None, 'POSE_FACE_REGION_EMPTY_AFTER_CHILD_CLIP', pose_input
        # Require meaningful coverage; otherwise segmentation may be incorrect.
        overlap = clipped.sum() / max(candidate.sum(), 1)
        if overlap < 0.30:
            return None, 'POSE_FACE_REGION_LOW_CHILD_OVERLAP', pose_input
        candidate = clipped
        status = 'POSE_EYE_NOSE_REGION_CLIPPED_TO_CHILD_MASK'
    else:
        status = 'POSE_EYE_NOSE_FACE_POLYGON_NO_CHILD_MASK'
    return candidate,'SAPIENS2_1B_'+status,pose_input


def draw_manual_face_box(rgb, child_mask):
    """Colab canvas: draw full-image face rectangle; preview final 75% rule."""
    import base64,io,uuid
    from PIL import Image
    from IPython.display import HTML,display
    from google.colab.output import eval_js
    uid='face_'+uuid.uuid4().hex
    def url(array):
        buf=io.BytesIO();Image.fromarray(array).save(buf,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    image_url=url(rgb)
    mask_url=url(child_mask.astype(np.uint8)*255)
    html=r"""
<div id="__ID__">
<p><b>Automatic masking failed. Drag a rectangle around the entire face.</b><br>
Preview applies the 75% rule: intersection if overlap ≥75%; otherwise full box.</p>
<canvas style="width:540px;max-width:100%;border:1px solid #888;touch-action:none"></canvas><br>
<button class="reset">Reset</button> <button class="apply">Apply mask</button>
<button class="skip">Skip manual masking</button> <span class="status"></span>
</div>
<script>
(()=>{
 const root=document.getElementById('__ID__'),c=root.querySelector('canvas'),ctx=c.getContext('2d');
 const im=new Image(),seg=new Image();let ready=0,start=null,box=null,drag=false,pixels=null,done;
 root.selectionPromise=new Promise(r=>done=r);
 function loaded(){if(++ready!==2)return;c.width=im.naturalWidth;c.height=im.naturalHeight;
  const mc=document.createElement('canvas');mc.width=c.width;mc.height=c.height;
  const mx=mc.getContext('2d');mx.drawImage(seg,0,0,c.width,c.height);pixels=mx.getImageData(0,0,c.width,c.height).data;draw();}
 function draw(){if(!pixels)return;ctx.drawImage(im,0,0);if(!box)return;
  const [a,b,d,e]=box;let total=0,inside=0;
  for(let y=b;y<e;y++)for(let x=a;x<d;x++){total++;if(pixels[(y*c.width+x)*4]>0)inside++;}
  const ratio=inside/Math.max(total,1);const data=ctx.getImageData(a,b,d-a,e-b);
  for(let y=b;y<e;y++)for(let x=a;x<d;x++)if(ratio<0.75||pixels[(y*c.width+x)*4]>0){
   const k=((y-b)*(d-a)+x-a)*4;data.data[k]=data.data[k+1]=data.data[k+2]=0;}
  ctx.putImageData(data,a,b);ctx.strokeStyle='#00ff00';ctx.lineWidth=2;ctx.strokeRect(a,b,d-a,e-b);
  root.querySelector('.status').textContent='Overlap '+(100*ratio).toFixed(1)+'% | '+(ratio>=0.75?'intersection':'full rectangle');}
 function pos(e){const r=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width,Math.round((e.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.round((e.clientY-r.top)*c.height/r.height)))];}
 function update(e){const p=pos(e);box=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];if(box[2]>box[0]&&box[3]>box[1])draw();}
 c.onpointerdown=e=>{if(!pixels)return;e.preventDefault();start=pos(e);drag=true;c.setPointerCapture(e.pointerId);};
 c.onpointermove=e=>{if(drag)update(e);};c.onpointerup=e=>{if(drag){update(e);drag=false;}};
 root.querySelector('.reset').onclick=()=>{box=null;root.querySelector('.status').textContent='';draw();};
 root.querySelector('.apply').onclick=()=>{if(!drag&&box&&box[2]-box[0]>1&&box[3]-box[1]>1)done(box);else root.querySelector('.status').textContent='Draw a valid rectangle first';};
 root.querySelector('.skip').onclick=()=>done(null);
 im.onload=loaded;seg.onload=loaded;im.src=__IMAGE__;seg.src=__MASK__;
})();
</script>
"""
    html=html.replace('__ID__',uid).replace('__IMAGE__',json.dumps(image_url)).replace('__MASK__',json.dumps(mask_url))
    display(HTML(html))
    return eval_js(f"document.getElementById('{uid}').selectionPromise")
