# 3. YOLO -> SAM; if no person is detected, run SAM on full RGB WITHOUT bbox.
from ultralytics import YOLO, SAM


detector = YOLO('yolo11n.pt')
pose_model = YOLO('yolo11n-pose.pt')
r = detector.predict(bgr, classes=[0], conf=0.35, verbose=False)[0]
boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0,4))
bbox = None
if len(boxes):
    preview = rgb.copy()
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = np.round(box).astype(int)
        cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(preview,str(i),(x1,y1+20),0,0.8,(255,0,0),2)
    plt.figure(figsize=(6,10));plt.imshow(preview);plt.title('YOLO detections');plt.show()
    if len(boxes)>1 and PERSON_INDEX is None:
        raise ValueError('Set PERSON_INDEX to the child/mannequin detection and rerun')
    idx = 0 if PERSON_INDEX is None else int(PERSON_INDEX)
    assert 0 <= idx < len(boxes)
    bbox = boxes[idx]
    print('YOLO detected person: SAM uses its bbox prompt.')
else:
    print('YOLO found no person: passing FULL RGB to SAM WITHOUT a bounding box.')

if MANUAL_CHILD_POLYGON is not None:
    polygon = np.asarray(MANUAL_CHILD_POLYGON,np.int32)
    assert polygon.ndim==2 and polygon.shape[1]==2 and len(polygon)>=3
    child = np.zeros(rgb.shape[:2],np.uint8)
    cv2.fillPoly(child,[polygon],1);child=child.astype(bool)
else:
    sam = SAM('sam_b.pt')
    # Convert RGB to BGR only for Ultralytics ndarray input convention.
    # This is the full original image, not a crop or a blacked-out image.
    sam_input = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    if bbox is None:
        sr = sam.predict(source=sam_input, verbose=False)[0]  # NO bboxes argument
    else:
        sr = sam.predict(source=sam_input,bboxes=[bbox.tolist()],verbose=False)[0]
    if sr.masks is None or len(sr.masks.data)==0:
        raise ValueError('SAM returned no masks. Supply MANUAL_CHILD_POLYGON.')
    masks = sr.masks.data.cpu().numpy()
    print('SAM masks returned:',len(masks))
    # Select masks whose foreground includes the IMAGE centre (not principal point).
    image_h,image_w=rgb.shape[:2]
    center_x,center_y=image_w//2,image_h//2
    full_masks=np.stack([cv2.resize(m.astype(np.uint8),(image_w,image_h),
                        interpolation=cv2.INTER_NEAREST)>0 for m in masks])
    if SAM_MASK_INDEX is not None:
        mi=int(SAM_MASK_INDEX)
        assert 0<=mi<len(masks)
    elif bbox is not None:
        # YOLO succeeded: use its box-prompted SAM result, regardless of image centre.
        if len(masks)!=1:
            raise ValueError('Multiple box-prompted masks: set SAM_MASK_INDEX explicitly.')
        mi=0
    else:
        # ONLY the no-YOLO, unprompted fallback uses the IMAGE centre.
        candidates=np.flatnonzero(full_masks[:,center_y,center_x])
        print(f'Fallback centre ({center_x}, {center_y}); containing masks: {candidates.tolist()}')
        if not len(candidates):
            raise ValueError('No unprompted SAM mask contains the image centre. Set SAM_MASK_INDEX manually.')
        areas=full_masks[candidates].sum(axis=(1,2))
        mi=int(candidates[np.argmin(areas)])
    print('Selected SAM mask:',mi)
    selected_preview=rgb.copy()
    chosen=full_masks[mi]
    selected_preview[chosen]=(0.5*selected_preview[chosen]+np.array([0,127,0])).clip(0,255).astype(np.uint8)
    if bbox is None:
        cv2.drawMarker(selected_preview,(center_x,center_y),(255,0,0),cv2.MARKER_CROSS,20,2)
    plt.figure(figsize=(6,10));plt.imshow(selected_preview)
    plt.title(f'Selected mask {mi} | '+('YOLO box prompt' if bbox is not None else 'image-centre fallback'));plt.show()
    child=cv2.resize(masks[mi].astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>0
n, labels, stats, _=cv2.connectedComponentsWithStats(child.astype(np.uint8),8)
assert n>1, 'Selected mask is empty'
# Centre-based component selection is also restricted to the YOLO-failure path.
center_label=labels[child.shape[0]//2,child.shape[1]//2]
use_center=(bbox is None and MANUAL_CHILD_POLYGON is None and SAM_MASK_INDEX is None and center_label>0)
component=int(center_label) if use_center else 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
child=labels==component
ys,xs=np.where(child)
assert len(ys)>0
# Keep downstream pose matching compatible with the unprompted SAM path.
if bbox is None:
    bbox=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],dtype=float)
segmented_rgb=rgb.copy();segmented_rgb[~child]=0
cv2.imwrite(str(OUT/'child_mask.png'),child.astype(np.uint8)*255)
plt.figure(figsize=(12,8));plt.subplot(121);plt.imshow(child,cmap='gray');plt.title('Selected binary silhouette')
plt.subplot(122);plt.imshow(segmented_rgb);plt.title('Segmented RGB');plt.show()
if xs.min()==0 or ys.min()==0 or xs.max()==rgb.shape[1]-1 or ys.max()==rgb.shape[0]-1:
    raise ValueError('Selected silhouette touches image border: inspect for background mask or truncated subject')
