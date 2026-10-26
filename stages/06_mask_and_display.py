# 7. Face masking: MediaPipe -> 75% box rule -> YOLO pose -> Sapiens2 1B -> manual.
# Set FORCE_MANUAL=True to redraw even when automatic masking succeeds.

face_mask=None;face_status='NOT_RUN';overlap=None;FACE_BOX=None
masking_trace=[];pose_debug=None
# Crop selected child for pose and face inference; retain full image coordinates.
yy,xx=np.where(child)
pad=max(2,int(0.05*max(np.ptp(xx),np.ptp(yy))))
x0=max(0,int(xx.min())-pad);x1=min(rgb.shape[1],int(xx.max())+pad+1)
y0=max(0,int(yy.min())-pad);y1=min(rgb.shape[0],int(yy.max())+pad+1)
crop_rgb=rgb[y0:y1,x0:x1];crop_mask=child[y0:y1,x0:x1]

# A. MediaPipe FACE DETECTOR (not MediaPipe pose).
if not FORCE_MANUAL:
    try:
        import mediapipe as mp
        import urllib.request
        model_path=Path('/content/blaze_face_short_range.tflite')
        if not model_path.exists():
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
                str(model_path))
        options=mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,min_detection_confidence=0.5)
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            detection=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop_rgb)))
        candidates=[]
        for item in detection.detections:
            b=item.bounding_box
            a=max(0,int(b.origin_x+x0));btop=max(0,int(b.origin_y+y0))
            c=min(rgb.shape[1],int(b.origin_x+x0+b.width));d=min(rgb.shape[0],int(b.origin_y+y0+b.height))
            if c<=a or d<=btop:continue
            cx,cy=(a+c)//2,(btop+d)//2
            # Associate face with selected silhouette and its upper region.
            if not child[cy,cx] or cy>yy.min()+0.4*(yy.max()-yy.min()+1):continue
            candidates.append((float(item.categories[0].score),[a,btop,c,d]))
        if candidates:
            score,FACE_BOX=max(candidates,key=lambda item:item[0])
            face_mask,logic,overlap=intersection_logic(FACE_BOX,child,*child.shape)
            face_status='MEDIAPIPE_FACE_'+logic
            masking_trace.append({'stage':'MEDIAPIPE_FACE','status':face_status,'score':score,'overlap':overlap})
        else:masking_trace.append({'stage':'MEDIAPIPE_FACE','status':'NO_MATCHING_FACE'})
    except (ImportError,OSError,ValueError,RuntimeError) as exc:
        masking_trace.append({'stage':'MEDIAPIPE_FACE','status':'ERROR','reason':str(exc)})

# B. YOLO pose fallback, only when MediaPipe produced no usable mask.
if face_mask is None and not FORCE_MANUAL:
    try:
        local,status,_=pose_face_mask(crop_rgb,crop_mask,pose_model)
        face_status='YOLO_'+status
        if local is not None:
            face_mask=np.zeros_like(child);face_mask[y0:y1,x0:x1]=local
        masking_trace.append({'stage':'YOLO_POSE','status':face_status})
    except (ValueError,RuntimeError,AttributeError,IndexError) as exc:
        masking_trace.append({'stage':'YOLO_POSE','status':'ERROR','reason':str(exc)})

# C. Sapiens2 1B pose fallback; lazy loading only when needed.
if face_mask is None and not FORCE_MANUAL:
    try:
        local,face_status,_=sapiens2_face_mask(crop_rgb,crop_mask)
        if local is not None:
            face_mask=np.zeros_like(child);face_mask[y0:y1,x0:x1]=local
        masking_trace.append({'stage':'SAPIENS2_1B_POSE','status':face_status})
    except (ImportError,OSError,ValueError,RuntimeError,KeyError) as exc:
        masking_trace.append({'stage':'SAPIENS2_1B_POSE','status':'ERROR','reason':str(exc)})

# D. Manual rectangle -> child segmentation intersection using same 75% rule.
if face_mask is None:
    print('Automatic masking unavailable or manual override selected.')
    print(json.dumps(masking_trace,indent=2))
    FACE_BOX=draw_manual_face_box(rgb,child)
    if FACE_BOX is not None:
        face_mask,logic,overlap=intersection_logic(FACE_BOX,child,*child.shape)
        face_status='MANUAL_DRAW_'+logic
        masking_trace.append({'stage':'MANUAL_DRAW','status':face_status,'overlap':overlap})
    else:
        face_status='MANUAL_SKIPPED_UNMASKED'
        masking_trace.append({'stage':'MANUAL_DRAW','status':face_status})
print('Face/head:',face_status,'overlap:',overlap)
anonymized=None
if face_mask is not None:
    anonymized=rgb.copy();anonymized[face_mask]=0
    cv2.imwrite(str(OUT/'face_mask.png'),face_mask.astype(np.uint8)*255)
    cv2.imwrite(str(OUT/'face_masked_review_required.png'),cv2.cvtColor(anonymized,cv2.COLOR_RGB2BGR))
else:
    for stale in ['face_mask.png','face_masked_review_required.png']:
        (OUT/stale).unlink(missing_ok=True)
    print('MANUAL REVIEW: no face mask produced; no masked RGB exported.')
result={'height':height,'camera_to_child_distance':camera_distance,'face_status':face_status,'masking_trace':masking_trace,'face_bbox_xyxy':FACE_BOX,'face_overlap':overlap,'face_review_required':True,
        'calibration':dict(fx=FX,fy=FY,cx=CX,cy=CY,depth_scale=DEPTH_SCALE),
        'interpretation':'Geometric prototype; not validated clinical anthropometry'}
(OUT/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
# 8. Clean six-panel output summary; pose uses the actual inference above.
import textwrap
measurement_rgb=rgb.copy()
head_xy=tuple(map(int,head_pixel));foot_xy=tuple(map(int,toe_pixel))
cv2.line(measurement_rgb,head_xy,foot_xy,(0,255,0),3)
for point,color,label in [(head_xy,(255,70,70),'Head tip'),(foot_xy,(50,160,255),'Foot endpoint')]:
    cv2.circle(measurement_rgb,point,7,color,-1)
    tx=max(4,min(point[0]+12,rgb.shape[1]-160));ty=max(25,min(point[1]-10,rgb.shape[0]-10))
    cv2.putText(measurement_rgb,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)
height_text=f"Height: {height['height_tip_euclidean_cm']:.2f} cm (3D)"
d=primary_distance['distance_m']
distance_text=f"Camera distance: {d:.3f} m" if d is not None else 'Camera distance: unavailable'
cv2.rectangle(measurement_rgb,(8,8),(rgb.shape[1]-8,105),(20,25,30),-1)
cv2.putText(measurement_rgb,height_text,(20,44),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
cv2.putText(measurement_rgb,distance_text,(20,82),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2,cv2.LINE_AA)
pose_rgb=rgb.copy()
pose_title='Pose not used / unavailable'
if pose_debug is not None:
    pose_title=pose_debug['source']+' pose'
    points=np.asarray(pose_debug['xy'])+np.array([x0,y0])
    confidence=np.asarray(pose_debug['scores'])
    visible=np.isfinite(points).all(axis=1)&(confidence>=pose_debug['threshold'])
    visible&=(points[:,0]>=0)&(points[:,0]<rgb.shape[1])&(points[:,1]>=0)&(points[:,1]<rgb.shape[0])
    coco_links=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    if pose_debug['source']=='YOLO':links=coco_links
    else:
        names=pose_debug['name2id']
        body_names=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
        links=[(names[body_names[a]],names[body_names[b]]) for a,b in coco_links if body_names[a] in names and body_names[b] in names]
    for a,b in links:
        if visible[a] and visible[b]:cv2.line(pose_rgb,tuple(np.rint(points[a]).astype(int)),tuple(np.rint(points[b]).astype(int)),(0,255,100),2)
    for point in points[visible]:cv2.circle(pose_rgb,tuple(np.rint(point).astype(int)),2,(255,70,30),-1)

fig,axes=plt.subplots(2,3,figsize=(15,18),layout='constrained')
fig.suptitle(f'RGB-D child height pipeline | Pair {PAIR_ID}',fontsize=19,fontweight='bold')
axes[0,0].imshow(rgb);axes[0,0].set_title('1. Original RGB',fontsize=13)
observed_values=depth[np.isfinite(depth)]
lo,hi=np.percentile(observed_values,[2,98])
cmap=plt.get_cmap('turbo').copy();cmap.set_bad('#eeeeee')
im1=axes[0,1].imshow(np.ma.masked_invalid(depth),cmap=cmap,vmin=lo,vmax=hi)
axes[0,1].set_title('2. Original depth | missing = grey',fontsize=13)
axes[0,2].imshow(np.ma.masked_invalid(filled),cmap=cmap,vmin=lo,vmax=hi)
axes[0,2].set_title('3. Inpainted depth | same colour scale',fontsize=13)
fig.colorbar(im1,ax=[axes[0,1],axes[0,2]],label='Depth (m)',shrink=0.65,pad=0.02)
axes[1,0].imshow(measurement_rgb);axes[1,0].set_title('4. Height + camera distance',fontsize=13)
if height['endpoint_warning']:
    axes[1,0].text(0.02,0.02,'REVIEW: column may miss foot',transform=axes[1,0].transAxes,color='white',fontsize=10,bbox=dict(facecolor='#a33',alpha=0.9,pad=5))
axes[1,1].imshow(pose_rgb);axes[1,1].set_title('5. '+pose_title,fontsize=13)
if pose_debug is None:
    axes[1,1].text(0.5,0.08,'No confident pose detected',ha='center',transform=axes[1,1].transAxes,color='white',bbox=dict(facecolor='#a33',pad=5))
if anonymized is not None:
    axes[1,2].imshow(anonymized)
    face_summary='MASKED | REVIEW REQUIRED'
else:
    # Do not present the unmasked original as a masked output.
    axes[1,2].imshow(np.full_like(rgb,235))
    face_summary='FAILED | MANUAL MASKING REQUIRED'
axes[1,2].set_title('6. Face masking',fontsize=13)
status_text=face_summary+'\n'+textwrap.fill(face_status.replace('_',' '),38)
axes[1,2].text(0.5,0.02,status_text,ha='center',va='bottom',transform=axes[1,2].transAxes,fontsize=9,color='white',bbox=dict(facecolor='#263238',alpha=0.92,pad=6))
for ax in axes.flat:ax.axis('off')
fig.savefig(OUT/'pipeline_summary.png',dpi=150,bbox_inches='tight')
# Persist source and standalone depth previews alongside the displayed summary.
cv2.imwrite(str(OUT/'original_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
np.save(OUT/'original_depth_m.npy',depth)
plt.imsave(OUT/'original_depth_preview.png',np.ma.masked_invalid(depth),cmap=cmap,vmin=lo,vmax=hi)
plt.imsave(OUT/'inpainted_depth_preview.png',np.ma.masked_invalid(filled),cmap=cmap,vmin=lo,vmax=hi)

plt.show()
for filename,array in [('height_distance_rgb.png',measurement_rgb),('pose_rgb.png',pose_rgb)]:
    cv2.imwrite(str(OUT/filename),cv2.cvtColor(array,cv2.COLOR_RGB2BGR))
print('All outputs saved in:',OUT.resolve())
print('Displayed summary:',OUT/'pipeline_summary.png')
print(height_text+' | '+distance_text+' | '+face_summary)

# Export only after reviewing all masks. Depth/child masks may remain identifying data.

if EXPORT_REVIEWED:
    import shutil
    print('Reviewed output archive:',shutil.make_archive(str(OUT.parent/(OUT.name+'_reviewed')),'zip',OUT))
