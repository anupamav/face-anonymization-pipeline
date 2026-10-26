# 5. Height: head tip and lowest foreground in EXACTLY the same image column.
assert REGISTERED_TO_RGB
assert PORTRAIT_CALIBRATION_CONFIRMED

def xyz(u,v,z):
    return np.array([(u-CX)*z/FX,(v-CY)*z/FY,z],float)

def estimate_height(mask,z,observed):
    rows=np.flatnonzero(mask.any(axis=1))
    if not len(rows):raise ValueError('Empty silhouette')
    head_y=int(rows[0])
    head_columns=np.flatnonzero(mask[head_y])
    # Pick an actual foreground pixel nearest the centre of the topmost row.
    head_x=int(head_columns[np.argmin(abs(head_columns-np.median(head_columns)))])
    column_rows=np.flatnonzero(mask[:,head_x])
    foot_y=int(column_rows[-1])
    if foot_y<=head_y:raise ValueError('No lower silhouette point in head column')
    # Do not switch to a different foot column when this column ends at torso/leg.
    row_span=int(rows[-1]-rows[0]+1)
    warning=None
    if foot_y < rows[0]+0.9*(row_span-1):
        warning='Head column does not reach lowest 10% of silhouette; endpoint may not be a foot. Review overlay.'
    # Preserve previous 10%-band robust depth rule, now anchored at chosen endpoints.
    n=max(1,int(np.ceil(row_span*0.10)))
    yy=np.arange(mask.shape[0])[:,None]
    top=mask & (yy>=head_y) & (yy<head_y+n)
    bottom=mask & (yy<=foot_y) & (yy>foot_y-n)
    za=iqr_mean(z[top]);zb=iqr_mean(z[bottom])
    if not np.isfinite([za,zb]).all():raise ValueError('No usable endpoint-band depth')
    a=xyz(head_x,head_y,za);b=xyz(head_x,foot_y,zb)
    distance_cm=float(np.linalg.norm(a-b)*100)
    result={'height_tip_euclidean_cm':distance_cm,
            'head_pixel_xy':[head_x,head_y],'foot_pixel_xy':[head_x,foot_y],
            'head_xyz_m':a.tolist(),'foot_xyz_m':b.tolist(),
            'head_band_depth_m':za,'foot_band_depth_m':zb,
            'top_band_observed_fraction':float(np.isfinite(observed[top]).mean()),
            'bottom_band_observed_fraction':float(np.isfinite(observed[bottom]).mean()),
            'same_column':True,'endpoint_warning':warning}
    if GRAVITY_CAMERA is not None:
        g=np.asarray(GRAVITY_CAMERA,float)
        assert g.shape==(3,) and np.isfinite(g).all() and np.linalg.norm(g)>0
        result['height_gravity_projection_cm']=float(abs(np.dot(a-b,g/np.linalg.norm(g)))*100)
    return result,[head_x,head_y],[head_x,foot_y]
height,head_pixel,toe_pixel=estimate_height(child,filled,depth)
print(f"Height of child (3D Euclidean distance): {height['height_tip_euclidean_cm']:.2f} cm")
if height['endpoint_warning']:print('REVIEW:',height['endpoint_warning'])
print(json.dumps(height,indent=2))
height_preview=segmented_rgb.copy()
a=tuple(head_pixel);b=tuple(toe_pixel)
cv2.line(height_preview,a,b,(0,255,0),2)
cv2.circle(height_preview,a,6,(255,0,0),-1);cv2.circle(height_preview,b,6,(0,0,255),-1)
plt.figure(figsize=(6,10));plt.imshow(height_preview)
plt.title(f"Same-column height: {height['height_tip_euclidean_cm']:.2f} cm");plt.show()
