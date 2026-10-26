# 4. Independent IQR row filling for child and background.
# IQR mean = mean of values within Tukey fences [Q1-1.5*IQR,Q3+1.5*IQR].
# Entire missing region-row: linear interpolation of nearest ORIGINAL valid
# region-row means above/below, with nearest extrapolation at image edges.
def iqr_mean(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    if not len(a):return np.nan
    q1,q3=np.percentile(a,[25,75]);span=q3-q1
    keep=a[(a>=q1-1.5*span)&(a<=q3+1.5*span)]
    return float(keep.mean())

def fill_depth_rows(z,mask):
    result=np.array(z,dtype=float,copy=True)
    original_valid=np.isfinite(result)
    method=np.zeros(z.shape,np.uint8) # 0 observed, 1 same-row, 2 interpolated row, 3 edge extrapolation, 255 unresolved
    for region in [mask,~mask]:
        means=np.array([iqr_mean(z[y,region[y]]) for y in range(len(z))])
        good=np.flatnonzero(np.isfinite(means))
        for y in range(len(z)):
            missing=region[y]&~original_valid[y]
            if not missing.any():continue
            if np.isfinite(means[y]):value,flag=means[y],1
            elif len(good):
                value=float(np.interp(y,good,means[good]));flag=2 if good[0]<y<good[-1] else 3
            else:value,flag=np.nan,255
            result[y,missing]=value;method[y,missing]=flag
    return result,method
filled,fill_method=fill_depth_rows(depth,child)
print('Child missing fraction:',float((child&~np.isfinite(depth)).sum()/child.sum()))
print('Child fill flags:',dict(zip(*np.unique(fill_method[child],return_counts=True))))
np.save(OUT/'depth_filled_m.npy',filled);np.save(OUT/'depth_fill_method.npy',fill_method)
plt.figure(figsize=(15,4))
for i,(a,t) in enumerate([(depth,'Original metres'),(filled,'Filled metres'),(fill_method,'Fill provenance')]):
    plt.subplot(1,3,i+1);plt.imshow(a);plt.title(t);plt.colorbar()
plt.show()

# Camera-to-child depth: IQR-filtered mean over the segmented foreground.
# Optical-axis Z distance, not radial range. Primary result uses observed
# depth so filling does not artificially increase the weight of missing rows.
def segmented_camera_distance(z,mask):
    values=np.asarray(z)[mask]
    values=values[np.isfinite(values)&(values>0)]
    if not len(values):
        return {'distance_m':None,'distance_cm':None,'valid_pixels':0,
                'retained_pixels':0,'status':'NO_VALID_CHILD_DEPTH'}
    q1,q3=np.percentile(values,[25,75]);spread=q3-q1
    low,high=q1-1.5*spread,q3+1.5*spread
    retained=values[(values>=low)&(values<=high)]
    distance=float(retained.mean())
    return {'distance_m':distance,'distance_cm':distance*100,
            'valid_pixels':int(len(values)),'retained_pixels':int(len(retained)),
            'iqr_fence_low_m':float(low),'iqr_fence_high_m':float(high),
            'status':'OK','convention':'optical_axis_Z'}
camera_distance={
    'observed_segmented_depth':segmented_camera_distance(depth,child),
    'filled_segmented_depth':segmented_camera_distance(filled,child),
}
primary_distance=camera_distance['observed_segmented_depth']
if primary_distance['distance_m'] is not None:
    print(f"Camera-to-child distance (IQR mean, observed): {primary_distance['distance_m']:.3f} m ({primary_distance['distance_cm']:.2f} cm)")
else:
    print('Camera-to-child observed distance unavailable: no valid child depth.')
filled_distance=camera_distance['filled_segmented_depth']
if filled_distance['distance_m'] is not None:
    print(f"Camera-to-child distance (IQR mean, filled): {filled_distance['distance_m']:.3f} m ({filled_distance['distance_cm']:.2f} cm)")
print(json.dumps(camera_distance,indent=2))
