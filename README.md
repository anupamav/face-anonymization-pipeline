# Child height and face masking pipeline

This pipeline estimates height and camera-to-child depth from aligned RGB-D images and anonymizes the face. MUAC estimation is not included.

## Inputs and calibration

Inputs are an RGB image, a registered depth map, and camera intrinsics. NumPy arrays, RGB PNG/JPEG, and raw depth PNG/TIFF are supported. A coloured depth visualization is not a metric depth map.

The supplied portrait calibration uses height 1280 and width 720:

| Parameter | Value |
| --- | ---: |
| fx | 649.872 |
| fy | 649.872 |
| cx | 362.120112 |
| cy | 653.496832 |

Depth scale defaults to 1.0 for values already in metres. Calibration must match the stored orientation and resolution. Registration and calibration flags default to True; these settings do not verify physical alignment.

## 1. Child segmentation

YOLO first detects person bounding boxes. When several people are detected, the target person must be selected explicitly.

- **YOLO succeeds:** pass the selected bounding box to SAM and use its box-prompted segmentation. The image centre is not used for mask selection.
- **YOLO finds no person:** pass the full RGB image to SAM without a bounding box. Select the mask whose foreground contains the image centre. If several masks contain that pixel, select the smallest by area.
- If no unprompted mask contains the image centre, selection requires review. Multiple box-prompted masks require explicit selection. Manual polygon and mask-index overrides are available.

The selected silhouette becomes a binary mask. Inspect it for missing body regions or included background before measuring height.

## 2. Missing-depth filling

Process child and background regions independently using the segmentation mask:

1. Treat nonfinite and out-of-range depth values as missing.
2. For each region in each row, calculate the mean of valid depths after IQR filtering.
3. Fill missing pixels with that region's row mean.
4. When an entire region-row lacks valid depth, interpolate between the nearest original valid region-row means above and below. At the edges, use the nearest available row mean.
5. If a region has no valid depth anywhere, leave it unresolved.

Original valid depth pixels remain unchanged. IQR filtering retains values between Q1 − 1.5 × IQR and Q3 + 1.5 × IQR.

## 3. Height estimation

Identify the topmost silhouette row and choose a foreground pixel nearest its horizontal median as the head tip. Find the lowest foreground pixel in **exactly the same column** as the foot endpoint.

Estimate robust depth for each endpoint using the IQR-filtered mean in the adjacent 10% silhouette-height band. Deproject using:

```text
X = (u − cx) × Z / fx
Y = (v − cy) × Z / fy
Z = metric depth
```

Height is the 3D Euclidean distance between the head and foot endpoints, expressed in centimetres. An optional calibrated gravity vector enables gravity-projected height.

If the head column does not reach a foot, its lowest foreground pixel may be on the torso or leg. A review warning identifies endpoints above the lowest 10% of the silhouette. Posture, hair, shoes, segmentation errors and depth bias affect the estimate.

## 4. Camera-to-child distance

Take positive finite depth values inside the selected child mask, remove IQR outliers, and calculate their mean. Report both observed-depth and filled-depth estimates; the primary estimate uses observed depth to avoid overweighting filled pixels.

This is optical-axis Z distance, rather than radial distance from the camera origin.

## 5. Face masking

Fallback order:

1. **MediaPipe face detection:** detect a face associated with the selected silhouette and its upper region.
2. **75% segmentation-intersection rule:** intersect the detected face rectangle with the child mask. If the intersection covers at least 75% of the rectangle, use the intersection; otherwise use the full rectangle.
3. **YOLO pose fallback:** if no usable detected-face mask exists, derive a tilted facial band from confident eye and nose landmarks using the reference masking logic.
4. **Sapiens2 1B pose fallback:** if YOLO face pose fails, estimate facial landmarks with Sapiens2 1B and apply the same facial-band geometry and child clipping. Model-specific keypoint scores are used.
5. **Manual drawing:** if automatic masking fails, draw a face rectangle. Preview and apply the same 75% segmentation-intersection rule. Manual override can also replace an automatic result.

The final region is blacked out in RGB. Every face mask requires visual review for exposed facial pixels. If manual masking is skipped, the image remains explicitly unmasked.

## Limitations

Person, pose and face models may fail on mannequins. A centre-containing SAM mask may represent only a body part. Geometric height is a prototype estimate requiring validation against reference measurements. Automatic face masking does not guarantee anonymity.
