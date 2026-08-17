# Object Detection and Depth Estimation

Monocular car-distance estimation on the **KITTI dataset**, combining 2D object detection with geometric depth estimation from a single camera. Cars are detected in 2D images with **YOLO11X**, their distances are estimated using intrinsic camera parameters and a pinhole-camera projection model, and results are evaluated against KITTI ground truth using Precision, Recall, and IoU.

> University project for *Computer Vision*, Hochschule Ravensburg-Weingarten, guided by Prof. Dr. rer. nat. Stefan Elser.

## How It Works

1. **Detection**: Run YOLO11X on KITTI images, keeping only the `car` class (pedestrians, trucks, traffic signs, etc. filtered out). Each detection produces a 2D bounding box and confidence score.
2. **Distance estimation**: Using the per-image intrinsic camera matrix `K` from KITTI's calibration files, back-project the bottom-center pixel of each car's bounding box toward 3D world coordinates.
3. **Resolving depth ambiguity**: A single 2D point maps to an infinite line in 3D. This is resolved using the known, fixed camera height (1.65 m) and the assumption that the road is a flat plane — letting the depth scalar `z` be solved directly from the projection equation.
4. **Matching to ground truth**: Ground-truth 3D bounding boxes are converted to 2D and compared to YOLO detections using Intersection over Union (IoU), with a 0.4 threshold (lowered to 0.3 for heavily crowded scenes) to filter extra detections not present in the labels.
5. **Evaluation**: Precision and Recall are computed per image and aggregated, and estimated distances are plotted against KITTI ground-truth distances to visualize accuracy.

## Results

- **98** ground-truth labeled cars in the evaluation set
- **120** total detections by YOLO11X
- **90** points matched and plotted for distance comparison
- **8 cars went undetected** across 6 images (see `Report.pdf` for the full per-image breakdown)
- Precision and Recall per image ranged widely (roughly 0.5–1.0 depending on scene density), with performance dropping in heavily crowded street scenes

Distance accuracy is strong at close range but **degrades with distance and in specific failure conditions** (see below) — the report concludes that a distance error above 3–5 m is not acceptable for real automotive use, and that monocular estimation alone is not sufficient — a single camera should not be relied on as the only distance source.

## Key Failure Modes Identified

- **Half-visible / occluded cars**: when a bounding box doesn't fully enclose the car (e.g., partially out of frame), the reference point used for distance calculation falls outside the actual vehicle, producing large distance errors.
- **Non-flat road assumption**: the depth model assumes a flat driving plane; on sloped/hilly roads this assumption breaks down and produces significantly overestimated distances (e.g., one case measured 100.4 m vs. a 67.3 m ground truth).
- **Detection class confusion**: in dense scenes, some cars were detected under the `truck` class instead of `car` and were excluded from car-only evaluation.
- **Crowded-scene IoU mismatches**: overlapping bounding boxes in dense traffic required a lower IoU threshold (0.3 instead of 0.4) for correct matching in at least one heavily crowded image.

## Repo Structure

```
├── Code/                    # Python scripts for detection, distance estimation, evaluation
├── output_Final/            # Output images with bounding boxes, distances, precision/recall
├── KITTI_Selection.zip      # Selected KITTI images + calibration files used in this project
├── Report.pdf               # Full write-up: methodology, results, error analysis
└── README.md
```

## Tech Stack

- Python
- YOLO11X ([Ultralytics](https://docs.ultralytics.com/tasks/detect/)) — 2D car detection
- OpenCV (`cv2`) — image processing, bounding box visualization
- NumPy — camera geometry / matrix operations
- Matplotlib — result visualization (precision/recall, distance comparison plots)
- [KITTI dataset selection](https://elearning.rwu.de/mod/resource/view.php?id=193664)

## References

1. [YOLO11X documentation](https://docs.ultralytics.com/tasks/detect/)
2. [KITTI dataset](http://www.cvlibs.net/datasets/kitti/)

## Contributors

- Arjun Rameshbhai Beladiya
- Dhaval Jagdish Mistry
