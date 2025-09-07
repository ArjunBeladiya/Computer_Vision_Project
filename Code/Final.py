import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11x.pt")

# Ensure output directories exist
output_folder = f"output_Final"
os.makedirs(output_folder, exist_ok=True)

image_folder = "KITTI_Selection/images"

# Function to read calibration data
def read_calibration(calibration_file):
    with open(calibration_file, "r") as file:
        return np.loadtxt(calibration_file, delimiter=' ')

# Function to read ground truth data
def read_ground_truth(ground_truth_file):
    ground_truth = []
    with open(ground_truth_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            label = parts[0]
            x_min, y_min, x_max, y_max = map(float, parts[1:5])
            distance = float(parts[5])
            ground_truth.append({"label": label, "bbox": [x_min, y_min, x_max, y_max], "distance": distance})
    return ground_truth

# Function to calculate IoU
def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) for two bounding boxes.
    Args:
        box1: [x_min, y_min, x_max, y_max] for box1
        box2: [x_min, y_min, x_max, y_max] for box2
    Returns:
        IoU: Intersection over Union value (0 <= IoU <= 1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of union
    union = box1_area + box2_area - intersection

    # Compute IoU
    iou = intersection / union #if union != 0 else 0
    return iou

# Function to draw a legend on the image
def draw_legend(image, legend_data):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)          # Black background

    header = ["Ref", "GT_Dist", "YOLO_Dist", "IoU"]
    margin = 15
    line_height = 20
    x_start = image.shape[1] - 290 # Fixed width for the legend
    y_start = margin

    # Draw header
    header_text = f"{header[0]:<6}{header[1]:<9}{header[2]:<12}{header[3]:<12}"
    cv2.rectangle(image, (x_start - 5, y_start - 5), (x_start + 275, y_start + line_height), bg_color, -1)
    cv2.putText(image, header_text, (x_start, y_start + 15), font, font_scale, text_color, font_thickness)

    # Draw each row
    for i, (ref, gt_distance, yolo_distance,best_iou) in enumerate(legend_data):
        y_pos = y_start + (i + 2) * line_height
        row_text = f"{ref:<7}{gt_distance:<9.2f}{yolo_distance:<10.2f}{best_iou:<8.2f}"
        cv2.rectangle(image, (x_start - 5, y_pos - 15), (x_start + 275, y_pos + 5), bg_color, -1)
        cv2.putText(image, row_text, (x_start, y_pos), font, font_scale, text_color, font_thickness)

    return image

# Function to calculate precision and recall
def calculate_precision_recall(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall

# Modify the code inside the image processing loop to display precision and recall
output_file = os.path.join(output_folder, f"Task_2.txt")
with open(output_file, 'w') as file:
    file.write("Image,\tRef,\tGT_Dist (m),\t\t YOLO_Dist (m),\t\tIoU,Confidence, Precision, Recall\n") 
    
    # Initialize overall counters for TP, FP, FN
    overall_true_positives = 0
    overall_false_positives = 0
    overall_false_negatives = 0
    
    # Process each image
    for idx, file_name in enumerate(os.listdir(image_folder)):
        # File paths
        base_name = os.path.splitext(file_name)[0]
        image_file = os.path.join(image_folder, file_name)
        calibration_file = os.path.join(f"KITTI_Selection/calib/", f"{base_name}.txt")
        ground_truth_file = os.path.join(f"KITTI_Selection/labels/", f"{base_name}.txt")
        image = cv2.imread(image_file)

        if image is None:
            raise ValueError(f"Failed to load image '{image_file}'")
        
        # Run inference on the image
        results = model(image, classes=[2])

        # Process YOLO detections
        yolo_boxes = []
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy.numpy()[0])
            yolo_boxes.append({"bbox": [x_min, y_min, x_max, y_max]})
            # Draw YOLO bounding box in red
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Read ground truth data
        ground_truth_boxes = read_ground_truth(ground_truth_file)
        for gt_box in ground_truth_boxes:
            x_min, y_min, x_max, y_max = map(int, gt_box["bbox"])
            # Draw ground truth bounding box in green
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Calculate YOLO distances and match with ground truth
        legend_data = []
        true_positives = 0
        false_positives = 0
        false_negatives = len(ground_truth_boxes)

        for i, yolo_box in enumerate(yolo_boxes):
            x_min, y_min, x_max, y_max = yolo_box["bbox"]
            points = [
                
                (x_min, y_max),
                (x_max, y_max),
                ((x_min + x_max) / 2, y_max)
                
            ]

            # Calculate the YOLO distance
            min_distance = float('inf')

            K = read_calibration(calibration_file)
            
            # For each point in the YOLO box, calculate the distance
            for px, py in points:
                z1, z2 = int(px), int(py)
                Pixel_vector = np.array([z1, z2, 1]).reshape(3, 1)
                K_inv = np.linalg.inv(K)
                intermediate_result = np.dot(K_inv, Pixel_vector)
                cent_dist = 1.65 / intermediate_result[1, 0]
                cent_dist = abs(cent_dist)
                if cent_dist < min_distance:
                    min_distance = cent_dist

            # Match with ground truth boxes using IoU
            best_iou = 0.4
            matched_gt_distance = None
            matched_gt_box = None
            for gt_box in ground_truth_boxes:
                iou = calculate_iou(yolo_box["bbox"], gt_box["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_distance = gt_box["distance"]
                    matched_gt_box = gt_box

            # If IoU is greater than a threshold, consider it a match
            if best_iou >= 0.0 and matched_gt_distance is not None:
                true_positives += 1
                false_negatives -= 1
                legend_data.append((i + 1, matched_gt_distance, min_distance, best_iou))
                # Add reference number and IoU to the image
                ref_text = f"{i + 1}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max((y_max - y_min) / 300, 0.3)
                font_scale = min(font_scale,0.5)
                font_thickness = max(int((y_max - y_min) / 75), 1)
                text_color = (255, 255, 255)  # Yellow text
                bg_color = (0, 0, 0)

                h,w,_ = image.shape
                
                if x_min <= 5:
                    x_min = x_min + 10

                if x_max >= w-5:
                    x_max = x_max - 10

                if y_min <= 5:
                    y_min = y_min + 10

                if y_max >=h-5:
                    y_max = y_max - 10
                

                # Add reference number
                (text_width, text_height), basis = cv2.getTextSize(ref_text, font, font_scale, font_thickness)
                cv2.rectangle(image, (x_min, y_min - int(2 * text_height)), (x_min + int (1.1 * text_width), y_min), bg_color, -1)
                cv2.putText(image, ref_text, (x_min, y_min - basis), font, font_scale, text_color, font_thickness)

            # If no match is found, it's a false positive
            if matched_gt_box is None:
                false_positives += 1

            # Write output to the text file
            output_line = f"{idx + 1},\t{(i + 1)},\t{matched_gt_distance if matched_gt_distance else 'N/A'},\t{min_distance},\t{best_iou}\n"
            file.write(output_line)

        # Calculate precision and recall for the current image
        precision, recall = calculate_precision_recall(true_positives, false_positives, false_negatives)
        
        # Write the calculated precision and recall to the file
        file.write(f"Precision: {precision:.2f}, Recall: {recall:.2f}\n")
        
        # Update overall counters for precision and recall
        overall_true_positives += true_positives
        overall_false_positives += false_positives
        overall_false_negatives += false_negatives

        # Calculate the center of the image to display precision and recall
        h, w, _ = image.shape
        text = f"Precision: {precision:.2f}, Recall: {recall:.2f}"

        # Get the size of the text to be displayed
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Calculate the position to center the text
        x = (w - text_size[0]) // 2
        y = h - 20  # Near the bottom (adjust as needed)

        # Draw a black rectangle behind the text to create a background
        cv2.rectangle(image, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)

        # Display the precision and recall in the center of the image
        cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), font_thickness)

        # Draw the legend on the image
        image_with_legend = draw_legend(image, legend_data)

        # Save the final output
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, image_with_legend)

    # Calculate and write overall precision and recall
    overall_precision, overall_recall = calculate_precision_recall(overall_true_positives, overall_false_positives, overall_false_negatives)
    with open(output_file, 'a') as file:
        file.write(f"\nOverall Precision: {overall_precision:.2f}, Overall Recall: {overall_recall:.2f}")
