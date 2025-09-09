import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # replace with your trained weights

# Confidence threshold
CONF_THRESHOLD = 0.5

# Input image
image_path = "/Users/maitrithamke/Downloads/tushar/input_images/pan3.png"  # replace with your image file
image = cv2.imread(image_path)

# Create output folder
output_dir = "/Users/maitrithamke/Downloads/tushar/yolov11n_signature_detector/testing_data/authorised_signature"
os.makedirs(output_dir, exist_ok=True)

# Extract base name without extension
base_name = os.path.splitext(os.path.basename(image_path))[0]

# Run YOLO inference
results = model(image_path)

# Process detections
count = 0
for box in results[0].boxes:
    conf = float(box.conf[0])  # confidence score

    if conf >= CONF_THRESHOLD:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the detected region
        cropped = image[y1:y2, x1:x2]

        # Convert to grayscale
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Save cropped image as authorise_signature_<base_name>_<count>.png
        count += 1
        output_path = os.path.join(output_dir, f"authorise_{base_name}_{count}.png")
        cv2.imwrite(output_path, gray_cropped)
        print(f"✅ Saved: {output_path} (Confidence: {conf:.2f})")

if count == 0:
    print("⚠ No detections above confidence threshold.")
else:
    print(f"🎯 {count} signature(s) saved in '{output_dir}' folder.")
