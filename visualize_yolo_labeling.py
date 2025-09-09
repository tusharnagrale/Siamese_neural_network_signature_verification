import os
import cv2

# Paths
images_dir = '/Users/maitrithamke/Downloads/tushar/yolov11n_signature_detector/datasets/signature/images/train'
labels_dir = '/Users/maitrithamke/Downloads/tushar/yolov11n_signature_detector/datasets/signature/labels/train'
output_dir = '/Users/maitrithamke/Downloads/tushar/yolov11n_signature_detector/datasets/signature/images/annotated_images'

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Class names (customize as needed)
class_names = ['signature']  # Add more class labels as needed

# Supported image extensions
image_extensions = ['.jpg', '.png', '.jpeg']

# Helper: Find corresponding image file
def find_image_for_label(label_filename):
    base_name = os.path.splitext(label_filename)[0]
    for ext in image_extensions:
        image_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

# Loop through label files
for label_file in os.listdir(labels_dir):
    if not label_file.endswith('.txt'):
        continue

    label_path = os.path.join(labels_dir, label_file)
    image_path = find_image_for_label(label_file)

    if image_path is None:
        print(f"⚠️ Image not found for: {label_file}")
        continue

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to read image: {image_path}")
        continue

    height, width = image.shape[:2]

    # Draw bounding boxes from label
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, x_center, y_center, box_w, box_h = map(float, parts)
            x1 = int((x_center - box_w / 2) * width)
            y1 = int((y_center - box_h / 2) * height)
            x2 = int((x_center + box_w / 2) * width)
            y2 = int((y_center + box_h / 2) * height)

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{class_id}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save annotated image with same filename
    image_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, image)
    print(f"✅ Saved: {output_path}")
