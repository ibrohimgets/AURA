from ultralytics import YOLO
import cv2
import json

# Load YOLO model
model = YOLO('yolov8n.pt')  # Pre-trained YOLO model

# Load the image
image_path = r"C:\Users\User\Desktop\do2\dataSet\test\images\8_jpg.rf.baeb8c59a287bc06e4ddf36386cc57eb.jpg"  
image = cv2.imread(image_path)

# Run YOLO detection
results = model(image_path)

# Prepare data for JSON and draw bounding boxes
detected_objects = []

# Check if any detections were made
if results and results[0].boxes.data.shape[0] > 0:
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            label = result.names[int(class_id)]
            confidence = float(confidence)
            
            # Save detection details to JSON list
            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "bounding_box": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                }
            })
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label} ({confidence:.2f})",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
else:
    print("No objects detected by YOLO.")
    detected_objects.append({"message": "No objects detected."})
    cv2.putText(
        image,
        "No objects detected",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

# Save the image (even if empty)
output_image_path = 'output_with_boxes.jpg'
cv2.imwrite(output_image_path, image)
print(f"Image saved as {output_image_path}")

# Save detection details to a JSON file
output_json_path = 'yolo_results.json'
with open(output_json_path, 'w') as json_file:
    json.dump(detected_objects, json_file, indent=2)
print(f"Detection details saved as {output_json_path}")
