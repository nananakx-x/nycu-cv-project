import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("./yolo/yolov7.weights", "./yolo/yolov7.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# Load class labels
with open("./yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

multi_tracker = cv2.legacy.MultiTracker_create()

video = cv2.VideoCapture("./sample_test_videos/MOT17-02-SDP-raw.mp4")
success, frame = video.read()
if not success:
    print("Failed to read video")
    exit()

def non_max_suppression(boxes, confidences, threshold=0.4):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices]
    return []

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_overlap * y_overlap
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area

def get_yolo_bboxes(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    boxes = non_max_suppression(boxes, confidences, threshold=0.4)
    return boxes, class_ids, confidences

tracked_boxes = []

boxes, class_ids, confidences = get_yolo_bboxes(frame)

# Add detected YOLO boxes to the multi-tracker
for box in boxes:
    tracker = cv2.legacy.TrackerKCF_create()
    multi_tracker.add(tracker, frame, tuple(box))

frame_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    frame_count += 1

    success, boxes = multi_tracker.update(frame)

    if frame_count % 350 == 0:
        new_boxes, new_class_ids, confidences = get_yolo_bboxes(frame)
        for box, class_id in zip(new_boxes, new_class_ids):
            x, y, w, h = box
            already_tracked = False

            for tracked_box in tracked_boxes:
                iou_value = iou(tracked_box, box)
                if iou_value > 0.65:
                    already_tracked = True
                    break

            if not already_tracked:
                tracker = cv2.legacy.TrackerKCF_create()
                multi_tracker.add(tracker, frame, tuple(box))
                tracked_boxes.append(box)

        for box, class_id in zip(new_boxes, new_class_ids):
            x, y, w, h = box
            already_tracked = False
            for tracked_box in tracked_boxes:
                iou_value = iou(tracked_box, box)
                if iou_value > 0.5:
                    already_tracked = True
                    break
            if not already_tracked:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw boxes with class names
    for i, box in enumerate(boxes):
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, classes[class_ids[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
