import cv2
import torch
import numpy as np

def main():
    # 1. Load a YOLO model (yolov5m) with higher confidence
    # You can specify 'yolov5s', 'yolov5m', 'yolov5l', etc.
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.conf = 0.6  # Confidence threshold
    model.iou = 0.45
    # model.classes = [0, 2]  # (Optional) detect multiple classes by indices; 
    # e.g. person=0, car=2. If you omit this, YOLO will detect all classes.

    video_path = "./sample_test_videos/MOT17-13-SDP-raw.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video!")
        return

    success, frame = cap.read()
    if not success:
        print("Failed to read video.")
        return

    # 2. Create MultiTracker
    multi_tracker = cv2.legacy.MultiTracker_create()

    # We'll keep a separate list to store the labels of each tracked box
    # because cv2.MultiTracker only returns bounding boxes, not class labels.
    tracker_labels = []

    def detect_and_init_trackers(_frame):
        """
        Runs YOLO on the frame, creates a new MultiTracker + label list.
        Returns (temp_multi_tracker, bboxes_temp, labels_temp).
        """
        temp_multi_tracker = cv2.legacy.MultiTracker_create()
        labels_temp = []

        # Detect objects with YOLO
        results = model(_frame)
        # results.xyxy[0] => [[x1, y1, x2, y2, conf, class], ...]

        bboxes_temp = []
        for *xyxy, conf, cls_id in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            w = x2 - x1
            h = y2 - y1

            # Filter out extremely small boxes if desired
            if w > 0 and h > 0:
                bboxes_temp.append((x1, y1, w, h))
                # Get class label name from YOLO model
                class_name = model.names[int(cls_id)]
                labels_temp.append(class_name)

        # For each bounding box, create a CSRT tracker and add to MultiTracker
        for bbox in bboxes_temp:
            tracker = cv2.legacy.TrackerCSRT_create()
            temp_multi_tracker.add(tracker, _frame, bbox)

        return temp_multi_tracker, bboxes_temp, labels_temp

    # First detection + tracker initialization
    multi_tracker, bboxes, tracker_labels = detect_and_init_trackers(frame)

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        # Update trackers
        success, tracked_boxes = multi_tracker.update(frame)

        # Optionally re-run YOLO every N frames to correct drift
        if frame_count % 30 == 0:
            multi_tracker, bboxes, tracker_labels = detect_and_init_trackers(frame)
            success, tracked_boxes = multi_tracker.update(frame)

        # Draw bounding boxes and label text
        # tracked_boxes and tracker_labels have the same order (by creation).
        for i, (box, label) in enumerate(zip(tracked_boxes, tracker_labels)):
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # You can display both class label and ID
            text = f"{label} (ID {i+1})"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
