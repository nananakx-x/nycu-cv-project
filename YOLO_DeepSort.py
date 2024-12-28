import cv2
import torch
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    # 1. Load YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4
    model.iou = 0.45

    # 2. Initialize deep_sort_realtime
    deepsort = DeepSort(
        max_age=70,
        n_init=3,
        max_cosine_distance=0.2,
        max_iou_distance=0.7,
        nn_budget=100,
        embedder_gpu=True
    )

    # 3. Open video/camera
    video_path = "./sample_test_videos/MOT17-13-SDP-raw.mp4"  # Replace with your own video path or 0 for webcam
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    # Query the original frames-per-second from the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        # Fallback if FPS is not available or is zero
        original_fps = 30  # Arbitrary default
    # Calculate delay (in milliseconds) per frame
    delay = int(1000 / original_fps)

    while True:
        success, frame = cap.read()
        if not success:
            print("No more frames or cannot read video.")
            break

        # 4. Run YOLO detection
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()

        # 5. Format detections for deep_sort_realtime
        detections = []
        for (x1, y1, x2, y2, conf, cls_id) in dets:
            w = x2 - x1
            h = y2 - y1
            bbox = [x1, y1, w, h]
            detections.append((bbox, conf, cls_id))

        # 6. Update DeepSort
        tracks = deepsort.update_tracks(detections, frame=frame)

        # 7. Draw bounding boxes & IDs
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            track_id = track.track_id
            cls_id = track.det_class  # The class ID

            # Convert track bounding box (ltrb)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convert YOLO class ID to text
            if int(cls_id) in model.names:
                label_str = model.names[int(cls_id)]
            else:
                label_str = str(cls_id)

            text = f"{label_str} (ID {track_id})"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO + DeepSort Realtime (Original FPS)", frame)

        # Use 'delay' (ms) derived from the original video FPS for playback
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
