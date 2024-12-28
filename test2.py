import cv2

multi_tracker = cv2.legacy.MultiTracker_create()

video = cv2.VideoCapture("./sample_test_videos/MOT17-02-SDP-raw.mp4")
success, frame = video.read()
if not success:
    print("Failed to read video")
    exit()

# Allow user to select multiple bounding boxes
while True:
    bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    if bbox == (0, 0, 0, 0):
        break
    tracker = cv2.legacy.TrackerKCF_create()
    multi_tracker.add(tracker, frame, bbox)

# Start tracking
frame_count = 0 
while True:
    success, frame = video.read()
    if not success:
        break

    frame_count += 1
    if frame_count % 1 != 0:
        continue

    success, boxes = multi_tracker.update(frame)
    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
