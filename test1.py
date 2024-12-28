import cv2

tracker = cv2.TrackerKCF_create()  # Initialize KCF tracker

video = cv2.VideoCapture("./sample_test_videos/MOT17-02-SDP-raw.mp4")  # Load video file

success, frame = video.read()
bbox = cv2.selectROI("Frame", frame, False)  # Select the object to track

tracker.init(frame, bbox)  # Initialize tracker with the initial bounding box

while True:
    success, frame = video.read()
    if not success:
        break
    
    success, bbox = tracker.update(frame)  # Update the tracker
    
    if success:
        # Draw bounding box on the object
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()