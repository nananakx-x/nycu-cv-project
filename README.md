# nycu-cv-project
| Output Videos |
|---------------|
| **NOTE:** All output videos can be found in the folder 'sample_output_videos' [here](https://drive.google.com/drive/folders/1cbEbHUKzyUglIhAzeMJcvnM_Uz0-RWFF?usp=sharing) |
---

## Set-up

YOLO models can be downloaded with PyTorch or using this [guide] (https://www.kaggle.com/code/nisafatima/object-detection-using-yolo).

---

## Report

### Introduction

Through this project, we aim to explore various methodologies, compare their effectiveness, and highlight the challenges we encountered while implementing them.

The first part of our presentation focuses on our own implementation, where we will guide you through the procedures, experimental results, and discussion as we progress from simple tracking methods to advanced models like YOLO and DeepSORT. 

In the second part, we will analyze findings from research papers, highlight their strengths and limitations, and identify ideal solutions. The reports used are [A New Multimodal Map Building Method Using Multiple Object Tracking and Gaussian Process Regression] (https://www.mdpi.com/2072-4292/16/14/2622) and [A novel real-time multiple objects detection and tracking framework for different challenges] (https://www.sciencedirect.com/science/article/pii/S111001682200165X).

By the end, we aim to provide a comprehensive understanding of this field and inspire further exploration.

### Implementation Procedure

To begin, we will walk you through the implementation procedures we followed in our project, detailing the step-by-step progression of our approach. 

Starting with foundational methods for single-object tracking, we gradually expanded the complexity to incorporate multi-object tracking. We explored both manual labeling techniques and advanced deep learning-based models, such as YOLOv7 and YOLOv5, leveraging pre-trained models for improved efficiency. Finally, we implemented DeepSORT, a tracking framework that effectively integrates object detection and association techniques. 

Along the way, we encountered and addressed various challenges, which we will later on discuss.

1. **Simple single-object tracking with manual labelling**  
   *(test1.py)*

The first implementation we attempted was a simple single-object tracker with manual labeling. Before diving into the hands-on aspects of our project, we conducted extensive research and came across a straightforward yet practical beginner’s guide on Medium (https://medium.com/@khwabkalra1/object-tracking-2fe4127e58bf). This guide formed the foundation of our initial implementation. The code starts by initializing a KCF tracker using OpenCV’s cv2.TrackerKCF_create() function. The video file is loaded using cv2.VideoCapture(), and the user is prompted to manually select the object to track through cv2.selectROI(), where a bounding box is drawn around the object of interest. Once the object is selected, the tracker is initialized with the frame and bounding box using tracker.init(). In the tracking loop, the tracker is updated frame-by-frame using tracker.update(), and the bounding box is drawn on the object if tracking is successful. 

While it worked reasonably well, we observed several limitations during experimentation. (Click) For instance, if the labeled person was obstructed or blocked during the video, tracking would fail entirely. Additionally, the bounding box often lost precision, gradually drifting away from the person as the video progressed. Other than accuracy and precision, the main challenge with this initial approach is its lack of user-friendliness, as manual labeling is required. This process can be time-consuming and prone to human error, especially in videos with fast-moving or multiple objects. Furthermore, in practical scenarios, such as those involving real-world applications like surveillance or autonomous vehicles, there are often multiple objects to track simultaneously. 

As a result, focusing on a single object is insufficient. The limitations of tracking only one object at a time hinder its scalability and real-world applicability, where tracking multiple objects with varying speeds and directions is crucial. This led us to explore more advanced techniques to overcome these constraints and enhance our tracker.

2. **Simple multi-object tracking with manual labelling**  
   *(test2.py)*

Building on the limitations of the first implementation, we moved to our second version, which addresses some of the key drawbacks, particularly the need for manual labeling and the focus on a single object. In this updated version, we shifted to multi-object tracking, allowing us to track multiple objects simultaneously in a video.

Instead of manually selecting a single object to track, this approach enables the user to select multiple bounding boxes, which are then tracked independently throughout the video. We used OpenCV’s MultiTracker for this purpose, which allows us to add multiple individual trackers - in this case, KCF trackers - to track different objects at the same time. This version also improves the tracking experience by providing an ID label for each object, making it easier to distinguish between tracked items.

Furthermore, we optimized the tracking process by only updating the tracker every few frames, which reduces unnecessary computational load. These modifications significantly improve the scalability and usability of the tracking system, especially in scenarios where multiple objects need to be monitored simultaneously.

3. **Multi-object tracking with YOLOv7**  
   *(test3_yolo.py)*

In our third implementation, we took a step forward by integrating YOLOv7 for object detection, aiming to improve the accuracy and scalability of our tracker. 

Unlike version 2, which relied on manual bounding box selection and a simple multi-tracker setup, this version automates the detection process using YOLO to identify multiple objects in each frame. The YOLO model allows us to detect objects with higher precision and handle dynamic, real-world scenarios more effectively. We then paired this with the MultiTracker class to track these detected objects across frames.

One major change from version 2 is the use of YOLO’s object detection capabilities. In version 2, users had to manually select objects for tracking, limiting the tracker to a few objects that could be manually labeled at the start. With YOLOv7, we now automatically detect objects, which allows for the tracking of multiple objects without manual intervention. 

The detection process involves using the YOLO model to generate predictions for class labels, confidence scores, and bounding box coordinates. After the detection, we applied non-max suppression (NMS) to eliminate redundant boxes and keep the most confident detections. 

However, we still faced challenges in maintaining accuracy, particularly when objects were obscured or when the detector misclassified objects. For instance, an umbrella that was far away was sometimes incorrectly labeled as a person, and in certain frames, not all people or objects were detected, leading to incomplete tracking.

To address some of these detection inaccuracies, we implemented a mechanism for adding new trackers dynamically when fresh detections appeared in the video stream. Using Intersection-over-Union, IoU, we matched new detections with existing trackers to ensure that objects already being tracked weren’t mistakenly re-tracked. This was important for cases where new objects entered the scene or when objects were occluded. 

Despite these improvements, challenges remain in optimizing tracking performance: Not all objects are reliably detected, and occasionally, the tracker would drift due to imperfect bounding box predictions or misclassifications. 

4. **Multi-object tracking with YOLOv5m**  
   *(mine_multiclass.py)*

Moving on to our fourth implementation - Version 4 introduces some improvements over Version 3, focusing primarily on enhanced accuracy and better handling of overlapping bounding boxes. 

In Version 3, the detection relied on YOLOv7 for object classification and tracking, which worked effectively in many scenarios but still faced issues such as misclassification and occasional overlapping boxes. These problems might be partly due to the YOLOv7's confidence threshold, which sometimes caused confusion when objects were too close together or poorly detected.

In Version 4, we used yolov5m, which offers better accuracy based on our experimental results. The new model is not only more precise in detecting objects but also less prone to misclassification. For example, umbrellas are now less likely to be mistakenly classified as people, and other objects benefit from more accurate labeling overall. Additionally, the implementation now uses a more sophisticated tracking mechanism, with the MultiTracker being paired with the more robust CSRT, Channel and Spatial Reliability Tracking, algorithm instead of the KCF tracker used in Version 3. This change helps reduce issues with overlapping boxes, ensuring that objects are tracked more consistently, even in crowded scenes.

Furthermore, Version 4 includes a periodic re-run of the YOLO detection every few frames to correct for instances where the tracker may drift. This mechanism helps reinitialize trackers and maintain their accuracy, even when objects temporarily disappear or become occluded. 

Overall, Version 4 not only provides better detection accuracy but also performs better in handling complex scenarios, such as overlapping objects or cases where some objects are initially undetected.

5. **Multi-object tracking using DeepSORT**  
   *(YOLO_DeepSort.py)*

​​While YOLO by itself provides accurate detections on a per-frame basis, it has no built-in mechanism to maintain consistent identities for each object across consecutive frames. As a result, an object in one frame might not be recognized as the same entity in the next. By combining YOLO with DeepSORT, the system gains temporal consistency - an object once identified is “followed” through occlusions or motion, preserving its track ID over time. This leads to a more informative and stable understanding of object behavior in a video (e.g., how many distinct objects there are, where each one moves, etc.), which pure detection (YOLO alone) cannot achieve. As such, we have decided to integrate DeepSORT together with YOLO for our final implementation.

This code integrates YOLOv5 (for object detection) with DeepSORT (for multi-object tracking) and plays the resulting tracked video at the source’s original frames per second. First, the script loads a YOLOv5 model (yolov5s) and configures detection thresholds. Next, it initializes DeepSORT, which uses a re-identification, ReID, model and Kalman filters to assign consistent IDs to each detected object across frames. The video capture reads frames from the specified file, and each frame is passed through YOLO to generate bounding boxes and class labels. These detections are then fed into DeepSORT, which tracks objects over time, returning bounding boxes and stable IDs. The code then draws these labeled boxes, including class names and IDs, on the video frames for display. To preserve realistic playback speed, it calculates the appropriate cv2.waitKey() delay based on the video’s original FPS.

### Experimental Results 

All output videos can be found in the folder 'sample_output_videos' [here](https://drive.google.com/drive/folders/1cbEbHUKzyUglIhAzeMJcvnM_Uz0-RWFF?usp=sharing) 

### Discussion

Aside from the challenges mentioned in the previous section, one issue we encountered was determining the most suitable YOLO model for our project. Based on our research, YOLOv7 was expected to outperform YOLOv5; however, our experimental results did not align with this expectation. YOLOv5 demonstrated better consistency and accuracy across our sample dataset videos. We hypothesize that this discrepancy might be due to factors such as dataset characteristics, model tuning, or implementation differences. Additionally, we tested YOLOv3, which performed poorly in comparison. Moving forward, we plan to explore more YOLO versions and fine-tune their configurations to identify the best fit.

Another problem that we have faced is that we were unable to preserve the video’s original FPS despite using the cv2.waitKey(). This means that while simply matching cv2.waitKey() to the nominal FPS may appear to solve playback-speed concerns, in reality the heavy computational demands of YOLO + DeepSORT can exceed the time budget per frame. As a result, the pipeline either drops frames or runs slower than the desired rate, so the displayed video no longer aligns perfectly with its original timing. Consequently, a more robust solution often involves adopting lighter or more optimized models (e.g., smaller YOLO architectures, hardware-accelerated backends), running computations asynchronously, or intentionally skipping frames to preserve fluid playback. By carefully balancing detection accuracy, computational overhead, and real-time display constraints, we can maintain a workflow closer to the source’s natural frame rate while still benefiting from reliable detection and tracking across the video.

### Conclusion

In summary, our project demonstrates how object detection and multi-object tracking can evolve from simple single-object trackers with manual labeling to fully automated pipelines using YOLO and DeepSORT. Throughout this progression, we faced several difficulties: manually labeling objects was time-consuming and prone to drift when occlusions occurred; detecting and tracking multiple objects introduced added complexity; and the heavy computational overhead of deep learning models often prevented preserving the video’s original FPS. By iterating on these challenges - fine-tuning detection thresholds, switching to more robust tracking frameworks, and balancing model size with hardware capabilities - we steadily improved accuracy and stability. Ultimately, this work highlights the importance of iterative refinements and the need to balance model complexity, real-time constraints, and practical requirements for reliable, real-world computer vision applications.


