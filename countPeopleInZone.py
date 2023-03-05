from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import supervision as sv
from utils import process_frame

#os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

#cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("./Videos/subway1.mp4") # For video
aboutDeveloper = cv2.imread("static/about_developer.png", cv2.IMREAD_UNCHANGED)
aboutDeveloper = cv2.resize(aboutDeveloper, (300, 90))

model  = YOLO("./YoloWeights/yolov8l.pt") #large model works better with the GPU
frame_resolution_wh = (1280, 720)
polygon = np.array([
    [500, 720],
    [640, 720],
    [780, 2],
    [750, 2]
])

zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution_wh)

# initiate annotator
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red(), thickness=4, text_thickness=4, text_scale=3)


# For continuous streaming
# while True:
#     success, frame = cap.read()
#     #frame = cv2.resize(img, frame_resolution_wh)
#     frame = process_frame(model, frame, zone, zone_annotator)

#     frame = cvzone.overlayPNG(frame, aboutDeveloper, (980, 610))

#     cv2.imshow('Subway Monitoring', frame)
#     cv2.waitKey(1)

# # For Saving video file
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    detections = detections[detections.class_id == 0]
    zone.trigger(detections=detections)

    # annotate
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1, color=sv.Color.white(), text_padding=1)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)
    frame = cvzone.overlayPNG(frame, aboutDeveloper, (980, 610))

    return frame

sv.process_video(source_path="./Videos/subway1.mp4", target_path=f"./Videos/subway-result.mp4", callback=process_frame)
