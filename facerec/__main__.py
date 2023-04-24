import cv2
from facerec.src.face_recognition_webcam_all_classifier_updated import FaceRec

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(1)
    # video_capture = cv2.VideoCapture('rtsp://admin:admin123@10.10.5.117:554/cam/realmonitor?channel=1&subtype=0')
    video_capture.set(3, 1080)
    video_capture.set(4, 960)
    FaceRec(video_capture)
