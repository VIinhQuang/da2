from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import os
import time
from threading import Thread
from imutils.video import VideoStream
app = Flask(__name__)

recording_in_progress = False

def e_dist(pA, pB):
    return np.linalg.norm(pA - pB)

def eye_ratio(eye):
    d_V1 = e_dist(eye[1], eye[5])
    d_V2 = e_dist(eye[2], eye[4])
    d_H = e_dist(eye[0], eye[3])
    eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)
    return eye_ratio_val

class Webcam():
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        self.face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.left_eye_start, self.left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_start, self.right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.eye_ratio_threshold = 0.25
        self.max_sleep_frames = 16
        self.sleep_frames = 0
        time.sleep(1.0)

    def detect_sleeping(self, frame):
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmark = self.landmark_detect(gray, rect)
            landmark = face_utils.shape_to_np(landmark)
            leftEye = landmark[self.left_eye_start:self.left_eye_end]
            rightEye = landmark[self.right_eye_start:self.right_eye_end]
            left_eye_ratio = eye_ratio(leftEye)
            right_eye_ratio = eye_ratio(rightEye)
            eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

            left_eye_bound = cv2.convexHull(leftEye)
            right_eye_bound = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [left_eye_bound], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_bound], -1, (0, 255, 0), 1)

            if eye_avg_ratio < self.eye_ratio_threshold:
                self.sleep_frames += 1
                if self.sleep_frames >= self.max_sleep_frames:
                    cv2.putText(frame, "BUON NGU THI DI NGU DI ONG OI!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.sleep_frames = 0
                cv2.putText(frame, "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

    def get_frame(self):
        if not self.vid.isOpened():
            return

        while True:
            _, img = self.vid.read()
            img = imutils.resize(img, width=640)
            frame_with_alert = self.detect_sleeping(img)

            # Encoding the frame in JPG format and yielding the result
            yield cv2.imencode('.jpg', frame_with_alert)[1].tobytes()
webcam = Webcam()

@app.route("/")
def index():
    return render_template("index.html")


def read_from_webcam():
    while True:
        # Read image from the Webcam class
        image = next(webcam.get_frame())

        # Perform YOLO object detection here

        # Yield the frame as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')
@app.route("/image_feed")
def image_feed():
    return Response(read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
