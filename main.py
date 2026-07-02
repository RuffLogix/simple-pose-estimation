import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from src.counter.six_seven_counter import SixSevenCounter
from src.pose_estimator.live_stream_pose_estimator import LiveStreamPoseEstimator
from src.visualizer import Visualizer

CAMERA_INDEX = 2

estimator = LiveStreamPoseEstimator()
visualizer = Visualizer()
counter = SixSevenCounter()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        frame_counter += 1

        # Skip every other frame to reduce processing load
        if frame_counter % 2 != 0:
            continue

        if not ret:
            break

        # Convert the frame to RGB and create a MediaPipe Image
        mp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)

        # Estimate the pose using the LiveStreamPoseEstimator
        pose_landmarker_result, _ = estimator.estimate_pose(mp_image, 0)

        # Visualize the pose landmarks on the frame using OpenCV
        frame = visualizer.visualize_cv(frame, pose_landmarker_result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Display the frame with pose landmarks
        cv2.imshow("Pose Estimation", frame)
        counter.update(pose_landmarker_result)


if __name__ == "__main__":
    main()
