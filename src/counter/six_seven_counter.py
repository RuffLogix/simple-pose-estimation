import math
import numpy as np
import mediapipe as mp


class SixSevenCounter:
    def __init__(self):
        self.count = 0
        self.state = (
            0  # 0 = no arms up, 1 = left arm up, 2 = right arm up, 3 = both arms up
        )
        self.prev_left = False
        self.prev_right = False

    def _calculate_angle(self, shoulder, elbow, wrist):
        a = np.array(shoulder)
        b = np.array(elbow)
        c = np.array(wrist)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def _is_left_arm_up(self, landmarks):
        left_shoulder = (
            landmarks[mp.tasks.vision.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp.tasks.vision.PoseLandmark.LEFT_SHOULDER].y,
        )
        left_elbow = (
            landmarks[mp.tasks.vision.PoseLandmark.LEFT_ELBOW].x,
            landmarks[mp.tasks.vision.PoseLandmark.LEFT_ELBOW].y,
        )
        left_wrist = (
            landmarks[mp.tasks.vision.PoseLandmark.LEFT_WRIST].x,
            landmarks[mp.tasks.vision.PoseLandmark.LEFT_WRIST].y,
        )

        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)

        if left_angle > 180:
            left_angle = 360 - left_angle

        return left_angle < 90

    def _is_right_arm_up(self, landmarks):
        right_shoulder = (
            landmarks[mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER].y,
        )
        right_elbow = (
            landmarks[mp.tasks.vision.PoseLandmark.RIGHT_ELBOW].x,
            landmarks[mp.tasks.vision.PoseLandmark.RIGHT_ELBOW].y,
        )
        right_wrist = (
            landmarks[mp.tasks.vision.PoseLandmark.RIGHT_WRIST].x,
            landmarks[mp.tasks.vision.PoseLandmark.RIGHT_WRIST].y,
        )

        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

        if right_angle > 180:
            right_angle = 360 - right_angle

        return right_angle < 90

    def _update_state(self, left_arm_up, right_arm_up):
        if left_arm_up and not self.prev_left:
            self.state |= 1 << 0
        if right_arm_up and not self.prev_right:
            self.state |= 1 << 1

        if self.state == 3:
            self.count += 1
            self.state = 0

        self.prev_left = left_arm_up
        self.prev_right = right_arm_up

    def update(self, pose_landmarker_result):
        if not pose_landmarker_result.pose_landmarks:
            return self.count

        landmarks = pose_landmarker_result.pose_landmarks[0]

        left_arm_up = self._is_left_arm_up(landmarks)
        right_arm_up = self._is_right_arm_up(landmarks)

        self._update_state(left_arm_up, right_arm_up)

        print(
            f"Count: {self.count}, State: {self.state}, Left Arm Up: {left_arm_up}, Right Arm Up: {right_arm_up}"
        )

        return self.count
