import matplotlib.pyplot as plt
import math
import mediapipe as mp


class Visualizer:

    @staticmethod
    def visualize(image, pose_landmarker_result):
        h, w, _ = image.shape

        _, ax = plt.subplots()
        ax.imshow(image)

        for pose in pose_landmarker_result.pose_landmarks:
            for landmark in pose:
                x = math.ceil(landmark.x * w)
                y = math.ceil(landmark.y * h)

                ax.scatter(x, y, c="red", s=10)

            connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

            for connection in connections:
                start_idx = connection.start
                end_idx = connection.end

                start_landmark = pose[start_idx]
                end_landmark = pose[end_idx]

                x_start = int(start_landmark.x * w)
                y_start = int(start_landmark.y * h)

                x_end = int(end_landmark.x * w)
                y_end = int(end_landmark.y * h)

                ax.plot([x_start, x_end], [y_start, y_end], c="blue", linewidth=1)

        plt.show()
