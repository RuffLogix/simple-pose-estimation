from src.app import ImagePoseEstimator
from src.visualizer import Visualizer


def main():
    estimator = ImagePoseEstimator()

    pose_landmarker_result, mp_image = estimator.estimate_pose()

    Visualizer.visualize(mp_image.numpy_view(), pose_landmarker_result)


if __name__ == "__main__":
    main()
