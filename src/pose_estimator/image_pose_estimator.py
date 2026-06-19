import mediapipe as mp

from .base_pose_estimator import BaseEstimator
from .constant import LITE_MODEL_PATH, SAMPLE_IMAGE_PATH

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class ImagePoseEstimator(BaseEstimator):
    def __init__(self, model_path: str = LITE_MODEL_PATH):
        super().__init__(model_path)
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
        )

    def estimate_pose(self, image_path: str = SAMPLE_IMAGE_PATH):
        mp_image = mp.Image.create_from_file(image_path)

        with PoseLandmarker.create_from_options(self.options) as landmarker:
            pose_landmarker_result = landmarker.detect(mp_image)

        return pose_landmarker_result, mp_image
