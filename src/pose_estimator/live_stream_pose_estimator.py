import mediapipe as mp

from .base_pose_estimator import BaseEstimator
from .constant import LITE_MODEL_PATH
from typing import Any, Optional

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class LiveStreamPoseEstimator(BaseEstimator):
    def __init__(self, model_path: str = LITE_MODEL_PATH):
        super().__init__(model_path)
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._pose_landmarker_result_callback,
        )
        self.pose_landmarker_result: Optional[Any] = None

    def _pose_landmarker_result_callback(
        self, result: Any, input_image: mp.Image, timestamp_ms: int
    ):
        self.pose_landmarker_result = result

    def estimate_pose(self, input_image: mp.Image, timestamp_ms: int):
        with PoseLandmarker.create_from_options(self.options) as landmarker:
            landmarker.detect_async(input_image, timestamp_ms)

        return self.pose_landmarker_result, input_image
