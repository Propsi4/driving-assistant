from ultralytics import YOLO
from typing import List
from src.models.types.YOLOPrediction import YOLOPrediction
from src.config.settings import settings
from PIL import Image
import torch


class YOLOModel:
    """
    Class representing the YOLO model for object detection.
    """

    def __init__(self,
                 weights_path: str = settings.obj_detect_weights_path):
        """
        Initializes the YOLO model.
        """
        self.model = YOLO(
            model=weights_path,
            task='detect'
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def detect_traffic_signs(self, image: Image, confidence_threshold: float = 0.5) -> List[YOLOPrediction]:
        """
        Predicts the traffic signs in the given image.

        Parameters
        ----------
        image: Image
            The image to predict the traffic signs in.
        confidence_threshold: float
            The confidence threshold for the predictions.

        Returns
        -------
        List[YOLOPrediction]
            The list of predicted traffic signs.
        """
        predictions = self.model.predict(image, device=self.device)[0].boxes
        yolo_predictions = []

        for pred in predictions:
            pred = pred.cpu().numpy()
            for cls, conf, xywh in zip(pred.cls, pred.conf, pred.xywh):
                if conf > confidence_threshold:
                    yolo_prediction = YOLOPrediction(
                        class_id=int(cls),
                        confidence=conf,
                        x=xywh[0],
                        y=xywh[1],
                        w=xywh[2],
                        h=xywh[3],
                    )
                    yolo_predictions.append(yolo_prediction)
        return yolo_predictions
