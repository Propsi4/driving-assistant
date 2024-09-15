from src.models.LLM import LLM
from src.models.YOLOModel import YOLOModel
from PIL import Image


class DrivingAssistant:
    """
    Class representing the Driving Assistant.
    """

    def __init__(self, llm_model_name: str = 'llama-v3p1-405b-instruct'):
        """
        Initializes the Driving Assistant.

        Parameters
        ----------
        obj_detect_weights_path: str
            The path to the object detection model weights.
        use_gpu: bool
            Flag indicating whether to use GPU for inference.
        """
        self.yolo_model = YOLOModel()
        self.llm = LLM(llm_model_name=llm_model_name)

    def predict(self, image: Image, confidence_threshold: float = 0.5, llm_model_name: str = 'llama-v3p1-405b-instruct') -> str:
        """
        Predicts the traffic signs in the given image and generates text based on the detected road signs.

        Parameters
        ----------
        image: str
            The image to predict the traffic signs in.

        Returns
        -------
        str
            The generated text.
        """
        road_signs = self.yolo_model.detect_traffic_signs(image, confidence_threshold=confidence_threshold)
        return self.llm.get_driving_hints(road_signs, llm_model_name=llm_model_name)
