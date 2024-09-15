from src.models.types.BBox import BBox, IsNormalizedBBoxArgType
from src.models.types.TrafficSign import TrafficSign
from pydantic import field_validator, Field, BaseModel


class YOLOPrediction(BaseModel):
    """
    Class representing a YOLO prediction.

    Parameters
    ----------
    class_id: int
        Class ID of the predicted object.
    confidence: float
        Confidence score of the prediction.
    x: float
        X-coordinate of the bounding box.
    y: float
        Y-coordinate of the bounding box.
    w: float
        Width of the bounding box.
    h: float
        Height of the bounding box.
    normalized_bbox: bool
        Flag indicating whether the bounding box coordinates are normalized.
    category: str
        The category of the traffic sign according to the Ukrainian traffic sign classification.
    description: str
        A description of the traffic sign according to the Ukrainian traffic sign classification.
    sign_image: str
        The URL of the image of the traffic sign.
    sign_code: str
        The code of the traffic sign according to the training dataset.
    """

    confidence: float = Field(None)
    traffic_sign: TrafficSign = Field(None, description="Traffic sign information")
    bbox: BBox = Field(None, description="Bounding box information")

    @field_validator('confidence')
    def validate_confidence(cls, value):
        if value < 0 or value > 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return value

    def __init__(self, class_id: int, confidence: float,
                 x: float, y: float, w: float, h: float,
                 is_normalized_bbox: IsNormalizedBBoxArgType = 'auto'):
        '''
        Initializes the YOLO prediction with the given parameters.

        Parameters
        ----------
        class_id: int
            Class ID of the predicted object.
        confidence: float
            Confidence score of the prediction.
        x: float
            X-coordinate of the bounding box.
        y: float
            Y-coordinate of the bounding box.
        w: float
            Width of the bounding box.
        h: float
            Height of the bounding box.
        is_normalized_bbox: NormalizedBBox(bool or str "auto")
            Flag indicating whether the bounding box coordinates are normalized.
            If set to 'auto', the bounding box coordinates are checked to determine if they are normalized.
        '''
        super().__init__(confidence=confidence)
        self.traffic_sign = TrafficSign(class_id=class_id)
        self.bbox = BBox(x=x, y=y, w=w, h=h, is_normalized_bbox=is_normalized_bbox)

    def __getattr__(self, name):
        # Forward property access to the nested objects if they exist
        if hasattr(self.traffic_sign, name):
            return getattr(self.traffic_sign, name)
        if hasattr(self.bbox, name):
            return getattr(self.bbox, name)

        # Optionally handle the case where the property is not found
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
