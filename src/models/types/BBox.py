from pydantic import BaseModel, field_validator
import numpy as np
from typing import Union, Literal

IsNormalizedBBoxArgType = Union[bool, Literal['auto']]


class BBox(BaseModel):
    """
    Class representing a bounding box in YOLO format (x_center, y_center, width, height).

    Parameters
    ----------
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
    """

    x: float
    y: float
    w: float
    h: float
    normalized_bbox: bool = False

    @field_validator('x', 'y', 'w', 'h')
    def validate_bbox(cls, value, field):
        if not isinstance(value, (float)):
            raise TypeError(f"{field.field_name.capitalize()} must be a number")
        if value < 0:
            raise ValueError(f"{field.field_name.capitalize()} must be non-negative")
        return value

    def __init__(self, x: int, y: int, w: int, h: int, is_normalized_bbox: IsNormalizedBBoxArgType = 'auto'):
        '''
        Initializes the bounding box with the given coordinates.

        Parameters
        ----------
        x: int
            X-coordinate of the bounding box.
        y: int
            Y-coordinate of the bounding box.
        w: int
            Width of the bounding box.
        h: int
            Height of the bounding box.
        is_normalized_bbox: IsNormalizedBBoxArgType(bool or str "auto")
            Flag indicating whether the bounding box coordinates are normalized.
            If set to 'auto', the bounding box coordinates are checked to determine if they are normalized.
        '''
        super().__init__(x=x, y=y, w=w, h=h)
        _normalized_bbox = is_normalized_bbox if is_normalized_bbox != 'auto' else self._is_normalized_bbox()
        self.normalized_bbox = _normalized_bbox

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __getitem__(self, item):
        return (self.x, self.y, self.w, self.h)[item]

    def _is_normalized_bbox(self):
        '''
        Checks if the bounding box coordinates are normalized (i.e., in the range [0, 1]).

        Returns
        -------
        bool
            True if the bounding box coordinates are normalized, False otherwise.
        '''
        return np.all(np.array([self.x, self.y, self.w, self.h]) <= 1)

    def normalize_bbox(self, width: int, height: int) -> 'BBox':
        '''
        Normalizes the bounding box coordinates based on the given width and height.

        Parameters
        ----------
        width: int
            Width of the image.
        height: int
            Height of the image.

        Returns
        -------
        BBox

        Raises
        ------
        ValueError
            If the provided width and height do not match the bounding box coordinates.
        '''
        self.normalized_bbox = True
        self.x = self.x / width
        self.y = self.y / height
        self.w = self.w / width
        self.h = self.h / height

        if not self._is_normalized_bbox():
            raise ValueError("Provided width and height do not match the bounding box coordinates")
        return self

    def denormalize_bbox(self, width: int, height: int) -> 'BBox':
        '''
        Denormalizes the bounding box coordinates based on the given width and height.

        Parameters
        ----------
        width: int
            Width of the image.
        height: int
            Height of the image.

        Returns
        -------
        BBox
        '''
        self.normalized_bbox = False
        self.x = self.x * width
        self.y = self.y * height
        self.w = self.w * width
        self.h = self.h * height
        return self
