from src.config.settings import settings
from pydantic import BaseModel, model_validator, Field
from typing import Optional
import requests as req
import json
from bs4 import BeautifulSoup
import re


class TrafficSign(BaseModel):
    """
    Class representing a traffic sign.

    Parameters
    ----------
    sign_code: str
        The code of the traffic sign according to the Ukrainian traffic sign classification.
    description: str
        A description of the traffic sign according to the Ukrainian traffic sign classification.
    category: str
        The category of the traffic sign according to the Ukrainian traffic sign classification.
    sign_image: str
        The image of the traffic sign.
    sign_code: str
        The code of the traffic sign according to the Ukrainian traffic sign classification.
    class_id: int
        The class ID of the traffic sign according to the training dataset.
    """

    category: str = Field(None)
    description: str = Field(None)
    name: Optional[str] = Field(None)
    sign_image: Optional[str] = Field(None)
    sign_code: Optional[str] = Field(None)
    class_id: Optional[int] = Field(None)
    category_mapping: dict = Field(None)
    sign_info_url_template: str = Field(None)
    sign_image_url_template: str = Field(None)

    @model_validator(mode='before')
    def validate_sign(cls, values):
        class_id = values.get('class_id')
        sign_code = values.get('sign_code')
        if not class_id and not sign_code:
            raise ValueError("Either 'class_id' or 'sign_code' must be provided.")

        if class_id and class_id not in values['category_mapping'].keys():
            raise ValueError(f'Invalid class ID: {class_id}')
        if sign_code and sign_code not in values['category_mapping'].values():
            raise ValueError(f'Invalid sign code: {sign_code}')

        return values

    def __init__(self, class_id: Optional[int] = None, sign_code: Optional[str] = None,
                 category_mapping=json.load(open(settings.category_mapping_path, 'r')),
                 sign_info_url_template=settings.sign_info_url_template,
                 sign_image_url_template=settings.sign_image_url_template):
        """
        Initializes the traffic sign with either `class_id` or `sign_code`.

        Parameters
        ----------
        class_id: Optional[int]
            The class ID of the traffic sign (optional).
        sign_code: Optional[str]
            The sign code of the traffic sign (optional).

        Raises
        ------
        ValueError
            If neither `class_id` nor `sign_code` is provided.
        """

        category_mapping = {int(k): v for k, v in category_mapping.items()}
        super().__init__(class_id=class_id, sign_code=sign_code,
                         category_mapping=category_mapping,
                         sign_info_url_template=sign_info_url_template,
                         sign_image_url_template=sign_image_url_template)

        # Convert the keys of the category mapping to integers

        if class_id:
            self.sign_code = category_mapping[class_id]
        if sign_code:
            self.class_id = [k for k, v in category_mapping.items() if v == sign_code][0]

        self.load_sign_info()

    def _filter_markdown(self, text):
        '''
        Removes markdown syntax like \\n, asterisks, and square brackets and etc.
        '''
        # Regex pattern to match markdown syntax like \\n, asterisks, and square brackets
        pattern = r'[\[\]\*]'
        return re.sub(pattern, '', text).replace('\\n', ' ').strip()

    def _filter_sign_code(self, text):
        """
        Removes sign codes (e.g., '5.1.2', '1.1') from the beginning of the text.

        Parameters
        ----------
        text : str
            Input text containing sign code and description.

        Returns
        -------
        str
            Text with the sign code removed.
        """
        # Regex pattern to match sign codes like '5.1.2' or '1.1'
        text = self._filter_markdown(text)
        pattern = r'^\d+(\.\d+)*\s+'
        return re.sub(pattern, '', text)

    def load_sign_info(self):
        '''
        Fetches the description and image of the traffic sign.
        '''
        sign_category = self.sign_code.split('.')[0]
        sign_info_url = self.sign_info_url_template.format(category=sign_category, sign_code=self.sign_code)
        response = req.get(sign_info_url)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Fetch the description
        try:
            description = self._filter_markdown(soup.find('div', class_='mark_markpage_block').find('p').text)
        except AttributeError:
            description = None

        # Fetch the image source
        try:
            img_source = soup.find('div', class_='contain_mar').find('img')['src'][1:]
        except AttributeError:
            img_source = None

        # Fetch the category
        try:
            category = self._filter_markdown(soup.find('div', class_='title_pdr').find('h1').text)
        except AttributeError:
            category = None

        # Fetch the sign name
        try:
            name = self._filter_sign_code(soup.find('div', class_='mark-markpage').find('h2').text)
        except AttributeError:
            name = None

        self.name = name
        self.sign_image = self.sign_image_url_template.format(image_source=img_source)
        self.description = description
        self.category = category
