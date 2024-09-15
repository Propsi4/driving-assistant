'''
Configurations for the Object Detection Model
'''

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
import gdown


class Settings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file='.env',
        description='Project configuration'
    )

    # Model Parameters
    obj_detect_weights_path: str = Field('src/weights/traffic_signs_detection.pt', alias='OBJ_DETECT_WEIGHTS_PATH', description='Path to the model weights')
    category_mapping_path: str = Field('src/config/class_to_sign.json', alias='CATEGORY_MAPPING_PATH', description='Path to the category mapping json file')
    confidence_threshold: float = Field(0.5, alias='CONFIDENCE_THRESHOLD', description='Confidence threshold for detections')
    max_tokens: int = Field(100, alias='MAX_TOKENS', description='Maximum number of tokens to generate')
    temperature: float = Field(0, alias='TEMPERATURE', description='Temperature for completition generation')

    # Image Parameters
    image_width: int = Field(640, alias='IMAGE_WIDTH', description='Width of the input image')
    image_height: int = Field(640, alias='IMAGE_HEIGHT', description='Height of the input image')

    # Server Parameters
    host: str = Field('0.0.0.0', alias='API_HOST', description='Host for the FastAPI server')
    port: int = Field(8000, alias='API_PORT', description='Port for the FastAPI server')

    # API Credentials
    fireworks_api_key: str = Field(alias='FIREWORKS_API_KEY', description='API key for the Fireworks LLM API')

    # Available LLMs
    available_llms: tuple = Field(
        (
            "llama-v3p1-405b-instruct",
            "llama-v3p1-70b-instruct",
            "llama-v3p1-8b-instruct",
            "llama-v3-70b-instruct",
            "mixtral-8x22b-instruct",
            "mixtral-8x7b-instruct"
        ),
        alias='AVAILABLE_LLMS',
        description='List of supported LLMs for Fireworks API'
    )

    # Additional Parameters
    sign_info_url_template: str = Field('https://vodiy.ua/znaky/{category}/{sign_code}', description='URL template for searching traffic sign information')
    sign_image_url_template: str = Field('https://vodiy.ua/{image_source}', description='URL template for searching traffic sign images')


settings = Settings()

if not os.path.exists(settings.obj_detect_weights_path):
    print(f"Downloading object detection weights to {settings.obj_detect_weights_path}")
    os.makedirs(os.path.dirname(settings.obj_detect_weights_path), exist_ok=True)
    gdown.download('https://drive.google.com/file/d/1q9M9w4r16Bp7T6wh-lHXJPnCcA1rSNF-/view?usp=sharing', settings.obj_detect_weights_path, quiet=False, fuzzy=True)
