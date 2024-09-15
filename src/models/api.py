import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, status
from src.config.settings import settings
from src.models.DrivingAssistant import DrivingAssistant
from src.models.LLM import ModelNotAvailableError
from PIL import Image
import json


app = FastAPI()

driving_assistant = DrivingAssistant(llm_model_name='llama-v3p1-405b-instruct')


@app.get("/api/available_models")
def available_models():
    """
    Get the available LLM models.
    """

    return {"llm_models": settings.available_llms}


@app.get("/api/available_classes")
def available_classes():
    """
    Get the available traffic sign classes.
    """

    return {"classes": set([sign_code for sign_code in json.load(open(settings.category_mapping_path, 'r')).values()])}


@app.post("/api/predict")
async def predict(
    image: UploadFile,
    llm_model_name: str = "llama-v3p1-405b-instruct",
    confidence_threshold: float = 0.5
) -> dict:
    """
    Predicts the traffic signs in the given image and generates text based on the detected road signs.

    Parameters
    ----------
    image: UploadFile
        The image to predict the traffic signs in.
    llm_model_name: str
        The name of the LLM model to use.
    confidence_threshold: float
        The confidence threshold for the predictions.

    Raises
    ------
        HTTPException: If the image is not provided or the image type is invalid
    """

    content_type = image.content_type

    if content_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image not provided"
        )

    if content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image type"
        )

    image = Image.open(image.file)

    try:
        response = driving_assistant.predict(image, confidence_threshold=confidence_threshold, llm_model_name=llm_model_name)
    except ModelNotAvailableError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    return {'hints': response}


if __name__ == "__main__":
    uvicorn.run("src.models.api:app", host=settings.host, port=settings.port)
