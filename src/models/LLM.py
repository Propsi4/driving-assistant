from typing import List, Optional
from pydantic import BaseModel, Field, PrivateAttr, SecretStr
from langchain_core.prompts.prompt import PromptTemplate
from src.models.prompts.templates import MAIN_PROMPT_TEMPLATE
from langchain_core.runnables.base import RunnableSequence
from langchain_fireworks import Fireworks
from src.models.types.TrafficSign import TrafficSign
from src.config.settings import settings


class ModelNotAvailableError(Exception):
    pass


class LLM(BaseModel):
    """
    Class representing a Language Model (LLM) for generating text.
    """

    _llm: Optional[Fireworks] = PrivateAttr(None)
    _llm_chain: Optional[RunnableSequence] = PrivateAttr(None)
    _api_key: SecretStr = PrivateAttr(None)

    llm_model_name: str = Field(None)
    prompt: PromptTemplate = Field(None)
    max_tokens: int = Field(None)
    available_llms: List[str] = Field(None)
    temperature: float = Field(None)

    def __init__(self, llm_model_name: str, api_key: str = settings.fireworks_api_key,
                 prompt_template: str = MAIN_PROMPT_TEMPLATE,
                 available_llms: List[str] = settings.available_llms,
                 max_tokens: int = settings.max_tokens, temperature: float = settings.temperature):
        """
        Initializes the Language Model (LLM) with the specified name.

        Parameters
        ----------
        llm_name: str
            The name of the Language Model (LLM).
        api_key: str
            The API key for the Fireworks API.
        prompt_template: str
            The template for the prompt.
        available_llms: List[str]
            The list of available LLMs.
        max_tokens: int
            The maximum number of tokens to generate.
        temperature: float
            The temperature for completition generation.
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=['road_signs'])
        super().__init__(llm_model_name=llm_model_name,
                         prompt=prompt, available_llms=available_llms,
                         max_tokens=max_tokens, temperature=temperature)

        self._api_key = api_key

        self._init_llm(llm_model_name=llm_model_name)

    def _init_llm(self, llm_model_name: str):
        """
        Initializes the Language Model (LLM) with the specified name.

        Parameters
        ----------
        llm_model_name: str
            The name of the Language Model (LLM).
        """
        if llm_model_name not in self.available_llms:
            raise ModelNotAvailableError(f"Provided model name {llm_model_name} is not supported. Please choose from {self.available_llms}")

        llm = Fireworks(model=f"accounts/fireworks/models/{llm_model_name}", fireworks_api_key=self._api_key,
                        temperature=self.temperature, max_tokens=self.max_tokens)

        llm_chain = self.prompt | llm

        self._llm = llm
        self._llm_chain = llm_chain
        self.llm_model_name = llm_model_name

    def _format_input(self, road_signs: List[TrafficSign]) -> str:
        """
        Formats the input for the Language Model (LLM).

        Parameters
        ----------
        road_signs: List[TrafficSign]
            The list of road signs to format.

        Returns
        -------
        str
            The formatted input for the Language Model (LLM).
        """
        road_sign_template = """Road sign {sign_code}:
                                    SIGN_NAME: {name}
                                    SIGN_CATEGORY: {category}
                                    SIGN_DESCRIPTION: {description}\n"""
        input_str = ""
        for road_sign in road_signs:
            input_str += road_sign_template.format(sign_code=road_sign.sign_code,
                                                   name=road_sign.name,
                                                   category=road_sign.category,
                                                   description=road_sign.description)
        return input_str

    def get_driving_hints(self, road_signs: List[TrafficSign], llm_model_name: Optional[str] = None) -> str:
        """
        Generates text completition for the given prompt.

        Parameters
        ----------
        road_signs: List[TrafficSign]
            The list of road signs to generate completition for.
        llm_model_name: Optional[str]
            The name of the Language Model (LLM) to use.

        Returns
        -------
        str
            The generated completition.
        """
        if llm_model_name and llm_model_name != self.llm_model_name:
            self._init_llm(llm_model_name=llm_model_name)

        completition = self._llm_chain.invoke({'road_signs': self._format_input(road_signs)})
        return completition
