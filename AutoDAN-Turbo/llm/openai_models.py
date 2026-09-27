import json
import logging
from openai import OpenAI, AzureOpenAI


class OpenAIEmbeddingModel:
    def __init__(
        self,
        azure: bool = False,
        azure_endpoint: str = None,
        azure_api_version: str = "2024-02-01",
        azure_deployment_name: str = None,
        azure_api_key: str = None,
        openai_api_key: str = None,
        embedding_model: str = "text-embedding-3-small",
        logger: logging.Logger = None
    ):
        """
        Unified Embedding model wrapper for both OpenAI API and Azure OpenAI.
        Compatible with openai>=1.0.0
        """
        self.azure = azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment_name = azure_deployment_name
        self.embedding_model = embedding_model
        self.logger = logger or logging.getLogger(__name__)

        if self.azure:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint
            )
        else:
            self.client = OpenAI(api_key=openai_api_key)

    def encode(self, text):
        """
        Generate embeddings for a single string or list of strings.
        Returns list of embeddings or one embedding for single input.
        """
        try:
            single_input = isinstance(text, str)
            if single_input:
                text = [text]

            if self.azure:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.azure_deployment_name
                )
                data_json = response.model_dump_json(indent=2)
                data = json.loads(data_json).get("data", [])
                data = sorted(data, key=lambda x: x["index"])
                embeddings = [d["embedding"] for d in data]

            else:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embeddings = [item.embedding for item in response.data]

            if single_input and len(embeddings) == 1:
                return embeddings[0]
            return embeddings

        except Exception as e:
            if self.logger:
                self.logger.error(f"Embedding error: {e}", exc_info=True)
            else:
                print(f"[EmbeddingModel ERROR] {e}")
            return None