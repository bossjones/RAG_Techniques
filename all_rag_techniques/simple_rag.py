#!/usr/bin/env python3

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# # References
#
# - https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# - https://docs.smith.langchain.com/tutorials/Developers/rag#evaluator

# %%
from __future__ import annotations

import os
import sys
import traceback
import typing

from typing import List

import bpdb
import rich

from langchain_core.documents import Document
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from loguru import logger as LOGGER


# SOURCE: https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# SOURCE: https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# TODO: Look at this https://github.com/h2oai/h2ogpt/blob/542543dc23aa9eb7d4ce7fe6b9af1204a047b50f/src/enums.py#L386 and see if we can add some more models
_TOKENS_PER_TILE = 170
_TILE_SIZE = 512

_OLDER_MODEL_CONFIG = {
    "gpt-4-0613": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006,
    },
    "gpt-4-32k-0314": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-4-32k-0613": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000003,
        "completion_cost_per_token": 0.000004,
    },
    "gpt-3.5-turbo-instruct": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
}

_NEWER_MODEL_CONFIG = {
    "gpt-4o-2024-08-06": {
        "max_tokens": 128000,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "gpt-4o-mini-2024-07-18": {
        # "max_tokens": 128000,
        "max_tokens": 900,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.000000150,
        "completion_cost_per_token": 0.00000060,
    },
    "gpt-4o-2024-05-13": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000005,
        "completion_cost_per_token": 0.000015,
    },
    "gpt-4-turbo-2024-04-09": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-0125-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-1106-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-vision-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-3.5-turbo-0125": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000005,
        "completion_cost_per_token": 0.0000015,
    },
    "gpt-3.5-turbo-1106": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000001,
        "completion_cost_per_token": 0.000002,
    },
}

_NEWER_EMBEDDING_CONFIG = {
    "text-embedding-3-small": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.00000002,
    },
    "text-embedding-3-large": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.00000013,
    },
}

_OLDER_EMBEDDING_CONFIG = {
    "text-embedding-ada-002": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.0000001,
    },
}


EMBEDDING_CONFIG = {}
EMBEDDING_CONFIG.update(_OLDER_EMBEDDING_CONFIG)
EMBEDDING_CONFIG.update(_NEWER_EMBEDDING_CONFIG)

MODEL_CONFIG = {}
MODEL_CONFIG.update(_OLDER_MODEL_CONFIG)
MODEL_CONFIG.update(_NEWER_MODEL_CONFIG)

MODEL_POINT = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",
    "gpt-4-vision": "gpt-4-vision-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
}

_MODEL_POINT_CONFIG = {
    "gpt-4o-mini": MODEL_CONFIG[MODEL_POINT["gpt-4o-mini"]],
    "gpt-4o": MODEL_CONFIG[MODEL_POINT["gpt-4o"]],
    "gpt-4-turbo": MODEL_CONFIG[MODEL_POINT["gpt-4-turbo"]],
    "gpt-4": MODEL_CONFIG[MODEL_POINT["gpt-4"]],
    "gpt-4-32k": MODEL_CONFIG[MODEL_POINT["gpt-4-32k"]],
    "gpt-4-vision": MODEL_CONFIG[MODEL_POINT["gpt-4-vision"]],
    "gpt-3.5-turbo": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo"]],
    "gpt-3.5-turbo-16k": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo-16k"]],
}

# contains all the models and embeddings info
MODEL_CONFIG.update(_MODEL_POINT_CONFIG)

# produces a list of all models and embeddings available
MODEL_ZOO = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys())

# SOURCE: https://github.com/JuliusHenke/autopentest/blob/ca822f723a356ec974d2dff332c2d92389a4c5e3/src/text_embeddings.py#L19
# https://platform.openai.com/docs/guides/embeddings/embedding-models
EMBEDDING_MODEL_DIMENSIONS_DATA = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 512,
    "text-embedding-3-large": 1024,
}

# MAX_TOKENS_DATA = MAX_TOKENS = {
#     'text-embedding-ada-002': 8000,
#     'gpt-3.5-turbo': 16000,
#     'gpt-3.5-turbo-0125': 16000,
#     'gpt-3.5-turbo-0613': 4000,
#     'gpt-3.5-turbo-1106': 16000,
#     'gpt-3.5-turbo-16k': 16000,
#     'gpt-3.5-turbo-16k-0613': 16000,
#     'gpt-4': 8000,
#     'gpt-4-0613': 8000,
#     'gpt-4-32k': 32000,
#     'gpt-4-1106-preview': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4-0125-preview': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4o': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4o-2024-05-13': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4-turbo-preview': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4-turbo-2024-04-09': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4-turbo': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4o-mini': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4o-mini-2024-07-18': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'gpt-4o-2024-08-06': 128000,  # 128K, but may be limited by config.max_model_tokens
#     'claude-instant-1': 100000,
#     'claude-2': 100000,
#     'command-nightly': 4096,
#     'replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1': 4096,
#     'meta-llama/Llama-2-7b-chat-hf': 4096,
#     'vertex_ai/codechat-bison': 6144,
#     'vertex_ai/codechat-bison-32k': 32000,
#     'vertex_ai/claude-3-haiku@20240307': 100000,
#     'vertex_ai/claude-3-sonnet@20240229': 100000,
#     'vertex_ai/claude-3-opus@20240229': 100000,
#     'vertex_ai/claude-3-5-sonnet@20240620': 100000,
#     'vertex_ai/gemini-1.5-pro': 1048576,
#     'vertex_ai/gemini-1.5-flash': 1048576,
#     'vertex_ai/gemma2': 8200,
#     'codechat-bison': 6144,
#     'codechat-bison-32k': 32000,
#     'anthropic.claude-instant-v1': 100000,
#     'anthropic.claude-v1': 100000,
#     'anthropic.claude-v2': 100000,
#     'anthropic/claude-3-opus-20240229': 100000,
#     'anthropic/claude-3-5-sonnet-20240620': 100000,
#     'bedrock/anthropic.claude-instant-v1': 100000,
#     'bedrock/anthropic.claude-v2': 100000,
#     'bedrock/anthropic.claude-v2:1': 100000,
#     'bedrock/anthropic.claude-3-sonnet-20240229-v1:0': 100000,
#     'bedrock/anthropic.claude-3-haiku-20240307-v1:0': 100000,
#     'bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0': 100000,
#     'claude-3-5-sonnet': 100000,
#     'groq/llama3-8b-8192': 8192,
#     'groq/llama3-70b-8192': 8192,
#     'groq/mixtral-8x7b-32768': 32768,
#     'groq/llama-3.1-8b-instant': 131072,
#     'groq/llama-3.1-70b-versatile': 131072,
#     'groq/llama-3.1-405b-reasoning': 131072,
#     'ollama/llama3': 4096,
#     'watsonx/meta-llama/llama-3-8b-instruct': 4096,
#     "watsonx/meta-llama/llama-3-70b-instruct": 4096,
#     "watsonx/meta-llama/llama-3-405b-instruct": 16384,
#     "watsonx/ibm/granite-13b-chat-v2": 8191,
#     "watsonx/ibm/granite-34b-code-instruct": 8191,
#     "watsonx/mistralai/mistral-large": 32768,
# }

# MAX_RETRIES: int = 9
# MAX_TOKENS: int = MAX_TOKENS_DATA[LLM_MODEL_NAME]

# 'gpt-4o-mini': {
#     'max_tokens': 128000,
#     'max_output_tokens': 16384,
#     'prompt_cost_per_token': 1.5e-07,
#     'completion_cost_per_token': 6e-07
# },


# %%
# rich.print(MODEL_ZOO)
# rich.print(MODEL_CONFIG)


# %%
# Top level vars


EVAL_MAX_CONCURRENCY = 4
LLM_MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE: int=1000
CHUNK_OVERLAP: int = 200
ADD_START_INDEX: bool = False
# LLM_EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
LLM_EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
DEFAULT_SEARCH_KWARGS: dict[str, int]={"k": 2}
QUESTION_TO_ASK = "What is the main cause of climate change?"
DATASET_NAME = "Climate Change Q&A"



MAX_TOKENS: int = MODEL_CONFIG[LLM_MODEL_NAME]["max_tokens"]
MAX_OUTPUT_TOKENS: int = MODEL_CONFIG[LLM_MODEL_NAME]["max_output_tokens"]
PROMPT_COST_PER_TOKEN: float = MODEL_CONFIG[LLM_MODEL_NAME]["prompt_cost_per_token"]
COMPLETION_COST_PER_TOKEN: float = MODEL_CONFIG[LLM_MODEL_NAME]["completion_cost_per_token"]
MAX_RETRIES: int = 9

rich.print(MAX_TOKENS)
rich.print(MAX_OUTPUT_TOKENS)
rich.print(PROMPT_COST_PER_TOKEN)
rich.print(COMPLETION_COST_PER_TOKEN)

EMBEDDING_MAX_TOKENS: int = EMBEDDING_MODEL_DIMENSIONS_DATA[LLM_EMBEDDING_MODEL_NAME]
EMBEDDING_MODEL_DIMENSIONS: int = EMBEDDING_MODEL_DIMENSIONS_DATA[LLM_EMBEDDING_MODEL_NAME]


def get_model_config(model_name: str = LLM_MODEL_NAME, embedding_model_name: str = LLM_EMBEDDING_MODEL_NAME):
    return {
        "model_name": model_name,
        "embedding_model_name": embedding_model_name,
        "max_tokens": MODEL_CONFIG[model_name]["max_tokens"],
        "max_output_tokens": MODEL_CONFIG[model_name]["max_output_tokens"],
        "prompt_cost_per_token": MODEL_CONFIG[model_name]["prompt_cost_per_token"],
        "completion_cost_per_token": MODEL_CONFIG[model_name]["completion_cost_per_token"],
        "embedding_max_tokens": EMBEDDING_CONFIG[embedding_model_name]["max_tokens"],
        "embedding_model_dimensions": EMBEDDING_MODEL_DIMENSIONS_DATA[embedding_model_name],
    }

LLM_RUN_CONFIG = get_model_config()
rich.print(LLM_RUN_CONFIG)

# %% [markdown]
# # Simple RAG (Retrieval-Augmented Generation) System
#
# ## Overview
#
# This code implements a basic Retrieval-Augmented Generation (RAG) system for processing and querying PDF documents. The system encodes the document content into a vector store, which can then be queried to retrieve relevant information.
#
# ## Key Components
#
# 1. PDF processing and text extraction
# 2. Text chunking for manageable processing
# 3. Vector store creation using [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) and OpenAI embeddings
# 4. Retriever setup for querying the processed documents
# 5. Evaluation of the RAG system
#
# ## Method Details
#
# ### Document Preprocessing
#
# 1. The PDF is loaded using PyPDFLoader.
# 2. The text is split into chunks using RecursiveCharacterTextSplitter with specified chunk size and overlap.
#
# ### Text Cleaning
#
# A custom function `replace_t_with_space` is applied to clean the text chunks. This likely addresses specific formatting issues in the PDF.
#
# ### Vector Store Creation
#
# 1. OpenAI embeddings are used to create vector representations of the text chunks.
# 2. A FAISS vector store is created from these embeddings for efficient similarity search.
#
# ### Retriever Setup
#
# 1. A retriever is configured to fetch the top 2 most relevant chunks for a given query.
#
# ### Encoding Function
#
# The `encode_pdf` function encapsulates the entire process of loading, chunking, cleaning, and encoding the PDF into a vector store.
#
# ## Key Features
#
# 1. Modular Design: The encoding process is encapsulated in a single function for easy reuse.
# 2. Configurable Chunking: Allows adjustment of chunk size and overlap.
# 3. Efficient Retrieval: Uses FAISS for fast similarity search.
# 4. Evaluation: Includes a function to evaluate the RAG system's performance.
#
# ## Usage Example
#
# The code includes a test query: "What is the main cause of climate change?". This demonstrates how to use the retriever to fetch relevant context from the processed document.
#
# ## Evaluation
#
# The system includes an `evaluate_rag` function to assess the performance of the retriever, though the specific metrics used are not detailed in the provided code.
#
# ## Benefits of this Approach
#
# 1. Scalability: Can handle large documents by processing them in chunks.
# 2. Flexibility: Easy to adjust parameters like chunk size and number of retrieved results.
# 3. Efficiency: Utilizes FAISS for fast similarity search in high-dimensional spaces.
# 4. Integration with Advanced NLP: Uses OpenAI embeddings for state-of-the-art text representation.
#
# ## Conclusion
#
# This simple RAG system provides a solid foundation for building more complex information retrieval and question-answering systems. By encoding document content into a searchable vector store, it enables efficient retrieval of relevant information in response to queries. This approach is particularly useful for applications requiring quick access to specific information within large documents or document collections.

# %% [markdown]
# ### Import libraries and environment variables

# %%


import dotenv


# Reload the variables in your '.env' file (override the existing variables)
dotenv.load_dotenv("../.env", override=True)

# %%
import logging
import os
import sys

from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path since we work with notebooks
from importlib.metadata import version

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger
from loguru import logger as LOGGER
from loguru._defaults import LOGURU_FORMAT
from openai import OpenAI

from evaluation.evalute_rag import *
from helper_functions import *
from logging_utils import get_logger, global_log_config


global_log_config(
    log_level=logging.getLevelName("DEBUG"),
    json=False,
)
LOGGER.disable("ipykernel.")
LOGGER.disable("ipykernel.kernelbase")
LOGGER.disable("openai._base_client")
LOGGER.disable("httpcore._trace")

import warnings


os.environ["USER_AGENT"] =  (
    f"boss-rag-techniques/0.1.0 | Python/" f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)


# from pydantic.errors import PydanticDeprecatedSince20

# Filter and suppress the specific warning
# warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20, message="The `__fields__` attribute is deprecated*")


# %% [markdown]
# ### Read Docs

# %%
path = "../data/Understanding_Climate_Change.pdf"

# %% [markdown]
# ### Encode document

# %%
def encode_pdf(path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, add_start_index: bool = ADD_START_INDEX, llm_embedding_model_name: str = LLM_EMBEDDING_MODEL_NAME):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    LOGGER.debug(f"Length of documents: {len(documents)}")

    # TODO: Check if we should enable add_start_index
    # In this case we'll split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. We use the RecursiveCharacterTextSplitter, which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

    # We set add_start_index=True so that the character index at which each split Document starts within the initial Document is preserved as metadata attribute "start_index".
    # FIXME: See https://python.langchain.com/v0.2/docs/tutorials/rag/


    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=add_start_index
    )
    texts = text_splitter.split_documents(documents)
    LOGGER.debug(f"Num of chunks: {len(texts)}")
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model=llm_embedding_model_name, dimensions=EMBEDDING_MODEL_DIMENSIONS)
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

# %%
chunks_vector_store = encode_pdf(path, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP, add_start_index = ADD_START_INDEX, llm_embedding_model_name = LLM_EMBEDDING_MODEL_NAME)

# %% [markdown]
# ### Create retriever

# %%
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs=DEFAULT_SEARCH_KWARGS)

# %% [markdown]
# ### Test retriever

# %%
test_query = QUESTION_TO_ASK
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)

# %% [markdown]
# ### Evaluate results

# %%
# evaluate_rag(chunks_query_retriever)

# %%
import bs4

# %%
# testing some stuff out
import rich

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


rag_prompt = hub.pull("rlm/rag-prompt")

# rich.inspect(rag_prompt, all=True)

# %% [markdown]
# # create sample rag bot that can be used for evaluation rag testing later on

# %%
### RAG bot
# SOURCE: https://docs.smith.langchain.com/tutorials/Developers/rag

# alias retriever
retriever = chunks_query_retriever

import logging

from typing import Any, List

import openai
import pysnooper
import rich

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.pydantic_v1 import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import ConfigurableField, Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI, OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai.types.chat.chat_completion import ChatCompletion


class RAGResponse(BaseModel):
    question: str
    context: list[Document]
    answer: str

def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

class RagBot:

    def __init__(self, retriever = retriever, model: str = LLM_MODEL_NAME, llm: Any = None):
        self._retriever = retriever
        # Wrapping the client instruments the LLM
        self._client = wrap_openai(openai.Client())
        self._model = model
        if llm is None:
            self._llm = ChatOpenAI(
                name="ChatOpenAI",
                streaming=True,
                model=self._model,
                max_retries=9,
                # max_tokens=900,
                # max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
        else:
            self._llm = llm

    @traceable()
    def execute_chain(self, question: str):
        chain = self.build_chain(question)
        # FIXME: Evaluators will expect "answer" and "context"
        # Evaluators will expect "answer" and "context"
        # EG:
        # return {
        #     "answer": response.choices[0].message.content,
        #     "context": [str(doc) for doc in docs],
        # }
        # return chain.invoke(question)
        return chain.invoke({"question": question})

    @traceable()
    async def aexecute_chain(self, question: str):
        chain = self.build_chain(question)
        return await chain.ainvoke(question)

    @traceable()
    def stream_chain(self, question: str):
        chain = self.build_chain(question)
        chunks = []
        for chunk in chain.stream(question):
            rich.print(chunk.content, flush=True)
            chunks.append(chunk.content)
        return "".join(chunks)

    @traceable()
    async def astream_chain(self, question: str):
        chain = self.build_chain(question)
        chunks = []
        async for chunk in chain.astream(question):
            chunks.append(chunk)
            rich.print(chunk.content, flush=True)
        return "".join(chunks)


    @traceable()
    def retrieve_docs(self, question):
        # bpdb.set_trace()
        docs = self._retriever.invoke(question)
        rich.print(docs)
        rich.inspect(docs, all=True)
        return self._retriever.invoke(question)

    # @pysnooper.snoop()
    @traceable()
    def build_chain(self, question):
        # NOTE: https://python.langchain.com/v0.2/docs/tutorials/rag/
        # NOTE: Look at this for inspiration
        """
        Builds a RAG chain using LangChain's ChatPromptTemplate.from_messages.

        Args:
            question: The user's question.

        Returns:
            A LangChain LLMChain object representing the RAG chain.
        """
        system_template = (
            "You are a helpful AI code assistant with expertise in climate change. "
            "Use the following docs to produce a concise code solution to the user question.\n\n"
            "## Docs\n\n{context}"
        )

        # SystemMessage
        # This represents a message with role "system", which tells the model how to behave. Not every model provider supports this.
        # system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        # human_template = "{question}"
        # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # # chain = LLMChain(llm=self._llm, prompt=chat_prompt)

        # question_answer_chain = create_stuff_documents_chain(self._llm, chat_prompt)
        # # rag_chain = create_retrieval_chain(self._retriever, question_answer_chain)
        # READ THIS: https://python.langchain.com/v0.2/docs/how_to/qa_sources/

        base_human_template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question} \n\nContext: {context} \n\nAnswer:"

        base_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", base_human_template),
        ])

        LOGGER.debug(base_prompt)

        # Version 1
        # rag_chain_from_docs = (
        #     {
        #         "question": RunnablePassthrough(), # input query. RunnablePassthrough() just passes the input through, ie the input query/question.
        #         "context": retriever | format_docs, # context
        #     }
        #     | base_prompt   # format query and context into prompt
        #     | self._llm # generate response
        #     | StrOutputParser()  # coerce to string
        # )


        rag_chain_from_docs = (
            {
                "question": lambda x: x["question"], # input query. RunnablePassthrough() just passes the input through, ie the input query/question.
                "context": lambda x: format_docs(x["context"]), # context
            }
            | base_prompt   # format query and context into prompt
            | self._llm # generate response
            | StrOutputParser()  # coerce to string
        )
        # Pass input query to retriever
        retrieve_docs = (lambda x: x["question"]) | retriever

        # Below, we chain `.assign` calls. This takes a dict and successively
        # adds keys-- "context" and "answer"-- where the value for each key
        # is determined by a Runnable. The Runnable operates on all existing
        # keys in the dict.
        chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
            answer=rag_chain_from_docs
        )

        chain.get_graph().print_ascii()

        return chain

    # @traceable()
    # def get_answer(self, question: str):
    #     docs = self.retrieve_docs(question)
    #     return self.invoke_llm(question, docs)

    @traceable()
    def get_answer(self, question: str):
        return self.execute_chain(question)

"""
# Example response from get_answer()
{
    'question': 'What is the main cause of climate change?',
    'context': [
        Document(
            metadata={'source': '../data/Understanding_Climate_Change.pdf', 'page': 0},
            page_content='driven by human activities, particularly the emission of greenhou se gases.  \nChapter 2: Causes of Climate Change  \nGreenhouse Gases  \nThe primary cause of recent climate change is the increase in greenhouse gases in the \natmosphere.
Greenhouse gases, such as carbon dioxide (CO2), methane (CH4), and nitrous \noxide (N2O), trap heat from the sun, creating a "greenhouse effect." This effect is  essential \nfor life on Earth, as it keeps the planet warm enough to support life. However, human \nactivities
have intensified this natural process, leading to a warmer climate.  \nFossil Fuels  \nBurning fossil fuels for energy releases large amounts of CO2. This includes coal, oil, and \nnatural gas used for electricity, heating, and transportation. The industrial revolution
marked \nthe beginning of a significant increase in fossil fuel consumption, which continues to rise \ntoday.  \nCoal'
        ),
        Document(
            metadata={'source': '../data/Understanding_Climate_Change.pdf', 'page': 0},
            page_content="Most of these climate changes are attributed to very small variations in Earth's orbit that \nchange the amount of solar energy our planet receives. During the Holocene epoch, which \nbegan at the end of the last ice age, human societies f
lourished, but the industrial era has seen \nunprecedented changes.  \nModern Observations  \nModern scientific observations indicate a rapid increase in global temperatures, sea levels, \nand extreme weather events. The Intergovernmental Panel on Climate Change (IPCC)
has \ndocumented these changes extensively. Ice core samples, tree rings, and ocean sediments \nprovide a historical record that scientists use to understand past climate conditions and \npredict future trends. The evidence overwhelmingly shows that recent changes are
primarily \ndriven by human activities, particularly the emission of greenhou se gases.  \nChapter 2: Causes of Climate Change  \nGreenhouse Gases"
        )
    ],
    'answer': 'The main cause of climate change is the increase in greenhouse gases in the atmosphere, primarily driven by human activities. These gases, such as carbon dioxide (CO2), trap heat from the sun, intensifying the natural greenhouse effect. The burning of
fossil fuels for energy significantly contributes to this increase.'
}
"""

rag_bot = RagBot(retriever)

# %%
# bot smoke test

response = rag_bot.execute_chain(test_query)
# rich.print(response["answer"][:150])
rich.print(response)
# bpdb.set_trace()

# rich.inspect(rag_bot._llm, all=True)

# rag_bot._llm.get_input_schema()


# %% [markdown]
# # Let's explore the chain a little bit
#

# %%
bot_chain = rag_bot.build_chain(test_query)
# rich.inspect(bot_chain, all=True)

# %% [markdown]
# # Define a function that will:
#
# 1. Take a dataset example
# 2. Extract the relevant key (e.g., question) from the example
# 3. Pass it to the RAG chain
# 4. Return the relevant output values from the RAG chain

# %%
def predict_rag_answer(example: dict):
    """Use this for answer evaluation"""
    # bpdb.set_trace()
    response = rag_bot.get_answer(example["input_question"])
    # DISABLED: return {"answer": response["answer"]}
    # FIXME: THIS IS WHERE WE ARE GETTING 'TypeError: string indices must be integers'  because response is a string!!!
    return {"answer": response["answer"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    # bpdb.set_trace()
    response = rag_bot.get_answer(example["input_question"])
    return {"answer": response["answer"], "context": response["context"]}

# %% [markdown]
# ## Evaluator[](https://docs.smith.langchain.com/tutorials/Developers/rag#evaluator)
#
# There are at least 4 types of RAG eval that users are typically interested in.
#
# 1. **Response vs reference answer**
#
# - `Goal`: Measure "*how similar/correct is the RAG chain answer, relative to a ground-truth answer*"
# - `Mode`: Uses ground truth (reference) answer supplied through a dataset
# - `Judge`: Use LLM-as-judge to assess answer correctness.
#
# 1. **Response vs input**
#
# - `Goal`: Measure "*how well does the generated response address the initial user input*"
# - `Mode`: Reference-free, because it will compare the answer to the input question
# - `Judge`: Use LLM-as-judge to assess answer relevance, helpfulness, etc.
#
# 1. **Response vs retrieved docs**
#
# - `Goal`: Measure "*to what extent does the generated response agree with the retrieved context*"
# - `Mode`: Reference-free, because it will compare the answer to the retrieved context
# - `Judge`: Use LLM-as-judge to assess faithfulness, hallucinations, etc.
#
# 1. **Retrieved docs vs input**
#
# - `Goal`: Measure "*how good are my retrieved results for this query*"
# - `Mode`: Reference-free, because it will compare the question to the retrieved context
# - `Judge`: Use LLM-as-judge to assess relevance

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### **Response vs reference answer**[](https://docs.smith.langchain.com/tutorials/Developers/rag#response-vs-reference-answer)
#
# Here is an example prompt that we can use:
#
# https://smith.langchain.com/hub/langchain-ai/rag-answer-vs-reference
#
# Here is the a video from our LangSmith evaluation series for reference:
#
# https://youtu.be/lTfhw_9cJqc?feature=shared
#
# Here is our evaluator function:
#
# - `run` is the invocation of `predict_rag_answer`, which has key `answer`
# - `example` is from our eval set, which has keys `question` and `output_answer`
# - We extract these values and pass them into our grader

# %%
import langsmith

from langsmith.evaluation import evaluate


langsmith_client = langsmith.Client()
dataset_name = DATASET_NAME

# %%
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResults, LangChainStringEvaluator, evaluate
from langsmith.run_trees import RunTree
from langsmith.schemas import Example, Run


examples = []
runs = []

# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

def answer_evaluator(run: Run, example: Example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """
    # bpdb.set_trace()
    examples.append(example)
    runs.append(run)

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["input_question"]
    reference = example.outputs["output_answer"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-v-reference", max_retries=MAX_RETRIES)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}

# %% [markdown]
# Now, we kick off evaluation:
#
# - `predict_rag_answer`: Takes an `example` from our eval set, extracts the question, passes to our RAG chain
# - `answer_evaluator`: Passes RAG chain answer, question, and ground truth answer to an evaluator

# %%
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # bpdb.set_trace()
        experiment_results = evaluate(
            predict_rag_answer,
            data=dataset_name,
            evaluators=[answer_evaluator],
            experiment_prefix="rag-answer-v-reference",
            max_concurrency=EVAL_MAX_CONCURRENCY,
            metadata={
                "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
                "langchain_version": version("langchain"),
                "langchain_community_version": version("langchain_community"),
                "langchain_core_version": version("langchain_core"),
                "langchain_openai_version": version("langchain_openai"),
                "langchain_text_splitters_version": version("langchain_text_splitters"),
                "langsmith_version": version("langsmith"),
                "pydantic_version": version("pydantic"),
                "pydantic_settings_version": version("pydantic_settings"),
            },
        )

        experiment_results
except Exception as ex:
    rich.print(f"{ex}")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    rich.print(f"Error Class: {ex.__class__}")
    output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
    rich.print(output)
    rich.print(f"exc_type: {exc_type}")
    rich.print(f"exc_value: {exc_value}")
    traceback.print_tb(exc_traceback)
    bpdb.pm()
# %%
rich.print(runs)

# %% [markdown]
# # Response vs input
# Here is an example prompt that we can use:
#
# https://smith.langchain.com/hub/langchain-ai/rag-answer-helpfulness
#
# The information flow is similar to above, but we simply look at the run answer versus the example question.

# %%
# Grade prompt
grade_prompt_answer_helpfulness = prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def answer_helpfulness_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer helpfulness
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["input_question"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-helpfulness", max_retries=MAX_RETRIES)


    # Structured prompt
    answer_grader = grade_prompt_answer_helpfulness | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_helpfulness_score", "score": score}

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[answer_helpfulness_evaluator],
        experiment_prefix="rag-answer-helpfulness",
        max_concurrency=EVAL_MAX_CONCURRENCY,
        metadata={
            "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
            "langchain_version": version("langchain"),
            "langchain_community_version": version("langchain_community"),
            "langchain_core_version": version("langchain_core"),
            "langchain_openai_version": version("langchain_openai"),
            "langchain_text_splitters_version": version("langchain_text_splitters"),
            "langsmith_version": version("langsmith"),
            "pydantic_version": version("pydantic"),
            "pydantic_settings_version": version("pydantic_settings"),
            "llm_run_config": LLM_RUN_CONFIG,
        },
    )

# %% [markdown]
# ### **Response vs retrieved docs**[](https://docs.smith.langchain.com/tutorials/Developers/rag#response-vs-retrieved-docs)
#
# Here is an example prompt that we can use:
#
# https://smith.langchain.com/hub/langchain-ai/rag-answer-hallucination
#
# Here is the a video from our LangSmith evaluation series for reference:
#
# https://youtu.be/IlNglM9bKLw?feature=shared

# %%
# Prompt
grade_prompt_hallucinations = prompt = hub.pull("langchain-ai/rag-answer-hallucination")

def answer_hallucination_evaluator(run, example) -> dict:
    """
    A simple evaluator for generation hallucination
    """

    # RAG inputs
    input_question = example.inputs["input_question"]
    context = run.outputs["context"]

    # RAG answer
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-hallucination", max_retries=MAX_RETRIES)

    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Get score
    score = answer_grader.invoke({"documents": context,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_hallucination", "score": score}

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    experiment_results = evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[answer_hallucination_evaluator],
        experiment_prefix="rag-answer-hallucination",
        max_concurrency=EVAL_MAX_CONCURRENCY,
        metadata={
            "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
            "langchain_version": version("langchain"),
            "langchain_community_version": version("langchain_community"),
            "langchain_core_version": version("langchain_core"),
            "langchain_openai_version": version("langchain_openai"),
            "langchain_text_splitters_version": version("langchain_text_splitters"),
            "langsmith_version": version("langsmith"),
            "pydantic_version": version("pydantic"),
            "pydantic_settings_version": version("pydantic_settings"),
            "llm_run_config": LLM_RUN_CONFIG,
        },
    )

# %% [markdown]
# ### **Retrieved docs vs input**[](https://docs.smith.langchain.com/tutorials/Developers/rag#retrieved-docs-vs-input)
#
# Here is an example prompt that we can use:
#
# https://smith.langchain.com/hub/langchain-ai/rag-document-relevance
#
# Here is the a video from our LangSmith evaluation series for reference:
#
# https://youtu.be/Fr_7HtHjcf0?feature=shared

# %%
# Grade prompt
grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")

def docs_relevance_evaluator(run, example) -> dict:
    """
    A simple evaluator for document relevance
    """

    # RAG inputs
    input_question = example.inputs["input_question"]
    context = run.outputs["context"]

    # LLM grader
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-document-relevance", max_retries=MAX_RETRIES)

    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Get score
    score = answer_grader.invoke({"question":input_question,
                                  "documents":context})
    score = score["Score"]

    return {"key": "document_relevance", "score": score}

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    experiment_results = evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[docs_relevance_evaluator],
        experiment_prefix="rag-doc-relevance",
        max_concurrency=EVAL_MAX_CONCURRENCY,
        metadata={
            "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
            "langchain_version": version("langchain"),
            "langchain_community_version": version("langchain_community"),
            "langchain_core_version": version("langchain_core"),
            "langchain_openai_version": version("langchain_openai"),
            "langchain_text_splitters_version": version("langchain_text_splitters"),
            "langsmith_version": version("langsmith"),
            "pydantic_version": version("pydantic"),
            "pydantic_settings_version": version("pydantic_settings"),
            "llm_run_config": LLM_RUN_CONFIG,
        },
    )

# %% [markdown]
# ## Evaluating intermediate steps[](https://docs.smith.langchain.com/tutorials/Developers/rag#evaluating-intermediate-steps)
#
# Above, we returned the retrieved documents as part of the final answer.
#
# However, we will show that this is not required.
#
# We can isolate them as intermediate chain steps.
#
# See detail on isolating intermediate chain steps [here](https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_on_intermediate_steps).
#
# Here is the a video from our LangSmith evaluation series for reference:
#
# https://youtu.be/yx3JMAaNggQ?feature=shared

# %%
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run


def document_relevance_grader(root_run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks to see if retrieved documents are relevant to the question
    """
    # bpdb.set_trace()
    # Get specific steps in our RAG pipeline, which are noted with @traceable decorator
    rag_pipeline_run = next(
        run for run in root_run.child_runs if run.name == "get_answer"
    )
    # DISABLED: # retrieve_run = next(
    # DISABLED: #     run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs"
    # DISABLED: # )
    # retrieve_run = next(
    #     run for run in rag_pipeline_run.child_runs if run.name == "Retriever"
    # )
    context = "\n\n".join(doc.page_content for doc in rag_pipeline_run.outputs["context"])
    input_question = example.inputs["input_question"]

    # LLM grader
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-document-relevance", max_retries=MAX_RETRIES)

    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Get score
    score = answer_grader.invoke({"question":input_question,
                                  "documents":context})
    score = score["Score"]

    return {"key": "document_relevance", "score": score}

def answer_hallucination_grader(root_run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks to see the answer is grounded in the documents
    """
    # bpdb.set_trace()

    # RAG input
    rag_pipeline_run = next(
        run for run in root_run.child_runs if run.name == "get_answer"
    )
    # DISABLED: # retrieve_run = next(
    # DISABLED: #     run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs"
    # DISABLED: # )
    # retrieve_run = next(
    #     run for run in rag_pipeline_run.child_runs if run.name == "Retriever"
    # )
    # context = "\n\n".join(doc.page_content for doc in retrieve_run.outputs["output"])
    context = "\n\n".join(doc.page_content for doc in rag_pipeline_run.outputs["context"])

    # RAG output
    prediction = rag_pipeline_run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-hallucination", max_retries=MAX_RETRIES)

    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Get score
    score = answer_grader.invoke({"documents": context,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_hallucination", "score": score}


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[document_relevance_grader, answer_hallucination_grader],
        experiment_prefix="rag-doc-relevance-and-hallucination-grader",
        max_concurrency=EVAL_MAX_CONCURRENCY,
        metadata={
            "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
            "langchain_version": version("langchain"),
            "langchain_community_version": version("langchain_community"),
            "langchain_core_version": version("langchain_core"),
            "langchain_openai_version": version("langchain_openai"),
            "langchain_text_splitters_version": version("langchain_text_splitters"),
            "langsmith_version": version("langsmith"),
            "pydantic_version": version("pydantic"),
            "pydantic_settings_version": version("pydantic_settings"),
            "llm_run_config": LLM_RUN_CONFIG,
        },
    )

# %% [markdown]
# -------------------------

# %% [markdown]
# # Evalute w/ Langsmith
#
# ## Define metrics
#
# After creating our dataset, we can now define some metrics to evaluate our responses on. Since we have an expected answer, we can compare to that as part of our evaluation. However, we do not expect our application to output those exact answers, but rather something that is similar. This makes our evaluation a little trickier.
#
# In addition to evaluating correctness, let's also make sure our answers are short and concise. This will be a little easier - we can define a simple Python function to measure the length of the response.
#
# Let's go ahead and define these two metrics.
#
# For the first, we will use an LLM to judge whether the output is correct (with respect to the expected output). This LLM-as-a-judge is relatively common for cases that are too complex to measure with a simple function. We can define our own prompt and LLM to use for evaluation here:

# %%
# # SOURCE: https://docs.smith.langchain.com/tutorials/Developers/evaluation

# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts.prompt import PromptTemplate
# from langsmith.evaluation import LangChainStringEvaluator

# dataset_name = "Climate Change Q&A"


# # # Storing inputs in a dataset lets us
# # # run chains and LLMs over a shared set of examples.
# # dataset = client.create_dataset(
# #     dataset_name=dataset_name,
# #     description="Questions and answers about climate change.",
# # )
# # for input_prompt, output_answer in example_inputs:
# #     client.create_example(
# #         inputs={"question": input_prompt},
# #         outputs={"answer": output_answer},
# #         metadata={"source": "Various"},
# #         dataset_id=dataset.id,
# #     )

# _PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
# You are grading the following question:
# {query}
# Here is the real answer:
# {answer}
# You are grading the following predicted answer:
# {result}
# Respond with CORRECT or INCORRECT:
# Grade:
# """

# PROMPT = PromptTemplate(
#     input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
# )
# eval_llm = ChatAnthropic(temperature=0.0)

# qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm, "prompt": PROMPT})

# %% [markdown]
# For evaluating the length of the response, this is a lot easier! We can just define a simple function that checks whether the actual output is less than 2x the length of the expected result.

# %%
# from langsmith.schemas import Run, Example

# def evaluate_length(run: Run, example: Example) -> dict:
#     prediction = run.outputs.get("output") or ""
#     required = example.outputs.get("answer") or ""
#     score = int(len(prediction) < 2 * len(required))
#     return {"key":"length", "score": score}

# %% [markdown]
# # Run Evaluations
#
# Great! So now how do we run evaluations? Now that we have a dataset and evaluators, all that we need is our application! We will build a simple application that just has a system message with instructions on how to respond and then passes it to the LLM. We will build this using the OpenAI SDK directly:

# %%
# # my app

# import openai

# openai_client = openai.Client()

# def my_app(question: str):
#     return openai_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Respond to the users question in a short, concise manner (one short sentence)."
#             },
#             {
#                 "role": "user",
#                 "content": question,
#             }
#         ],
#     ).choices[0].message.content

# %% [markdown]
# Before running this through LangSmith evaluations, we need to define a simple wrapper that maps the input keys from our dataset to the function we want to call, and then also maps the output of the function to the output key we expect.
#
#

# %%
# def langsmith_app(inputs):
#     output = my_app(inputs["question"])
#     return {"output": output}

# %% [markdown]
# Great! Now we're ready to run evaluation. Let's do it!

# %%
# from langsmith.evaluation import evaluate

# experiment_results = evaluate(
#     langsmith_app, # Your AI system
#     data=dataset_name, # The data to predict and grade over
#     evaluators=[evaluate_length, qa_evaluator], # The evaluators to score the results
#     experiment_prefix="openai-3.5", # A prefix for your experiment names to easily identify them
# )
