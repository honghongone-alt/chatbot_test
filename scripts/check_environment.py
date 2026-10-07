"""Check RAG dependencies without network requests or an API key."""

import sys
import warnings
from importlib.metadata import version
from io import StringIO

warnings.simplefilter("error")

import langchain
import streamlit
from dotenv import dotenv_values, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, got {sys.version}")

    for package in (
        "langchain",
        "langchain-openai",
        "langchain-text-splitters",
        "streamlit",
        "python-dotenv",
    ):
        print(f"{package}=={version(package)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=32, chunk_overlap=8)
    documents = splitter.create_documents(["RAG retrieves documents. " * 8])
    if len(documents) < 2 or any(len(doc.page_content) > 32 for doc in documents):
        raise RuntimeError("Text splitting check failed")
    if dotenv_values(stream=StringIO("ENV_CHECK=ok\n"))["ENV_CHECK"] != "ok":
        raise RuntimeError("dotenv parsing check failed")

    print(f"Python {sys.version.split()[0]}: imports and local checks passed (warnings treated as errors).")


if __name__ == "__main__":
    main()
