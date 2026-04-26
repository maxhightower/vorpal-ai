from .base import Document, Retriever, RetrievalResult
from .local_document_retriever import LocalDocumentRetriever
from .web_retriever_stub import WebRetrieverStub

__all__ = [
    "Document",
    "Retriever",
    "RetrievalResult",
    "LocalDocumentRetriever",
    "WebRetrieverStub",
]
