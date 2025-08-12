from typing import Any, Dict, List, Optional, TypedDict, Union

from vllm import SamplingParams


class MessageContent(TypedDict):
    """Type definition for message content in LLM requests."""

    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(TypedDict):
    """Type definition for a message in a LLM inference request."""

    role: str
    content: Union[str, List[MessageContent]]


class InferenceRequest(TypedDict, total=False):
    """Type definition for LLM inference request."""

    model: str
    messages: List[Message]
    temperature: float
    top_p: float
    max_completion_tokens: int
    seed: int
    response_format: Optional[Dict[str, Any]]


class VLLMInferenceRequest(TypedDict):
    """Type definition for VLLM inference request format."""

    messages: List[List[Message]]
    sampling_params: Union[SamplingParams, List[SamplingParams]]


class TextArtifact(TypedDict):
    content: str
    notes: Optional[str] = None


class ImageArtifact(TypedDict, total=False):
    description: str
    caption: str
    image_type: str
    position_top: Optional[str] = None
    position_left: Optional[str] = None


class TableArtifact(TypedDict, total=False):
    table_contents: dict
    table_info: str


class ArtifactCollection(TypedDict, total=False):
    text: TextArtifact
    images: List[ImageArtifact]
    tables: List[TableArtifact]


class ExtractedPage(TypedDict):
    doc_path: str
    image_path: str
    page_num: int
    artifacts: ArtifactCollection
