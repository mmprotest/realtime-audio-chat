from pydantic import BaseModel
from typing import Optional


class STTSettings(BaseModel):
    model_name: str = "small"
    device: str = "auto"  # "auto" -> "cuda" if available else "cpu"
    compute_type: str = "float16"  # if cpu, code will switch to "int8"
    beam_size: int = 1
    vad_filter: bool = True
    language: Optional[str] = None  # autodetect if None
