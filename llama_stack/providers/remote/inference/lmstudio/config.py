from typing import Any, Dict
from pydantic import BaseModel

DEFAULT_LMSTUDIO_URL = "http://localhost:1234"

class LmstudioImplConfig(BaseModel):
    url: str = DEFAULT_LMSTUDIO_URL

    @classmethod
    def sample_run_config(cls, url: str = DEFAULT_LMSTUDIO_URL, **kwargs) -> Dict[str, Any]:
        return {"url": url}
