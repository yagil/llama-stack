import lmstudio as lms
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.apis.inference import Inference

from llama_stack.log import get_logger
logger = get_logger(name=__name__, category="inference")

class LmstudioInferenceAdapter(Inference, ModelsProtocolPrivate):
    def client(self):
        return lms.AsyncClient()
    
    async def initialize(self) -> None:
        pass
