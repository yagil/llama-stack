from pydantic import BaseModel


class OpenTelemetryConfig(BaseModel):
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
