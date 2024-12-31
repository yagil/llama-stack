# Using Llama Stack as Library with Ollama
=
```
LLAMA_STACK_CONFIG=tests/client-sdk/safety/resources/ollama.yaml pytest tests/client-sdk/safety -v
```

# Using Llama Stack as Library with Together
=
```
export TOGETHER_API_KEY={your_api_key}
LLAMA_STACK_CONFIG=tests/client-sdk/safety/resources/together.yaml pytest tests/client-sdk/safety -v
```

# Using local Llama Stack server instance
```
# Export Llama Stack naming vars
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# Export Ollama naming vars
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
export OLLAMA_SAFETY_MODEL="llama-guard3:1b"

# Start Ollama instance
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
ollama run $OLLAMA_SAFETY_MODEL --keepalive 60m

# Start the Llama Stack server
llama stack run ./llama_stack/templates/ollama/run-with-safety.yaml

# Run the tests
LLAMA_STACK_BASE_URL=http://localhost:5000 pytest tests/client-sdk/safety -v
```
