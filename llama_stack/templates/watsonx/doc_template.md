---
orphan: true
---
# watsonx Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations.

{{ providers_table }}

{% if run_config_env_vars  %}

### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in run_config_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}

{% if default_models %}
### Models

The following models are available by default:

{% for model in default_models %}
- `{{ model.model_id }} {{ model.doc_string }}`
{% endfor %}
{% endif %}


### Prerequisite: API Keys

Make sure you have access to a watsonx API Key. You can get one by referring [watsonx.ai](https://www.ibm.com/docs/en/masv-and-l/maximo-manage/continuous-delivery?topic=setup-create-watsonx-api-key).


## Running Llama Stack with watsonx

You can do this via Conda (build code), venv or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=5001
docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-{{ name }} \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env WATSONX_API_KEY=$WATSONX_API_KEY \
  --env WATSONX_PROJECT_ID=$WATSONX_PROJECT_ID \
  --env WATSONX_BASE_URL=$WATSONX_BASE_URL
```

### Via Conda

```bash
llama stack build --template watsonx --image-type conda
llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env WATSONX_API_KEY=$WATSONX_API_KEY \
  --env WATSONX_PROJECT_ID=$WATSONX_PROJECT_ID
```
