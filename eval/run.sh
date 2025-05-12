#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --provider <provider_name> [--api-key <api_key>] [--model <model_name>] [--hf-model <huggingface_model>]"
    echo "Available providers: openai, ollama, llama, together"
    echo "If --api-key is not provided, will look for environment variable"
    echo "Model name is optional, defaults to provider-specific model"
    echo "HuggingFace model is optional, defaults to provider-specific tokenizer"
    exit 1
}

# Initialize variables
PROVIDER=""
API_KEY=""
MODEL=""
BASE_URL=""
HF_MODEL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --hf-model)
            HF_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if provider is specified
if [ -z "$PROVIDER" ]; then
    echo "Error: Provider must be specified"
    usage
fi

# Convert provider to lowercase
PROVIDER=$(echo "$PROVIDER" | tr '[:upper:]' '[:lower:]')

# Validate provider
case "$PROVIDER" in
    "openai"|"ollama"|"llama"|"together")
        ;;
    *)
        echo "Error: Invalid provider '$PROVIDER'"
        echo "Valid providers are: openai, ollama, llama, together"
        exit 1
        ;;
esac

# Set up provider-specific configurations and API key
case "$PROVIDER" in
    "openai")
        if [ -n "$API_KEY" ]; then
            PROVIDER_API_KEY="$API_KEY"
        elif [ -n "$OPENAI_API_KEY" ]; then
            PROVIDER_API_KEY="$OPENAI_API_KEY"
        else
            echo "Error: OPENAI_API_KEY not set and no API key provided"
            exit 1
        fi
        BASE_URL="https://api.openai.com"
        MODEL=${MODEL:-"gpt-3.5-turbo"}
        HF_MODEL=${HF_MODEL:-"gpt2"}  # Default tokenizer for OpenAI models
        ;;
    "together")
        if [ -n "$API_KEY" ]; then
            PROVIDER_API_KEY="$API_KEY"
        elif [ -n "$TOGETHER_API_KEY" ]; then
            PROVIDER_API_KEY="$TOGETHER_API_KEY"
        else
            echo "Error: TOGETHER_API_KEY not set and no API key provided"
            exit 1
        fi
        BASE_URL="https://api.together.xyz"
        MODEL=${MODEL:-"togethercomputer/llama-2-70b-chat"}
        HF_MODEL=${HF_MODEL:-"meta-llama/Llama-2-7b-hf"}  # Default tokenizer for Llama models
        ;;
    "llama")
        if [ -n "$API_KEY" ]; then
            PROVIDER_API_KEY="$API_KEY"
        elif [ -n "$LLAMA_API_KEY" ]; then
            PROVIDER_API_KEY="$LLAMA_API_KEY"
        else
            echo "Error: LLAMA_API_KEY not set and no API key provided"
            exit 1
        fi
        BASE_URL="https://api.llama.com/compat/v1/chat/completions"
        MODEL=${MODEL:-"llama-2-70b-chat"}
        HF_MODEL=${HF_MODEL:-"meta-llama/Llama-2-7b-hf"}  # Default tokenizer for Llama models
        ;;
    "ollama")
        # Ollama doesn't require an API key, but we'll set a dummy one for compatibility
        PROVIDER_API_KEY="ollama"
        BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
        MODEL=${MODEL:-"llama2"}
        HF_MODEL=${HF_MODEL:-"meta-llama/Llama-2-7b-hf"}  # Default tokenizer for Llama models
        # Check if Ollama is running
        if ! curl -s "${BASE_URL%/v1}/api/tags" > /dev/null; then
            echo "Error: Ollama is not running at ${BASE_URL%/v1}"
            exit 1
        fi
        ;;
esac

# Set OPENAI_API_KEY to the provider's API key for compatibility
export OPENAI_API_KEY="$PROVIDER_API_KEY"

# Export other configuration for use in Python scripts
export LLM_PROVIDER="$PROVIDER"
export LLM_BASE_URL="$BASE_URL"
export LLM_MODEL="$MODEL"
export HF_MODEL="$HF_MODEL"

echo "Using provider: $PROVIDER"
echo "Base URL: $BASE_URL"
echo "Model: $MODEL"
echo "HuggingFace Tokenizer: $HF_MODEL"
if [ "$PROVIDER" != "ollama" ]; then
    echo "API key is set for $PROVIDER"
fi

SYSTEM_PROMPT="<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\n"
APPEND_PROMPT="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
# Run lm_eval with provider-specific configuration
# lm_eval --model multi-provider-api \
#     --tasks gsm8k \
#     --limit 1 \
#     --output_path ./results \
#     --system_instruction "$SYSTEM_PROMPT" \
#     --model_args "provider=llama,model=$MODEL,base_url=$BASE_URL,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer_backend=huggingface,tokenizer=$HF_MODEL"


lm_eval --model openai-chat-completions \
    --tasks hendrycks_math \
    --limit 1 \
    --output_path ./results \
    --apply_chat_template \
    --model_args "model=$MODEL,base_url=$BASE_URL,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer_backend=huggingface,tokenizer=$HF_MODEL,top_p=0,top_k=0,temperature=0,max_gen_len=5120,max_length=5120,max_prompt_len=3072,seed=42"