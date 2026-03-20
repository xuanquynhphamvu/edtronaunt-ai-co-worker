#!/usr/bin/env bash

# Ensure you run this from the project root
echo "Pulling qwen2.5:32b in Ollama..."
ollama pull qwen2.5:32b
echo "Model pulled successfully! You can verify with 'ollama list'."
