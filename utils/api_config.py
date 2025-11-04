"""
Utility functions for configuring OpenAI-compatible APIs (OpenAI or OpenRouter)
"""
import os
from typing import Optional


def get_api_config():
    """
    Get API configuration from environment variables.
    Supports both OpenAI and OpenRouter.
    
    Returns:
        dict with api_key, base_url, and provider info
    """
    # Check for OpenRouter first
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "api_key": openrouter_key,
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter"
        }
    
    # Fall back to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "api_key": openai_key,
            "base_url": None,  # Use default OpenAI endpoint
            "provider": "openai"
        }
    
    return None


def get_embedding_model(provider: Optional[str] = None):
    """
    Get the appropriate embedding model name based on provider.
    
    Args:
        provider: 'openrouter' or 'openai', or None to auto-detect
    
    Returns:
        Model name string
    """
    if provider is None:
        config = get_api_config()
        if config:
            provider = config["provider"]
        else:
            provider = "openai"
    
    # For OpenRouter, use OpenAI models with prefix
    if provider == "openrouter":
        return "openai/text-embedding-3-small"
    else:
        return "text-embedding-3-small"


def get_llm_model(provider: Optional[str] = None, model: Optional[str] = None):
    """
    Get the appropriate LLM model name based on provider.
    
    Args:
        provider: 'openrouter' or 'openai', or None to auto-detect
        model: Specific model name (optional)
    
    Returns:
        Model name string
    """
    if provider is None:
        config = get_api_config()
        if config:
            provider = config["provider"]
        else:
            provider = "openai"
    
    # If model is specified, use it (with prefix for OpenRouter)
    if model:
        if provider == "openrouter" and not model.startswith("openai/"):
            return f"openai/{model}"
        return model
    
    # Default models
    if provider == "openrouter":
        return "openai/gpt-4o-mini"
    else:
        return "gpt-4o-mini"

