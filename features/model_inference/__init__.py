"""Model inference adapters."""

from .openrouter_client import OpenRouterAuthError, OpenRouterError, call_openrouter

__all__ = ["OpenRouterAuthError", "OpenRouterError", "call_openrouter"]
