"""
llm_service.py - Production-Ready Google Gemini API Service Layer for NexaAI.

Features:
- Official Google GenAI SDK (`google.genai`) integration with fallback to `google.generativeai`.
- Multi-turn conversation memory support with sliding context window.
- Accurate connection status testing (`test_connection`).
- High-precision latency measurement via `time.perf_counter`.
- Comprehensive, secure error handling (Missing Key, Quota 429, Invalid Key, Network Timeout)
  without deceptive fake responses.
"""

import os
import time
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv(Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

# Detect Google Gemini SDK availability
GENAI_SDK_AVAILABLE = False
LEGACY_GENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import google.generativeai as legacy_genai
    LEGACY_GENAI_AVAILABLE = True
except ImportError:
    LEGACY_GENAI_AVAILABLE = False


DEFAULT_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


def _error_category(error: Exception) -> str:
    """Classifies a Gemini SDK error without relying on one message format."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None

    detail = str(getattr(error, "message", "") or error).lower()
    if code == 429 or "resource_exhausted" in detail or "quota" in detail or "rate limit" in detail:
        return "QUOTA_EXCEEDED"
    if code == 401 or "api_key_invalid" in detail or "unauthenticated" in detail or "invalid api key" in detail:
        return "INVALID_KEY"
    if code == 403 or "permission_denied" in detail or "forbidden" in detail or "restricted" in detail:
        return "PERMISSION_DENIED"
    if code == 404 or "not_found" in detail or "model not found" in detail or "unsupported model" in detail:
        return "MODEL_UNAVAILABLE"
    if code == 503 or "high demand" in detail or "temporarily unavailable" in detail:
        return "SERVICE_HIGH_DEMAND"
    if code in {408, 500, 502, 504} or isinstance(error, (TimeoutError, ConnectionError, OSError)) or any(
        term in detail for term in ("timeout", "timed out", "connection", "network", "dns", "wsarecv")
    ):
        return "NETWORK_ERROR"
    return "API_ERROR"


def _friendly_error(category: str, model_name: str = "") -> str:
    messages = {
        "MISSING_KEY": "Add a Gemini API key in the sidebar or GEMINI_API_KEY in .env, then retry.",
        "INVALID_KEY": "Google rejected this API key. Verify your key at https://aistudio.google.com/app/apikey.",
        "PERMISSION_DENIED": "The key lacks permission or has an API restriction. In AI Studio or Google Cloud, allow access to the Gemini API for this key.",
        "MODEL_UNAVAILABLE": f"The model '{model_name}' is not supported for this API key tier. Choose an active model (e.g. gemini-3.8-flash) from the sidebar.",
        "SERVICE_HIGH_DEMAND": f"Google's server for '{model_name}' is experiencing a temporary spike in demand. Please retry in a few moments.",
        "QUOTA_EXCEEDED": "The key reached its rate or usage limit. Check AI Studio usage and quota, then retry later.",
        "NETWORK_ERROR": "Could not reach Google’s Gemini API due to network timeout or connection reset. Check your internet connection or proxy, then retry.",
        "API_ERROR": "Google returned an unexpected API error. Check the terminal details and retry.",
        "SDK_MISSING": "The Gemini SDK is not installed. Run: pip install -r requirements.txt",
    }
    return messages.get(category, messages["API_ERROR"])


def _log_api_error(error: Exception, api_key: str, model_name: str) -> str:
    category = _error_category(error)
    detail = str(getattr(error, "message", "") or error)
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    detail = re.sub(r"(?:AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_-]{16,}|gsk_[0-9A-Za-z]{16,})", "[REDACTED]", detail)
    detail = detail.replace("\n", " ")[:500]
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    logger.error("Gemini request failed: category=%s code=%s model=%s detail=%s", category, code, model_name, detail)
    return category


def list_available_models(api_key: str) -> List[str]:
    """Return text generation models the configured key can currently access."""
    if not api_key or not GENAI_SDK_AVAILABLE:
        return []
    try:
        client = genai.Client(api_key=api_key)
        available = set()
        for model in client.models.list():
            actions = set(getattr(model, "supported_actions", []) or [])
            name = (getattr(model, "name", "") or "").removeprefix("models/")
            if "generateContent" not in actions:
                continue
            if any(term in name for term in ("image", "tts", "live", "audio", "transcribe", "robotics", "computer-use", "clip")):
                continue
            if re.match(r"^gemini-[\w\.-]+", name):
                available.add(name)
        return sorted(available, key=lambda name: (DEFAULT_MODELS.index(name) if name in DEFAULT_MODELS else len(DEFAULT_MODELS), name))
    except Exception as error:
        _log_api_error(error, api_key, "model-list")
        return []


class LLMService:
    """Manages Gemini API authentication, multi-turn conversation generation, and error handling."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODELS[0]):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODELS[0])).strip()
        self._cached_connection_status: Optional[Tuple[bool, str]] = None
        self.connection_message = ""
        self._init_client()

    def _init_client(self):
        """Initializes the official Gemini client instance."""
        self.client = None
        self._cached_connection_status = None

        load_dotenv(Path(__file__).with_name(".env"), override=False)
        effective_key = (self.api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not effective_key or effective_key.startswith("your_"):
            return

        self.api_key = effective_key

        if GENAI_SDK_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as error:
                category = _log_api_error(error, self.api_key, self.model_name)
                self.connection_message = _friendly_error(category, self.model_name)
                self.client = None
        elif LEGACY_GENAI_AVAILABLE:
            try:
                legacy_genai.configure(api_key=self.api_key)
                self.client = legacy_genai
            except Exception as error:
                category = _log_api_error(error, self.api_key, self.model_name)
                self.connection_message = _friendly_error(category, self.model_name)
                self.client = None

    def set_api_key(self, api_key: str):
        """Updates the API key dynamically from UI."""
        new_key = api_key.strip()
        if new_key != self.api_key:
            self.api_key = new_key
            self._init_client()

    def set_model(self, model_name: str):
        """Updates model variant dynamically."""
        model_name = model_name.strip()
        if model_name != self.model_name:
            self.model_name = model_name
            self._cached_connection_status = None

    def has_valid_key(self) -> bool:
        """Checks if a non-empty, genuine API key exists (ignoring placeholders)."""
        load_dotenv(Path(__file__).with_name(".env"), override=False)
        key = (self.api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            return False
        if key.startswith("your_") or key == "your_gemini_api_key_here":
            return False
        return True

    def test_connection(self) -> Tuple[bool, str]:
        """
        Tests the actual connection status of the Gemini API.
        Returns: (is_connected: bool, status_message: str)
        """
        if not self.has_valid_key():
            self.connection_message = _friendly_error("MISSING_KEY")
            return False, "Key Missing"

        if self.client is None:
            self._init_client()

        if not GENAI_SDK_AVAILABLE and not LEGACY_GENAI_AVAILABLE:
            self.connection_message = _friendly_error("SDK_MISSING")
            return False, "SDK Not Installed"

        # Return cached status if verified recently
        if self._cached_connection_status is not None:
            return self._cached_connection_status

        # Perform a minimal verification ping
        try:
            if GENAI_SDK_AVAILABLE and self.client:
                models_to_try = [self.model_name]
                if not self.model_name.startswith("models/"):
                    models_to_try.append(f"models/{self.model_name}")
                else:
                    models_to_try.append(self.model_name.removeprefix("models/"))

                response = None
                last_ping_error = None
                for cand in models_to_try:
                    try:
                        response = self.client.models.generate_content(
                            model=cand,
                            contents="Reply with OK only.",
                            config=types.GenerateContentConfig(
                                max_output_tokens=100,
                                thinking_config=types.ThinkingConfig(thinking_budget=0)
                            )
                        )
                        if getattr(response, "text", None):
                            self.model_name = cand.removeprefix("models/")
                            break
                    except Exception as exc:
                        last_ping_error = exc
                        continue

                if response and getattr(response, "text", None):
                    self._cached_connection_status = (True, "Connected")
                    self.connection_message = "Gemini text generation succeeded."
                    return True, "Connected"
                elif last_ping_error:
                    raise last_ping_error
                else:
                    raise RuntimeError("Gemini returned no text for the connection test")
            elif LEGACY_GENAI_AVAILABLE:
                model = legacy_genai.GenerativeModel(self.model_name)
                response = model.generate_content("ping")
                if not getattr(response, "text", None):
                    raise RuntimeError("Gemini returned no text for the connection test")
                self._cached_connection_status = (True, "Connected")
                self.connection_message = "Gemini text generation succeeded."
                return True, "Connected"
        except Exception as error:
            category = _log_api_error(error, self.api_key, self.model_name)
            labels = {
                "INVALID_KEY": "Invalid API Key",
                "PERMISSION_DENIED": "Permission Denied",
                "MODEL_UNAVAILABLE": "Model Unavailable",
                "SERVICE_HIGH_DEMAND": "High Demand",
                "QUOTA_EXCEEDED": "Quota Exceeded",
                "NETWORK_ERROR": "Network Error",
                "API_ERROR": "API Error",
            }
            label = labels.get(category, "API Error")
            self.connection_message = _friendly_error(category, self.model_name)
            self._cached_connection_status = (False, label)
            return False, label

        self.connection_message = self.connection_message or _friendly_error("API_ERROR")
        return False, "API Error"

    def generate_response(
        self,
        system_instruction: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generates a response from the Google Gemini API with multi-turn conversation context.
        Accurately measures latency and handles all error conditions gracefully without fake outputs.
        """
        start_time = time.perf_counter()

        # 1. Validation: API Key missing
        if not self.has_valid_key():
            latency = round(time.perf_counter() - start_time, 3)
            return {
                "text": (
                    "⚠️ **Google Gemini API Key Required**\n\n"
                    "NexaAI is ready to assist you across Python programming, engineering, "
                    "academics, and general knowledge.\n\n"
                    "To activate live AI responses:\n"
                    "1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).\n"
                    "2. Enter your key in the left sidebar under **⚙️ Gemini API & Settings**, or add "
                    "`GEMINI_API_KEY=your_key` to a `.env` file in the project directory."
                ),
                "latency_sec": latency,
                "model_used": self.model_name,
                "is_error": True,
                "error_type": "MISSING_KEY"
            }

        # 2. Build multi-turn conversation payload (sliding window: last 8 messages)
        history_window = []
        if conversation_history:
            history_window = [
                m for m in conversation_history
                if m.get("role") in ["user", "assistant"] and m.get("content")
            ][-8:]
        # Ensure client is instantiated if key is present
        if self.client is None and self.has_valid_key():
            self._init_client()

        # 3. Call via official google-genai SDK
        if GENAI_SDK_AVAILABLE and hasattr(self.client, "models"):
            try:
                # Prepare contents list
                contents_payload = []
                for turn in history_window:
                    role_id = "user" if turn["role"] == "user" else "model"
                    contents_payload.append(
                        types.Content(
                            role=role_id,
                            parts=[types.Part.from_text(text=turn["content"])]
                        )
                    )

                # Add current user prompt
                contents_payload.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_prompt)]
                    )
                )

                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                models_to_try = [self.model_name]
                if not self.model_name.startswith("models/"):
                    models_to_try.append(f"models/{self.model_name}")
                else:
                    models_to_try.append(self.model_name.removeprefix("models/"))

                response = None
                last_call_error = None

                for candidate_model in models_to_try:
                    for attempt in range(2):
                        try:
                            response = self.client.models.generate_content(
                                model=candidate_model,
                                contents=contents_payload,
                                config=config,
                            )
                            if getattr(response, "text", None):
                                self.model_name = candidate_model.removeprefix("models/")
                                break
                        except Exception as error:
                            last_call_error = error
                            code = getattr(error, "code", None) or getattr(error, "status_code", None)
                            if code == 503 and attempt == 0:
                                time.sleep(1.0)
                                continue
                            break
                    if response and getattr(response, "text", None):
                        break

                if not response or not getattr(response, "text", None):
                    if last_call_error:
                        raise last_call_error
                    raise RuntimeError("Gemini returned no text for the generation request")

                latency = round(time.perf_counter() - start_time, 3)
                output_text = getattr(response, "text", None)

                return {
                    "text": output_text.strip(),
                    "latency_sec": latency,
                    "model_used": self.model_name,
                    "is_error": False,
                    "error_type": None
                }

            except Exception as error:
                latency = round(time.perf_counter() - start_time, 3)
                category = _log_api_error(error, self.api_key, self.model_name)
                return {
                    "text": f"⚠️ **Gemini API: {category.replace('_', ' ').title()}**\n\n{_friendly_error(category, self.model_name)}",
                    "latency_sec": latency,
                    "model_used": self.model_name,
                    "is_error": True,
                    "error_type": category,
                }

        # 4. Fallback to legacy SDK if google-genai is not active
        elif LEGACY_GENAI_AVAILABLE:
            try:
                model = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )

                # Format chat history for legacy SDK
                chat = model.start_chat(history=[
                    {"role": "user" if t["role"] == "user" else "model", "parts": [t["content"]]}
                    for t in history_window
                ])

                response = chat.send_message(
                    user_prompt,
                    generation_config=legacy_genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )

                latency = round(time.perf_counter() - start_time, 3)
                return {
                    "text": response.text.strip(),
                    "latency_sec": latency,
                    "model_used": self.model_name,
                    "is_error": False,
                    "error_type": None
                }
            except Exception as error:
                latency = round(time.perf_counter() - start_time, 3)
                category = _log_api_error(error, self.api_key, self.model_name)
                return {
                    "text": f"⚠️ **Gemini API: {category.replace('_', ' ').title()}**\n\n{_friendly_error(category, self.model_name)}",
                    "latency_sec": latency,
                    "model_used": self.model_name,
                    "is_error": True,
                    "error_type": category,
                }

        # 5. No SDK available
        latency = round(time.perf_counter() - start_time, 3)
        return {
            "text": f"⚠️ **Gemini API: SDK Missing**\n\n{_friendly_error('SDK_MISSING')}",
            "latency_sec": latency,
            "model_used": "N/A",
            "is_error": True,
            "error_type": "NO_SDK"
        }
