"""
test_gemini.py - Independent Command-Line Gemini API Verification Script for NexaAI.

Academic Context: AIDS Engineering 5th-Semester NLP Micro Project.
Usage:
    python test_gemini.py                # Run standard ping test with configured model
    python test_gemini.py --list-models  # Discover all text generation models for this key
    python test_gemini.py --model gemini-3.8-flash  # Test specific model
"""

import os
import sys
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Redaction regex to prevent leaking credentials in terminal outputs
REDACT_REGEX = re.compile(r"(?:AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_-]{16,}|gsk_[0-9A-Za-z]{16,})")


def sanitize_text(text: str, secret: str = "") -> str:
    """Removes sensitive credentials from text output."""
    if not text:
        return ""
    if secret:
        text = text.replace(secret, "[REDACTED_API_KEY]")
    return REDACT_REGEX.sub("[REDACTED_API_KEY]", str(text))


def main():
    parser = argparse.ArgumentParser(description="NexaAI - Independent Gemini API Connection Verifier")
    parser.add_argument("--list-models", action="store_true", help="List all accessible models for this API key")
    parser.add_argument("--model", type=str, default=None, help="Override model to test")
    args = parser.parse_args()

    # 1. Load local environment
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        print(f"❌ Error: .env file not found at {env_path.resolve()}")
        print("👉 Action: Copy .env.example to .env and add your GEMINI_API_KEY.")
        sys.exit(1)

    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    configured_model = args.model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

    # 2. Check Key Presence (NEVER print key)
    if not api_key or api_key == "your_gemini_api_key_here" or api_key.startswith("your_"):
        print("❌ Error: GEMINI_API_KEY is missing or contains placeholder in .env")
        print("👉 Action: Create an API key at https://aistudio.google.com/app/apikey and paste it into .env")
        sys.exit(1)

    masked_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    print("=" * 60)
    print("🤖 NexaAI - Google Gemini API Connection Verifier")
    print("=" * 60)
    print(f"✔️ Found GEMINI_API_KEY: [{masked_preview}] (length: {len(api_key)})")
    print(f"✔️ Target Model: {configured_model}")

    # 3. Import SDK
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        print(f"❌ Error: Google GenAI SDK is not installed in current Python environment.")
        print(f"   ImportError: {exc}")
        print("👉 Action: Run: pip install -r requirements.txt")
        sys.exit(1)

    # 4. Initialize Client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        safe_msg = sanitize_text(str(exc), api_key)
        print(f"❌ Error initializing Gemini Client: {safe_msg}")
        sys.exit(1)

    # 5. Diagnostic: List Models if requested
    if args.list_models:
        print("\n🔍 Fetching models available to your Google account/project...")
        try:
            found_models = []
            for m in client.models.list():
                actions = getattr(m, "supported_actions", []) or []
                if "generateContent" in actions:
                    name = getattr(m, "name", "")
                    disp = getattr(m, "display_name", "") or name
                    found_models.append((name, disp))

            print(f"✔️ Found {len(found_models)} text-generation models:")
            for m_name, m_disp in found_models:
                clean_name = m_name.removeprefix("models/")
                print(f"   • {clean_name:<30} ({m_disp})")
            print("=" * 60)
            print("✅ Model listing completed successfully!")
            sys.exit(0)
        except Exception as exc:
            safe_msg = sanitize_text(str(exc), api_key)
            print(f"❌ Error listing models: {safe_msg}")
            sys.exit(1)

    # 6. Test Model Connection with Real Prompt
    test_prompt = "Say hello in exactly 5 words."
    print(f"\n📡 Sending live test prompt: \"{test_prompt}\" to {configured_model}...", flush=True)

    # Normalize model ID: try target model, and if needed prepend/strip 'models/'
    models_to_try = [configured_model]
    if not configured_model.startswith("models/"):
        models_to_try.append(f"models/{configured_model}")
    else:
        models_to_try.append(configured_model.removeprefix("models/"))

    last_error = None
    response_text = None
    model_succeeded = None

    for candidate in models_to_try:
        try:
            print(f"   Connecting using identifier: {candidate}...", flush=True)
            response = client.models.generate_content(
                model=candidate,
                contents=test_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            response_text = getattr(response, "text", None)
            if response_text:
                model_succeeded = candidate
                break
        except Exception as exc:
            last_error = exc
            safe_err = sanitize_text(str(exc), api_key)
            print(f"   Candidate {candidate} returned error: {safe_err[:120]}...", flush=True)

    if response_text:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS: Gemini API Connection Verified!")
        print("=" * 60)
        print(f"Model: {model_succeeded}")
        print(f"AI Response:\n{response_text.strip()}")
        print("=" * 60)
        print("✅ You are ready to run: python -m streamlit run app.py")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED: Gemini API Generation Request Failed")
        print("=" * 60)
        safe_msg = sanitize_text(str(last_error), api_key)
        code = getattr(last_error, "code", None) or getattr(last_error, "status_code", None)

        print(f"Status Code: {code}")
        print(f"Diagnostic Error: {safe_msg}")

        # Actionable guidance
        err_lower = safe_msg.lower()
        if "404" in err_lower or "not_found" in err_lower:
            print("\n👉 Guidance: Model not found or deprecated for this API key tier.")
            print("   Run: python test_gemini.py --list-models")
            print("   Update GEMINI_MODEL in your .env file with an available model.")
        elif "429" in err_lower or "quota" in err_lower or "resource_exhausted" in err_lower:
            print("\n👉 Guidance: API Quota limit reached.")
            print("   Check your Google AI Studio quota and rate limits at https://aistudio.google.com/")
        elif "401" in err_lower or "403" in err_lower or "api_key" in err_lower:
            print("\n👉 Guidance: Invalid API key or permission denied.")
            print("   Verify your key in Google AI Studio: https://aistudio.google.com/app/apikey")
        elif "503" in err_lower or "high demand" in err_lower:
            print("\n👉 Guidance: Google's model server is currently experiencing a temporary demand spike.")
            print("   Wait a few seconds and retry.")
        elif "timeout" in err_lower or "connection" in err_lower:
            print("\n👉 Guidance: Network connectivity problem reaching Google API.")
            print("   Check your internet connection or proxy settings.")

        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
