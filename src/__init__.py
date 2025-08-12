# Automatically load environment variables from a .env file if present.
# This makes API keys available when the package is imported (e.g. from scripts).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, ignore; CLI will load .env as fallback.
    pass