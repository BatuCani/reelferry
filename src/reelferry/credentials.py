"""API keys: read from the environment first, then from the system keychain.

Keys are never written to the settings file, the database, or logs.
"""

from __future__ import annotations

import os

from reelferry.config import OPENROUTER_ENV, OPENROUTER_KEY_NAME, SECRET_SERVICE


def _keyring_get(name: str) -> str | None:
    try:
        import keyring

        value = keyring.get_password(SECRET_SERVICE, name)
    except Exception:  # no usable keychain backend (headless Linux, CI)
        return None
    return value if isinstance(value, str) and value else None


def openrouter_key() -> str | None:
    return os.environ.get(OPENROUTER_ENV) or _keyring_get(OPENROUTER_KEY_NAME)


def openrouter_key_source() -> str | None:
    if os.environ.get(OPENROUTER_ENV):
        return "environment"
    if _keyring_get(OPENROUTER_KEY_NAME):
        return "keychain"
    return None


def store_openrouter_key(value: str) -> None:
    import keyring

    keyring.set_password(SECRET_SERVICE, OPENROUTER_KEY_NAME, value.strip())


def delete_openrouter_key() -> bool:
    import keyring
    import keyring.errors

    try:
        keyring.delete_password(SECRET_SERVICE, OPENROUTER_KEY_NAME)
    except keyring.errors.PasswordDeleteError:
        return False
    return True
