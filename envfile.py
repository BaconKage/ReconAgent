"""Load `.env.local` / `.env` into the environment, once.

Shared by every module that reads a credential - currently `agent/llm.py` for
model keys and `integrations/razorpay.py` for Razorpay keys. It lives at the top
level rather than inside either of them so that neither has to import the other:
a data source should not depend on the reasoning layer, and vice versa.

`core/` does not use this and must not. The matching engine reads no credentials
and reaches no network, which `tests/test_layer_separation.py` enforces.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

#: Checked in order. `.env.local` first so a personal file beats a shared one.
ENV_FILES = (".env.local", ".env")

_loaded = False


def load_env_files(*, force: bool = False) -> None:
    """Read the dotenv files once, without overriding the real environment.

    Three properties worth keeping:

    * **Idempotent.** Repeated calls are free, so any entry point can call it
      defensively without worrying about who called it first.
    * **Never overrides.** A variable already exported in the shell wins over a
      file on disk, which is what makes `RAZORPAY_KEY_ID=... python -m ...` work
      as expected even when `.env.local` says something else.
    * **Skipped under pytest.** The suite must never pick up a developer's real
      key from a file and quietly turn itself into live API spend - or, worse,
      into a live API call against a real account.

    A missing `python-dotenv` is not an error. It is an optional convenience; the
    environment variables themselves are the contract.
    """
    global _loaded
    if _loaded and not force:
        return
    _loaded = True

    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION"):
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for name in ENV_FILES:
        path = REPO_ROOT / name
        if path.exists():
            load_dotenv(path, override=False)
