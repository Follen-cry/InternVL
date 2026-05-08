"""Pytest config: ensure repo + internvlu package are on sys.path."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

INTERNVLU_PKG = os.environ.get(
    "INTERNVLU_PKG_PATH", "/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U"
)
if os.path.isdir(INTERNVLU_PKG) and INTERNVLU_PKG not in sys.path:
    sys.path.insert(0, INTERNVLU_PKG)
