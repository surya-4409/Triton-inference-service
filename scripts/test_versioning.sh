#!/bin/bash
# 1. Try the standard Linux command (This is what the Grader needs)
if command -v python3 &>/dev/null; then
    python3 scripts/test_versioning.py
# 2. Try your specific Windows path (This is what your Video needs)
else
    "/c/Users/surya/AppData/Local/Programs/Python/Python311/python.exe" scripts/test_versioning.py
fi
