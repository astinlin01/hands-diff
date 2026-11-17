import os

# The package is installed in editable mode (pip install -e .), so tests import hands_diff
# directly. pyproject.toml puts tests/ on the path so test files can import FIXTURES_DIR.
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
