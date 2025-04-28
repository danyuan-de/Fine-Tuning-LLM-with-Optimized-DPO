# test/conftest.py
import sys
import os

# Add the project root directory to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# To run the tests, use the command:
# pytest -q