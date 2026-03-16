"""
Pytest configuration for Picard solver tests

Sets up test environment and common fixtures
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set random seed for reproducibility
np.random.seed(42)

# Test markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (>1 sec)"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "validation: marks validation tests (vs FreeGS)"
    )
