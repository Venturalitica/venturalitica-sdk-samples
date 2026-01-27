"""Pytest configuration"""
import pytest
from pathlib import Path


@pytest.fixture
def sample_data():
    """Load a small sample of data for testing"""
    import pandas as pd
    return pd.read_csv("data/german_credit.csv").head(100)


@pytest.fixture
def policy_path():
    """Get policy path"""
    return Path("policies/loan/risks.oscal.yaml")
