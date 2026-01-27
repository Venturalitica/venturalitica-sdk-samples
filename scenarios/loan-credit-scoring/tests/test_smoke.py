"""Minimal smoke tests for loan scenario"""
import subprocess
from pathlib import Path
import pandas as pd
import venturalitica as vl


def test_dataset_exists():
    """Dataset file must exist"""
    path = Path("../../datasets/loan/german_credit.csv")
    assert path.exists(), f"Dataset not found: {path}"


def test_dataset_has_required_columns():
    """Dataset must have required columns"""
    df = pd.read_csv("../../datasets/loan/german_credit.csv")
    required = ["target", "gender", "age", "duration", "credit_amount"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_policy_file_exists():
    """Policy file must exist"""
    policy = Path("../../policies/loan/risks.oscal.yaml")
    assert policy.exists(), f"Policy not found: {policy}"


def test_train_script_runs():
    """Can we run train.py with --smoke-test?"""
    result = subprocess.run(
        ["python", "train.py", "--smoke-test"],
        capture_output=True,
        timeout=60,
        cwd=Path(__file__).parent
    )
    assert result.returncode == 0, f"train.py failed: {result.stderr.decode()}"


def test_policy_enforcement_runs():
    """Can we enforce policies on sample data?"""
    df = pd.read_csv("../../datasets/loan/german_credit.csv").head(100)
    
    results = vl.enforce(
        data=df,
        target="target",
        gender="gender",
        policy=["../../policies/loan/risks.oscal.yaml"]
    )
    
    assert results is not None
    assert len(results) > 0
    # Each result should have passed/failed status
    for r in results:
        assert hasattr(r, 'passed') or hasattr(r, 'control_id')
