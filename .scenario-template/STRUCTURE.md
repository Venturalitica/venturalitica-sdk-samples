"""
Template directory structure for Venturalitica SDK Samples scenarios.

Use this as a reference when creating new scenarios or refactoring existing ones.
"""

# SCENARIO TEMPLATE STRUCTURE
# 
# scenarios/{domain}-{mlops}-{framework}/
# ├── README.md                    # Template: .scenario-template/README.md
# ├── QUICK_START.md               # 2-min walkthrough
# ├── train.py                     # Main script (simple version)
# ├── train_advanced.py            # Version with Green AI, logging (optional)
# ├── pyproject.toml               # Dependencies (pinned versions)
# ├── conftest.py                  # Pytest configuration (optional)
# │
# ├── tests/                       # MINIMAL test suite (smoke tests only)
# │   ├── __init__.py
# │   ├── conftest.py              # Shared fixtures
# │   ├── test_data_loading.py     # Can we load data?
# │   ├── test_policy_enforcement.py # Does policy work?
# │   └── test_smoke.py            # Run train.py with --smoke-test?
# │
# ├── fixtures/                    # Small test data (for CI/CD)
# │   ├── sample_data.csv          # 100 rows for unit tests
# │   ├── expected_output.json     # Expected policy report structure
# │   └── README.md                # How to use fixtures
# │
# ├── docs/                        # Domain-specific documentation (optional)
# │   ├── domain-primer.md         # Why this domain matters
# │   ├── architecture.md          # How the sample works
# │   └── troubleshooting.md       # Common issues
# │
# ├── results/                     # Generated on first run
# │   ├── compliance_report.json
# │   ├── fairness_metrics.csv
# │   └── policy_audit_log.txt
# │
# └── .gitignore                   # Ignore: __pycache__, *.pyc, results/

# KEY PRINCIPLES:
# 1. All .csv data in ../../datasets/{domain}/ (shared, not duplicated)
# 2. All policies in ../../policies/{domain}/ (shared, not duplicated)
# 3. Minimal test suite: only "can you run?" tests (smoke tests)
# 4. No heavy test infrastructure (pytest-cov, factories, mocks)
# 5. README follows standard template (see .scenario-template/README.md)
# 6. train.py is the reference implementation
# 7. Results/ is .gitignored (generated on each run)

# DEPENDENCIES (pyproject.toml):
# - venturalitica-sdk >= 1.2.0
# - pandas >= 2.0
# - scikit-learn >= 1.0 (or equivalent framework)
# - [MLOps tool]: mlflow, wandb, clearml (optional)
# - codecarbon (optional, for Green AI)
# - pytest >= 7.0 (dev-only)

# FILE EXAMPLES:

# 1. conftest.py (pytest configuration)
# pytest_plugins = ["fixtures"]
# 
# @pytest.fixture
# def sample_data():
#     return pd.read_csv("fixtures/sample_data.csv")

# 2. test_data_loading.py (minimal smoke test)
# def test_dataset_exists():
#     path = Path("../../datasets/{domain}/{dataset}.csv")
#     assert path.exists(), f"Dataset not found: {path}"
# 
# def test_dataset_has_required_columns():
#     df = pd.read_csv("../../datasets/{domain}/{dataset}.csv")
#     required = ['target', 'gender', 'age']
#     for col in required:
#         assert col in df.columns, f"Missing column: {col}"

# 3. test_policy_enforcement.py
# def test_policy_file_exists():
#     policy = Path("../../policies/{domain}/risks.oscal.yaml")
#     assert policy.exists(), f"Policy not found: {policy}"
# 
# def test_policy_enforcement_runs():
#     df = pd.read_csv("fixtures/sample_data.csv")
#     results = vl.enforce(
#         data=df,
#         target='target',
#         gender='gender',
#         policy=['../../policies/{domain}/risks.oscal.yaml']
#     )
#     assert results is not None
#     assert len(results) > 0

# 4. test_smoke.py
# def test_train_script_runs():
#     # Run with --smoke-test flag (uses small data, fast)
#     result = subprocess.run(
#         ["python", "train.py", "--smoke-test"],
#         capture_output=True,
#         timeout=30
#     )
#     assert result.returncode == 0, f"train.py failed: {result.stderr}"
#     assert "PASS" in result.stdout or "FAIL" in result.stdout

# 5. .gitignore
# __pycache__/
# *.pyc
# *.pyo
# .venv/
# results/
# .pytest_cache/
# *.egg-info/
# uv.lock (optional: version-pin dependencies)
