#!/usr/bin/env python3
"""
🔍 Venturalitica SDK Samples - Setup Verification
=================================================

This script checks that your environment is properly configured to run
all scenarios. Run this before starting any examples.

Usage: uv run check_setup.py [--verbose]
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple
import argparse

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_python_version() -> bool:
    """Verify Python >= 3.11"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"{GREEN}✓{RESET} Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"{RED}✗{RESET} Python {version.major}.{version.minor} (Need >=3.11)")
        return False


def check_package(package: str, import_name: str = None) -> bool:
    """Check if a Python package is installed"""
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
        print(f"{GREEN}✓{RESET} {package}")
        return True
    except ImportError:
        print(f"{RED}✗{RESET} {package} (Missing)")
        return False


def check_datasets() -> List[Tuple[str, bool]]:
    """Check which datasets are already downloaded"""
    base = Path(__file__).parent / "datasets"
    datasets = {
        "Loan (German Credit)": base / "loan/german_credit.csv",
        "Hiring (Adult Income)": base / "hiring/adult_income.csv",
        "Health (Heart Disease)": base / "health/heart_disease.csv",
        "LLM (CrowS-Pairs)": base / "llm/crows_pairs_cached.csv",
        "Vision (FairFace)": base / "vision/fairface_cache/metadata.csv",
    }
    
    results = []
    for name, path in datasets.items():
        exists = path.exists()
        status = f"{GREEN}✓{RESET}" if exists else f"{YELLOW}○{RESET}"
        print(f"{status} {name}")
        results.append((name, exists))
    
    return results


def check_scenarios() -> bool:
    """Check that all scenario directories exist"""
    base = Path(__file__).parent / "scenarios"
    scenarios = [
        "loan-mlflow-sklearn",
        "hiring-wandb-torch",
        "bias-llm-audit",
    ]
    
    all_exist = True
    for scenario in scenarios:
        path = base / scenario
        if path.exists():
            print(f"{GREEN}✓{RESET} {scenario}")
        else:
            print(f"{RED}✗{RESET} {scenario} (Missing)")
            all_exist = False
    
    return all_exist


def main():
    parser = argparse.ArgumentParser(description="Check Venturalitica SDK Samples setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    args = parser.parse_args()
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}🔍 Venturalitica SDK Samples - Setup Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    # 1. Python Version
    print(f"{BLUE}[1/4] Python Version{RESET}")
    python_ok = check_python_version()
    print()
    
    # 2. Core Dependencies
    print(f"{BLUE}[2/4] Core Dependencies{RESET}")
    packages = [
        ("venturalitica-sdk", "venturalitica"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
    ]
    
    deps_ok = all(check_package(pkg, imp) for pkg, imp in packages)
    print()
    
    # 3. Scenarios
    print(f"{BLUE}[3/4] Scenario Folders{RESET}")
    scenarios_ok = check_scenarios()
    print()
    
    # 4. Datasets
    print(f"{BLUE}[4/4] Downloaded Datasets{RESET}")
    dataset_results = check_datasets()
    missing_count = sum(1 for _, exists in dataset_results if not exists)
    print()
    
    # Summary
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}📊 SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    all_ok = python_ok and deps_ok and scenarios_ok
    
    if all_ok:
        print(f"{GREEN}✅ Environment is ready!{RESET}")
        print(f"{BLUE}💡 Core scenarios available: loan, hiring, llm{RESET}")
        print(f"{BLUE}   Advanced scenarios at: github.com/venturalitica/venturalitica-sdk-samples-extra{RESET}")
    else:
        print(f"{RED}⚠️  Some issues detected{RESET}")
        if not python_ok:
            print(f"   • Upgrade Python to >=3.11")
        if not deps_ok:
            print(f"   • Install dependencies: uv sync")
        if not scenarios_ok:
            print(f"   • Verify repository integrity")
    
    if missing_count > 0:
        print(f"\n{YELLOW}💡 {missing_count} dataset(s) not yet downloaded{RESET}")
        print(f"   Run prepare_data.py in each scenario folder:")
        print(f"   Example: cd scenarios/loan-mlflow-sklearn && uv run prepare_data.py")
    else:
        print(f"\n{GREEN}✅ All datasets downloaded{RESET}")
    
    print(f"\n{BLUE}🚀 Next Steps:{RESET}")
    print(f"   1. Pick a scenario from scenarios/")
    print(f"   2. Download data: uv run prepare_data.py")
    print(f"   3. Run your first audit: uv run 01_governance_audit.py")
    print()
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
