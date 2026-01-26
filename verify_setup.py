"""
Setup Verification Script
==========================
Verifies that the project structure and data files are correctly set up.
"""

import sys
from pathlib import Path
from config import *


def check_directory(path, name):
    """Check if a directory exists."""
    if path.exists():
        print(f"[OK] {name}: {path}")
        return True
    else:
        print(f"[MISSING] {name}: {path}")
        return False


def check_file(path, name):
    """Check if a file exists."""
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[OK] {name}: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"[MISSING] {name}: {path}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("MIMIC-IV PROJECT SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check directories
    print(">> Checking project directories...")
    print("-" * 70)
    all_checks_passed &= check_directory(PROJECT_ROOT / 'src', "Source code")
    all_checks_passed &= check_directory(PROJECT_ROOT / 'data', "Data directory")
    all_checks_passed &= check_directory(DATA_PROCESSED, "Processed data")
    all_checks_passed &= check_directory(DATA_INTERIM, "Interim data")
    all_checks_passed &= check_directory(MODELS_DIR, "Models directory")
    all_checks_passed &= check_directory(PROJECT_ROOT / 'notebooks', "Notebooks")
    all_checks_passed &= check_directory(REPORTS_DIR, "Reports")
    print()
    
    # Check raw data
    print(">> Checking raw MIMIC-IV data...")
    print("-" * 70)
    all_checks_passed &= check_directory(MIMIC_IV_ROOT, "MIMIC-IV root")
    all_checks_passed &= check_directory(MIMIC_IV_HOSP, "MIMIC-IV hospital data")
    all_checks_passed &= check_directory(MIMIC_IV_ICU, "MIMIC-IV ICU data")
    print()
    
    # Check critical data files
    print(">> Checking critical data files...")
    print("-" * 70)
    critical_files = [
        (MIMIC_IV_HOSP / 'patients.csv.gz', "Patients data"),
        (MIMIC_IV_HOSP / 'admissions.csv.gz', "Admissions data"),
        (MIMIC_IV_ICU / 'icustays.csv.gz', "ICU stays data"),
    ]
    
    for file_path, description in critical_files:
        all_checks_passed &= check_file(file_path, description)
    print()
    
    # Check Python environment
    print(">> Checking Python environment...")
    print("-" * 70)
    print(f"Python version: {sys.version.split()[0]}")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing_packages.append(package)
            all_checks_passed = False
    
    print()
    
    # Summary
    print("=" * 70)
    if all_checks_passed:
        print("SUCCESS: ALL CHECKS PASSED!")
        print("Your environment is ready. You can proceed with:")
        print("   python src/data/make_cohort.py")
    else:
        print("WARNING: SOME CHECKS FAILED")
        if missing_packages:
            print("\nInstall missing packages with:")
            print("   pip install -r requirements.txt")
        print("\nReview the errors above and fix them before proceeding.")
    print("=" * 70)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
