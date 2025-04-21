"""
Check that all required packages are installed correctly
and that they're compatible with the current Python version.
"""

import sys
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import subprocess
import json
import platform
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Warning: This project works best with Python 3.8 or newer")
    else:
        print("Python version is compatible ✓")
    
    return version.major >= 3 and version.minor >= 8

def check_pip_packages(requirements_file='requirements.txt'):
    """Check if required packages are installed with correct versions"""
    requirements = []
    try:
        with open(requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found.")
        return False
    
    print(f"\nChecking {len(requirements)} required packages:")
    
    satisfied = True
    missing = []
    wrong_version = []
    
    for req in requirements:
        try:
            pkg_resources.require(req)
            package_name = req.split('==')[0]
            version = pkg_resources.get_distribution(package_name).version
            print(f"✓ {package_name} {version}")
        except DistributionNotFound:
            package_name = req.split('==')[0]
            print(f"✗ {req} - Package not found")
            missing.append(req)
            satisfied = False
        except VersionConflict as e:
            print(f"✗ {req} - Version conflict: {e}")
            wrong_version.append(req)
            satisfied = False
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTo install missing packages, run:")
        print(f"pip install -r {requirements_file}")
    
    if wrong_version:
        print("\nPackages with version conflicts:")
        for pkg in wrong_version:
            print(f"  - {pkg}")
        print("\nTo fix version conflicts, run:")
        print(f"pip install -r {requirements_file} --force-reinstall")
    
    return satisfied

def check_ipl_file():
    """Check if ipl.csv exists"""
    if os.path.isfile('ipl.csv'):
        print("\n✓ ipl.csv file exists")
        return True
    else:
        print("\n✗ ipl.csv file not found - a sample file will be created when running the app")
        return False

def system_info():
    """Get system information"""
    print("\nSystem Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

def main():
    """Main function"""
    print("IPL Match Winner Prediction - Dependency Check\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check required packages
    packages_ok = check_pip_packages()
    
    # Check data file
    data_ok = check_ipl_file()
    
    # Print system info
    system_info()
    
    # Overall status
    print("\nOverall Status:")
    if python_ok and packages_ok and data_ok:
        print("✓ All checks passed. The system is ready to run the application.")
    else:
        print("⚠ Some checks failed. Please address the issues above.")

if __name__ == "__main__":
    main() 