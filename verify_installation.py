"""
Verify Installation Script
Checks if all required packages are installed correctly
"""

import sys
import importlib
from packaging import version


def check_package(package_name, min_version=None):
    """
    Check if a package is installed and optionally verify version
    
    Returns: (installed, current_version, status_message)
    """
    try:
        module = importlib.import_module(package_name)
        
        # Get version
        if hasattr(module, '__version__'):
            current_version = module.__version__
        else:
            current_version = "unknown"
        
        # Check minimum version if specified
        if min_version and current_version != "unknown":
            try:
                if version.parse(current_version) < version.parse(min_version):
                    return True, current_version, f"⚠️  Version {current_version} < {min_version}"
            except:
                pass
        
        return True, current_version, "✓"
    
    except ImportError:
        return False, None, "❌ NOT INSTALLED"


def main():
    """Check all required packages"""
    print("="*70)
    print("VERIFYING INSTALLATION")
    print("="*70)
    
    # Required packages with minimum versions
    packages = [
        ("torch", "2.0.0"),
        ("torchvision", None),
        ("torchaudio", None),
        ("librosa", "0.10.0"),
        ("soundfile", None),
        ("numpy", "1.24.0"),
        ("scipy", None),
        ("pandas", None),
        ("matplotlib", None),
        ("tensorboard", None),
        ("tqdm", None),
        ("yaml", None),  # PyYAML imports as 'yaml'
        ("sklearn", None),  # scikit-learn imports as 'sklearn'
    ]
    
    print("\nChecking packages:\n")
    
    all_installed = True
    results = []
    
    for package_name, min_version in packages:
        installed, current_version, status = check_package(package_name, min_version)
        
        # Format display name
        if package_name == "yaml":
            display_name = "PyYAML"
        elif package_name == "sklearn":
            display_name = "scikit-learn"
        else:
            display_name = package_name
        
        # Print result
        version_str = f"v{current_version}" if current_version else ""
        print(f"{display_name:20s} {version_str:15s} {status}")
        
        if not installed:
            all_installed = False
            results.append((display_name, False))
        else:
            results.append((display_name, True))
    
    print("\n" + "="*70)
    
    # Check CUDA availability
    print("\nCUDA Availability:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA is NOT available (will use CPU)")
    except:
        print("❌ Could not check CUDA availability")
    
    print("\n" + "="*70)
    
    # Summary
    if all_installed:
        print("\n✓ All required packages are installed!")
        print("\nYou can now:")
        print("  1. Prepare your dataset in data/train/ and data/val/")
        print("  2. Train the model: python train.py")
        print("  3. Test detection: python main.py --audio test.wav")
    else:
        print("\n❌ Some packages are missing!")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*70)
    
    return all_installed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
