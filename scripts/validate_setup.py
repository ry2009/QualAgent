#!/usr/bin/env python3
"""
QualGent Setup Validation Script
Validates that all components are properly installed and configured
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import json

class SetupValidator:
    """Validates QualGent installation and configuration"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_all(self):
        """Run all validation checks"""
        print("QualGent Setup Validation")
        print("=" * 40)
        
        self.check_python_version()
        self.check_dependencies()
        self.check_api_keys()
        self.check_android_setup()
        self.check_directory_structure()
        self.check_configuration()
        
        self.print_results()
        return len(self.errors) == 0
    
    def check_python_version(self):
        """Check Python version"""
        print("\nChecking Python version...")
        
        version = sys.version_info
        if version >= (3, 11):
            print(f"  [PASS] Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.errors.append(f"Python 3.11+ required, found {version.major}.{version.minor}")
            print(f"  [FAIL] Python {version.major}.{version.minor} (need 3.11+)")
    
    def check_dependencies(self):
        """Check required Python packages"""
        print("\nChecking dependencies...")
        
        required_packages = [
            ("openai", "OpenAI API client"),
            ("anthropic", "Anthropic API client"),
            ("google.generativeai", "Google AI client"),
            ("pydantic", "Data validation"),
            ("aiohttp", "Async HTTP client"),
            ("PIL", "Image processing"),
            ("numpy", "Numerical computing"),
            ("tenacity", "Retry logic")
        ]
        
        optional_packages = [
            ("android_env", "AndroidWorld integration"),
            ("gui_agents", "Agent-S integration")
        ]
        
        # Check required packages
        for package, description in required_packages:
            try:
                importlib.import_module(package)
                print(f"  [PASS] {package} ({description})")
            except ImportError:
                self.errors.append(f"Missing required package: {package}")
                print(f"  [FAIL] {package} - MISSING")
        
        # Check optional packages
        for package, description in optional_packages:
            try:
                importlib.import_module(package)
                print(f"  [PASS] {package} ({description})")
            except ImportError:
                self.warnings.append(f"Optional package not found: {package}")
                print(f"  [WARN] {package} - OPTIONAL")
    
    def check_api_keys(self):
        """Check API key configuration"""
        print("\nChecking API keys...")
        
        api_keys = [
            ("OPENAI_API_KEY", "OpenAI API", False),
            ("ANTHROPIC_API_KEY", "Anthropic API", False),
            ("GCP_API_KEY", "Google Cloud API", True)
        ]
        
        for env_var, name, required in api_keys:
            value = os.getenv(env_var)
            if value:
                print(f"  [PASS] {name} configured")
            elif required:
                self.errors.append(f"Missing required API key: {env_var}")
                print(f"  [FAIL] {name} - REQUIRED")
            else:
                self.warnings.append(f"Optional API key not set: {env_var}")
                print(f"  [WARN] {name} - OPTIONAL")
    
    def check_android_setup(self):
        """Check Android development environment"""
        print("\nChecking Android setup...")
        
        # Check ADB
        try:
            result = subprocess.run(["adb", "version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("  [PASS] ADB available")
            else:
                self.warnings.append("ADB not working properly")
                print("  [WARN] ADB issues detected")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("ADB not found in PATH")
            print("  [WARN] ADB not found")
        
        # Check emulator
        emulator_paths = [
            "~/Library/Android/sdk/emulator/emulator",
            "~/Android/Sdk/emulator/emulator",
            "/opt/android-sdk/emulator/emulator"
        ]
        
        emulator_found = False
        for path in emulator_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                print(f"  [PASS] Android Emulator found at {path}")
                emulator_found = True
                break
        
        if not emulator_found:
            self.warnings.append("Android Emulator not found")
            print("  [WARN] Android Emulator not found")
        
        # Check for AVD
        try:
            result = subprocess.run(["emulator", "-list-avds"], 
                                  capture_output=True, text=True, timeout=10)
            if "AndroidWorldAvd" in result.stdout:
                print("  [PASS] AndroidWorldAvd found")
            else:
                avds = result.stdout.strip().split('\n') if result.stdout.strip() else []
                if avds:
                    print(f"  [WARN] AndroidWorldAvd not found, available AVDs: {', '.join(avds)}")
                    self.warnings.append("AndroidWorldAvd not configured")
                else:
                    print("  [WARN] No AVDs configured")
                    self.warnings.append("No Android AVDs found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("Could not check AVD configuration")
            print("  [WARN] Could not check AVDs")
    
    def check_directory_structure(self):
        """Check project directory structure"""
        print("\nChecking directory structure...")
        
        required_dirs = [
            "src",
            "src/agents",
            "src/core",
            "src/models",
            "config",
            "tasks"
        ]
        
        optional_dirs = [
            "logs",
            "reports",
            "tests"
        ]
        
        # Check required directories
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print(f"  [PASS] {dir_path}/")
            else:
                self.errors.append(f"Missing directory: {dir_path}")
                print(f"  [FAIL] {dir_path}/ - MISSING")
        
        # Check optional directories
        for dir_path in optional_dirs:
            if Path(dir_path).exists():
                print(f"  [PASS] {dir_path}/")
            else:
                print(f"  [INFO] {dir_path}/ - will be created")
    
    def check_configuration(self):
        """Check configuration files"""
        print("\nChecking configuration...")
        
        config_files = [
            ("config/default_config.json", True),
            ("config/gemini_config.json", True),
            ("tasks/wifi_settings_test.json", True),
            ("tasks/email_search_test.json", False)
        ]
        
        for config_file, required in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        json.load(f)
                    print(f"  [PASS] {config_file}")
                except json.JSONDecodeError:
                    self.errors.append(f"Invalid JSON in {config_file}")
                    print(f"  [FAIL] {config_file} - INVALID JSON")
            elif required:
                self.errors.append(f"Missing configuration file: {config_file}")
                print(f"  [FAIL] {config_file} - MISSING")
            else:
                print(f"  [WARN] {config_file} - OPTIONAL")
    
    def print_results(self):
        """Print validation results"""
        print("\n" + "=" * 40)
        print("Validation Results")
        print("=" * 40)
        
        if not self.errors and not self.warnings:
            print("All checks passed! QualGent is ready to use.")
            print("\nNext steps:")
            print("  1. Start Android emulator:")
            print("     emulator -avd AndroidWorldAvd -grpc 8554")
            print("  2. Run demo:")
            print("     python run_demo.py")
            print("  3. Run your first test:")
            print("     python main.py --config config/gemini_config.json --task tasks/wifi_settings_test.json")
            
        else:
            if self.errors:
                print(f"[FAIL] {len(self.errors)} Error(s) found:")
                for error in self.errors:
                    print(f"  • {error}")
                print("\nFix these errors before using QualGent.")
            
            if self.warnings:
                print(f"\n[WARN] {len(self.warnings)} Warning(s):")
                for warning in self.warnings:
                    print(f"  • {warning}")
                print("\nThese warnings may affect some functionality.")
            
            print(f"\nSee README.md for setup instructions.")
    
    def generate_setup_script(self):
        """Generate a setup script based on findings"""
        script_lines = [
            "#!/bin/bash",
            "# QualGent Setup Script",
            "# Generated by validation script",
            "",
            "echo 'Setting up QualGent...'",
            ""
        ]
        
        # Add missing package installations
        missing_packages = [error.split(": ")[1] for error in self.errors if "Missing required package" in error]
        if missing_packages:
            script_lines.extend([
                "echo 'Installing missing packages...'",
                f"pip install {' '.join(missing_packages)}",
                ""
            ])
        
        # Add directory creation
        missing_dirs = [error.split(": ")[1] for error in self.errors if "Missing directory" in error]
        if missing_dirs:
            script_lines.extend([
                "echo 'Creating directories...'",
                *[f"mkdir -p {dir_path}" for dir_path in missing_dirs],
                ""
            ])
        
        script_lines.extend([
            "echo 'Setup complete!'",
            "echo 'Run python scripts/validate_setup.py to verify.'"
        ])
        
        with open("setup_fix.sh", "w") as f:
            f.write("\n".join(script_lines))
        
        print(f"\nGenerated setup_fix.sh script")

def main():
    """Main entry point"""
    validator = SetupValidator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        print("Running validation and generating fix script...")
        success = validator.validate_all()
        validator.generate_setup_script()
        return 0 if success else 1
    else:
        success = validator.validate_all()
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 