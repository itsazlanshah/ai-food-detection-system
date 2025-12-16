"""
Fix corrupted Python packages
"""

import os
import shutil
import sys

def fix_corrupted_packages():
    """Remove corrupted package files"""
    python_dir = sys.executable.replace("python.exe", "").replace("pythonw.exe", "")
    site_packages = os.path.join(python_dir, "Lib", "site-packages")
    
    print("🔧 Fixing corrupted packages...")
    print(f"Python directory: {python_dir}")
    
    # List files in site-packages
    try:
        files = os.listdir(site_packages)
        corrupted = []
        
        for file in files:
            if file.startswith("~") or file.endswith(".dist-info"):
                # Check if it's corrupted
                corrupted_path = os.path.join(site_packages, file)
                if os.path.exists(corrupted_path):
                    corrupted.append(corrupted_path)
                    print(f"Found: {file}")
        
        if corrupted:
            print(f"\nFound {len(corrupted)} corrupted files.")
            choice = input("Remove them? (Y/n): ").strip().lower()
            
            if choice in ['', 'y', 'yes']:
                for path in corrupted:
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                        print(f"✅ Removed: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"❌ Could not remove {path}: {e}")
                
                print("\n✅ Cleanup complete!")
                print("Restart Command Prompt/PowerShell for changes to take effect.")
            else:
                print("⚠️ Skipped cleanup.")
        else:
            print("✅ No corrupted files found!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_corrupted_packages()
    input("\nPress Enter to exit...")