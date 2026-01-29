#!/usr/bin/env python3
"""
Dataset Setup Helper for A/B Testing Repository

This script helps you download and verify the real-world datasets used
in the A/B testing pipelines.

Usage:
    python setup_datasets.py               # Check all datasets
    python setup_datasets.py --download    # Download missing datasets
    python setup_datasets.py --verify      # Verify existing datasets
"""

import argparse
import os
from pathlib import Path
import subprocess
import shutil


# ==============================================================================
# AUTO-LOAD ENVIRONMENT VARIABLES FROM .env FILE
# ==============================================================================
# This allows users to store their KAGGLE_API_TOKEN in a .env file at the
# project root instead of setting it manually in their shell.
# ==============================================================================

def load_env_file():
    """
    Load environment variables from .env file in project root.

    This is a lightweight implementation that doesn't require python-dotenv.
    If .env exists, it reads key=value pairs and sets them in os.environ.

    Returns:
        bool: True if .env was loaded, False otherwise
    """
    env_file = Path(__file__).parent / ".env"

    if not env_file.exists():
        # No .env file - this is fine, user may set env vars manually
        return False

    print(f"🔍 Found .env file at: {env_file}")
    print("   Loading environment variables...")

    try:
        loaded_vars = []
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Strip whitespace
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse key=value
                if '=' not in line:
                    print(f"   ⚠️  Line {line_num}: Skipping invalid line (no '='): {line}")
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present (handles both "value" and 'value')
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Set in environment
                os.environ[key] = value
                loaded_vars.append(key)

                # Show what was loaded (mask sensitive values)
                if 'TOKEN' in key or 'KEY' in key or 'SECRET' in key:
                    display_value = value[:20] + "..." if len(value) > 20 else value
                    print(f"   ✅ Loaded {key} = {display_value}")
                else:
                    print(f"   ✅ Loaded {key} = {value}")

        if loaded_vars:
            print(f"   ✅ Successfully loaded {len(loaded_vars)} environment variable(s)")
            return True
        else:
            print(f"   ⚠️  .env file exists but contains no valid variables")
            return False

    except Exception as e:
        print(f"   ❌ Error reading .env file: {e}")
        print(f"   Continuing without .env (you can set environment variables manually)")
        return False

# Load .env file if it exists (before any other code runs)
load_env_file()
print()  # Blank line for readability


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_status(dataset, status, message=""):
    """Print dataset status with emoji indicators."""
    icons = {
        "found": "✅",
        "missing": "❌",
        "downloading": "⬇️",
        "error": "⚠️",
        "verified": "✓",
    }
    icon = icons.get(status, "")
    print(f"{icon} {dataset}: {status.upper()}", end="")
    if message:
        print(f" - {message}")
    else:
        print()


class DatasetManager:
    """Manages dataset downloads and verification."""

    DATASETS = {
        "criteo": {
            "name": "Criteo Uplift",
            "source": "scikit-uplift (auto-download)",
            "path": None,  # Auto-managed by scikit-uplift
            "size": "~850MB (full), ~8.5MB (1% sample)",
            "requires_download": False,
            "verify_import": "from sklift.datasets import fetch_criteo",
        },
        "marketing": {
            "name": "Marketing A/B Testing",
            "source": "Kaggle: faviovaz/marketing-ab-testing",
            "path": "data/raw/marketing_ab/marketing_AB.csv",
            "kaggle_dataset": "faviovaz/marketing-ab-testing",
            "expected_file": "marketing_AB.csv",
            "size": "~19MB",
            "requires_download": True,
        },
        "cookie_cats": {
            "name": "Cookie Cats Mobile Game",
            "source": "Kaggle: mursideyarkin/mobile-games-ab-testing-cookie-cats",
            "path": "data/raw/cookie_cats/cookie_cats.csv",
            "kaggle_dataset": "mursideyarkin/mobile-games-ab-testing-cookie-cats",
            "expected_file": "cookie_cats.csv",
            "size": "~3MB",
            "requires_download": True,
        },
    }

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.project_root = Path(__file__).parent

    def check_all_datasets(self):
        """Check status of all datasets."""
        print_header("📊 DATASET STATUS CHECK")

        status = {}
        for dataset_id, info in self.DATASETS.items():
            status[dataset_id] = self.check_dataset(dataset_id, info)

        return status

    def check_dataset(self, dataset_id, info):
        """Check if a single dataset is available."""
        print(f"\n🔍 Checking: {info['name']}")
        print(f"   Source: {info['source']}")
        print(f"   Size: {info['size']}")

        if not info['requires_download']:
            # Criteo: Check if scikit-uplift is installed
            try:
                exec(info['verify_import'])
                print_status(dataset_id, "found", "scikit-uplift installed")
                return True
            except ImportError:
                print_status(dataset_id, "missing", "Install: pip install scikit-uplift")
                return False
        else:
            # Kaggle datasets: Check if file exists
            file_path = self.project_root / info['path']
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print_status(dataset_id, "found", f"{size_mb:.1f}MB")
                return True
            else:
                print_status(dataset_id, "missing", f"Expected at: {info['path']}")
                return False

    def check_kaggle_cli(self):
        """Check if Kaggle CLI is installed and configured."""
        print_header("🔑 KAGGLE API CHECK")

        # CRITICAL: Check credentials BEFORE importing kaggle
        # The kaggle library import modifies os.environ if it doesn't find
        # credentials in its expected format (KAGGLE_USERNAME + KAGGLE_KEY)

        # Check for API credentials (three methods)
        # Method 1: KAGGLE_API_TOKEN (newest - for CLI only)
        kaggle_token = os.environ.get('KAGGLE_API_TOKEN')
        if kaggle_token:
            print("✅ API token found in environment: KAGGLE_API_TOKEN")
            print(f"   Token: {kaggle_token[:20]}...")
            print("   Note: Token-based auth works with CLI but not Python kaggle library")

            # Verify kaggle CLI is available (don't import the library!)
            try:
                result = subprocess.run(
                    ["kaggle", "--version"],
                    capture_output=True,
                    text=True,
                    env=os.environ.copy()
                )
                if result.returncode == 0:
                    print(f"✅ Kaggle CLI installed: {result.stdout.strip()}")
                    return True
                else:
                    print(f"❌ Kaggle CLI error: {result.stderr}")
                    return False
            except FileNotFoundError:
                print("❌ Kaggle CLI not found in PATH")
                print("   Install with: pip install kaggle")
                return False

        # Method 2: Traditional username + key (for Python library)
        kaggle_username = os.environ.get('KAGGLE_USERNAME')
        kaggle_key = os.environ.get('KAGGLE_KEY')
        if kaggle_username and kaggle_key:
            print("✅ API credentials found: KAGGLE_USERNAME + KAGGLE_KEY")
            return True

        # Method 3: kaggle.json file (traditional method)
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            print(f"✅ API credentials found: {kaggle_json}")
            return True

        # No credentials found
        print("❌ API credentials not found")
        print("\nSetup options:")
        print("\n📍 Option 1: API Token (Recommended)")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Copy the token (starts with KGAT_...)")
        print("4. Set environment variable:")
        print("   Windows CMD: set KAGGLE_API_TOKEN=<your-token>")
        print("   Windows PowerShell: $env:KAGGLE_API_TOKEN=\"<your-token>\"")
        print("   Linux/Mac: export KAGGLE_API_TOKEN=<your-token>")
        print("\n📍 Option 2: kaggle.json File (Traditional)")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Save kaggle.json to:")
        print(f"   {Path.home() / '.kaggle' / 'kaggle.json'}")
        return False

    def download_dataset(self, dataset_id):
        """Download a single dataset from Kaggle."""
        info = self.DATASETS[dataset_id]

        if not info['requires_download']:
            print(f"ℹ️  {info['name']}: Auto-downloads when used (no manual download needed)")
            return True

        print(f"\n⬇️  Downloading: {info['name']}")
        print(f"   Source: {info['kaggle_dataset']}")

        # Create target directory
        target_dir = self.project_root / Path(info['path']).parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download using Kaggle CLI
        try:
            # Download to temp directory
            temp_dir = target_dir / "temp_download"
            temp_dir.mkdir(exist_ok=True)

            cmd = [
                "kaggle", "datasets", "download",
                "-d", info['kaggle_dataset'],
                "-p", str(temp_dir),
                "--unzip"
            ]

            print(f"   Running: {' '.join(cmd)}")

            # DEBUG: Show environment being passed to subprocess
            print(f"   🔍 DEBUG: KAGGLE_API_TOKEN in current env: {('KAGGLE_API_TOKEN' in os.environ)}")

            # Explicitly pass current environment to subprocess
            # This ensures KAGGLE_API_TOKEN is available to kaggle CLI
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())

            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr}")
                return False

            # Find the downloaded CSV file
            downloaded_files = list(temp_dir.glob("*.csv"))
            if not downloaded_files:
                print(f"❌ No CSV files found in download")
                return False

            # Move to expected location
            source_file = downloaded_files[0]
            target_file = self.project_root / info['path']
            shutil.move(str(source_file), str(target_file))

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            size_mb = target_file.stat().st_size / (1024 * 1024)
            print(f"✅ Downloaded successfully: {size_mb:.1f}MB")
            print(f"   Location: {info['path']}")
            return True

        except FileNotFoundError:
            print("❌ Kaggle CLI not found. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return False

    def download_all_missing(self):
        """Download all missing datasets."""
        print_header("⬇️  DOWNLOADING MISSING DATASETS")

        # Check Kaggle CLI first
        if not self.check_kaggle_cli():
            print("\n⚠️  Cannot download datasets without Kaggle API setup")
            print("Please follow the instructions above to configure Kaggle API")
            return False

        # Check what's missing
        status = self.check_all_datasets()
        missing = [dataset_id for dataset_id, is_found in status.items() if not is_found]

        if not missing:
            print("\n✅ All datasets are already available!")
            return True

        print(f"\n📥 Datasets to download: {', '.join(missing)}")

        # Download each missing dataset
        success = True
        for dataset_id in missing:
            if not self.download_dataset(dataset_id):
                success = False

        return success

    def verify_datasets(self):
        """Verify all datasets are correctly formatted."""
        print_header("✓ VERIFYING DATASETS")

        try:
            from ab_testing.data import loaders
        except ImportError:
            print("❌ Cannot import ab_testing package")
            return False

        all_valid = True

        # Test Criteo (small sample)
        try:
            print("\n🔍 Verifying: Criteo Uplift")
            df = loaders.load_criteo_uplift(sample_frac=0.001)
            print(f"   ✓ Loaded {len(df):,} rows")
            print(f"   ✓ Columns: {df.columns.tolist()}")
            assert 'treatment' in df.columns
            assert 'visit' in df.columns
            assert 'conversion' in df.columns
            print("   ✅ Criteo dataset valid")
        except Exception as e:
            print(f"   ❌ Criteo verification failed: {e}")
            all_valid = False

        # Test Marketing (if available)
        try:
            print("\n🔍 Verifying: Marketing A/B")
            df = loaders.load_marketing_ab(sample_frac=0.01)
            print(f"   ✓ Loaded {len(df):,} rows")
            print(f"   ✓ Columns: {df.columns.tolist()}")
            assert 'test_group' in df.columns
            assert 'converted' in df.columns
            print("   ✅ Marketing dataset valid")
        except FileNotFoundError:
            print("   ⚠️  Not downloaded yet (run with --download)")
        except Exception as e:
            print(f"   ❌ Marketing verification failed: {e}")
            all_valid = False

        # Test Cookie Cats (if available)
        try:
            print("\n🔍 Verifying: Cookie Cats")
            df = loaders.load_cookie_cats(sample_frac=0.1)
            print(f"   ✓ Loaded {len(df):,} rows")
            print(f"   ✓ Columns: {df.columns.tolist()}")
            assert 'version' in df.columns
            assert 'retention_1' in df.columns
            print("   ✅ Cookie Cats dataset valid")
        except FileNotFoundError:
            print("   ⚠️  Not downloaded yet (run with --download)")
        except Exception as e:
            print(f"   ❌ Cookie Cats verification failed: {e}")
            all_valid = False

        return all_valid

    def print_manual_instructions(self):
        """Print manual download instructions for users without Kaggle API."""
        print_header("📖 MANUAL DOWNLOAD INSTRUCTIONS")

        print("\nIf you prefer to download datasets manually:")

        for dataset_id, info in self.DATASETS.items():
            if not info['requires_download']:
                print(f"\n{info['name']}:")
                print(f"  Auto-downloads when used (via scikit-uplift)")
                print(f"  Install: pip install scikit-uplift")
            else:
                print(f"\n{info['name']}:")
                print(f"  1. Visit: https://www.kaggle.com/datasets/{info['kaggle_dataset']}")
                print(f"  2. Click 'Download' button (requires Kaggle account)")
                print(f"  3. Extract ZIP file")
                print(f"  4. Place {info['expected_file']} in: {info['path']}")
                print(f"  Expected size: {info['size']}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Setup Helper for A/B Testing Repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing datasets automatically"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing datasets are valid"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Show manual download instructions"
    )

    args = parser.parse_args()

    manager = DatasetManager()

    # Default: Just check status
    if not (args.download or args.verify or args.manual):
        manager.check_all_datasets()
        print("\n" + "=" * 80)
        print("💡 Next steps:")
        print("   python setup_datasets.py --download    # Download missing datasets")
        print("   python setup_datasets.py --verify      # Verify datasets are valid")
        print("   python setup_datasets.py --manual      # Show manual instructions")
        return

    # Download missing datasets
    if args.download:
        success = manager.download_all_missing()
        if success:
            print("\n✅ All datasets downloaded successfully!")
            print("Run pipelines with: uv run python -m ab_testing.pipelines.marketing_pipeline")
        else:
            print("\n⚠️  Some datasets could not be downloaded")
            print("See instructions above or run: python setup_datasets.py --manual")

    # Verify datasets
    if args.verify:
        success = manager.verify_datasets()
        if success:
            print("\n✅ All datasets verified successfully!")
        else:
            print("\n⚠️  Some datasets failed verification")

    # Show manual instructions
    if args.manual:
        manager.print_manual_instructions()


if __name__ == "__main__":
    main()
