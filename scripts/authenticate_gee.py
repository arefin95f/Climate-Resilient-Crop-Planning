#!/usr/bin/env python
"""One-time Google Earth Engine authentication helper."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _print_setup_steps(project: str, reason: str) -> None:
    print(f"\nEarth Engine setup incomplete ({reason}).")
    print(f"Project: {project}\n")
    print("Complete ALL steps below with the SAME Google account you use to authenticate:\n")
    print("  1. Register for Earth Engine:")
    print("     https://earthengine.google.com/signup/")
    print(f"     Select or link Cloud project: {project}\n")
    print("  2. Enable the Earth Engine API:")
    print(
        "     https://console.developers.google.com/apis/api/"
        f"earthengine.googleapis.com/overview?project={project}\n"
    )
    print("  3. Grant yourself IAM roles on the project:")
    print(f"     https://console.cloud.google.com/iam-admin/iam?project={project}")
    print("     Add your Google email with these roles:")
    print("       - Service Usage Consumer")
    print("       - Earth Engine Resource Admin (or Editor)\n")
    print("  4. Wait 2–5 minutes, then run:")
    print("     python scripts/authenticate_gee.py")


def main() -> int:
    try:
        import ee
    except ImportError:
        print("Install first: pip install earthengine-api")
        return 1

    from src.config import GEE_PROJECT

    print("Opening browser for Earth Engine authentication...")
    ee.Authenticate()

    print(f"Verifying access with project: {GEE_PROJECT}")
    try:
        ee.Initialize(project=GEE_PROJECT)
    except ee.ee_exception.EEException as e:
        msg = str(e)
        if "SERVICE_DISABLED" in msg or "has not been used in project" in msg:
            _print_setup_steps(GEE_PROJECT, "API not enabled")
            return 1
        if "USER_PROJECT_DENIED" in msg or "serviceusage.serviceUsageConsumer" in msg:
            _print_setup_steps(GEE_PROJECT, "missing IAM permission")
            return 1
        if "Earth Engine" in msg or "permission" in msg.lower():
            _print_setup_steps(GEE_PROJECT, "access denied")
            return 1
        raise

    print("Earth Engine is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
