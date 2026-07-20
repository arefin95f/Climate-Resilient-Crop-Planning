#!/usr/bin/env python
"""Download Bangladesh district boundaries GeoJSON from HDX (optional enhancement)."""
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "raw" / "bangladesh_districts.geojson"

# geoBoundaries ADM2 simplified GeoJSON (public, no auth)
URL = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/BGD/ADM2/geoBoundaries-BGD-ADM2_simplified.geojson"


def main():
    print(f"Downloading Bangladesh district boundaries...")
    try:
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(resp.content)
        print(f"Saved: {OUT}")
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        print("Maps will use district centroids instead.")
        sys.exit(1)


if __name__ == "__main__":
    main()
