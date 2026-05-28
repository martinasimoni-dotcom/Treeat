import json
import requests
from pathlib import Path

URL = (
    "https://data.wien.gv.at/daten/geo"
    "?service=WFS&request=GetFeature&version=1.1.0"
    "&typeName=ogdwien:BAUMKATOGD"
    "&srsName=EPSG:4326"
    "&outputFormat=json"
    "&CQL_FILTER=BEZIRK='02'"
)

OUTPUT = Path(__file__).parent.parent / "backend" / "data" / "baumkataster_leopoldstadt.geojson"


def main():
    print("Fetching Baumkataster for Leopoldstadt...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()

    data = response.json()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    count = len(data.get("features", []))
    print(f"Downloaded {count} trees -> {OUTPUT}")


if __name__ == "__main__":
    main()
