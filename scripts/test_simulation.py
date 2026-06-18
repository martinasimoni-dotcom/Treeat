import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from infrared_sdk import InfraredClient
from infrared_sdk.analyses.types import UtciModelRequest, UtciModelBaseRequest, AnalysesName
from infrared_sdk.models import TimePeriod, Location

POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [16.3750, 48.2100],
        [16.3950, 48.2100],
        [16.3950, 48.2250],
        [16.3750, 48.2250],
        [16.3750, 48.2100],
    ]],
}

LAT, LON = 48.215, 16.385
OUTPUT = Path(__file__).parent.parent / "backend" / "data" / "utci_leopoldstadt.npy"

tp = TimePeriod(
    start_month=7, start_day=1, start_hour=9,
    end_month=7, end_day=31, end_hour=18,
)

with InfraredClient() as client:
    # 1. Fetch area data
    print("Fetching buildings...")
    area = client.buildings.get_area(POLYGON)

    print("Fetching vegetation...")
    area_veg = client.vegetation.get_area(POLYGON)

    print("Fetching ground materials...")
    area_gm = client.ground_materials.get_area(POLYGON)

    # 2. Find nearest weather station
    print("Finding nearest weather station...")
    stations = client.weather.get_weather_file_from_location(lat=LAT, lon=LON, radius=50)
    station = stations[0]
    print(f"  Station: {station['fileName']}")

    # 3. Filter weather data for July
    print("Filtering weather data for July...")
    weather_data = client.weather.filter_weather_data(
        identifier=station["uuid"],
        time_period=tp,
    )

    # 4. Build UTCI request
    payload = UtciModelRequest.from_weatherfile_payload(
        payload=UtciModelBaseRequest(
            analysis_type=AnalysesName.thermal_comfort_index,
        ),
        location=Location(latitude=LAT, longitude=LON),
        time_period=tp,
        weather_data=weather_data,
    )

    # 5. Run simulation
    print("Running UTCI simulation (this may take a few minutes)...")
    result = client.run_area_and_wait(
        payload,
        POLYGON,
        buildings=area.buildings,
        vegetation=area_veg.features,
        ground_materials=area_gm.layers,
    )

    # 6. Print results
    print(f"\n--- Results ---")
    print(f"Grid shape:  {result.grid_shape}")
    print(f"min_legend:  {result.min_legend}")
    print(f"max_legend:  {result.max_legend}")
    print(f"Bounds:      {result.bounds}")
    succeeded = sum(1 for t in result.tiles if t.status == "succeeded") if hasattr(result, "tiles") and result.tiles else "n/a"
    print(f"Tiles succeeded: {succeeded}")

    # 7. Save grid
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, result.merged_grid)
    print(f"\nSaved grid -> {OUTPUT}")
