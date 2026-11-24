import requests
import pandas as pd
import time

from pathlib import Path
from datetime import datetime, timezone


# ==========================================
# 1. API URL
# ==========================================

URL = "https://overpass-api.de/api/interpreter"


# ==========================================
# 2. Geographic boundaries: Algiers
# ==========================================

query = """
[out:json][timeout:60];
(
  nwr["amenity"="doctors"](36.55,2.95,36.85,3.35);
  nwr["healthcare"="doctor"](36.55,2.95,36.85,3.35);
  nwr["amenity"="clinic"](36.55,2.95,36.85,3.35);
  nwr["healthcare"="clinic"](36.55,2.95,36.85,3.35);
);
out center tags;
"""


# ==========================================
# 3. Send API request with retry mechanism
# ==========================================

print("Connecting to OpenStreetMap...")

headers = {
    "User-Agent": "DoctorsDirectoryAlgeria/1.0 (educational project)"
}

response = None

for attempt in range(4):
    try:
        response = requests.post(
            URL,
            data={"data": query},
            headers=headers,
            timeout=120
        )

        if response.status_code == 200:
            print("Connection successful!")
            break

        if response.status_code in [429, 502, 503, 504]:
            wait = 15 * (2 ** attempt)
            print(
                f"Server unavailable: {response.status_code}. "
                f"Waiting {wait} seconds..."
            )
            if attempt < 3:
                time.sleep(wait)
            else:
                response.raise_for_status()

    except requests.exceptions.RequestException as error:
        print(f"Request error: {error}")
        if attempt < 3:
            time.sleep(15 * (2 ** attempt))
else:
    raise RuntimeError(
        "Could not connect to Overpass API after 4 attempts."
    )

response.raise_for_status()


# ==========================================
# 4. Extract JSON data
# ==========================================

result = response.json()
elements = result.get("elements", [])

print("Number of elements:", len(elements))


# ==========================================
# 5. Extract useful information
# ==========================================

records = []

for element in elements:
    tags = element.get("tags", {})

    # Extract coordinates
    latitude = element.get("lat")
    longitude = element.get("lon")

    # Ways and relations may have center coordinates
    if latitude is None or longitude is None:
        center = element.get("center", {})
        latitude = center.get("lat")
        longitude = center.get("lon")

    # Extract information
    records.append({
        "osm_id": element.get("id"),
        "osm_type": element.get("type"),
        "name": tags.get("name"),
        "specialty": tags.get("healthcare:speciality"),
        "facility_type": (
            tags.get("amenity")
            or tags.get("healthcare")
        ),
        "phone": (
            tags.get("phone")
            or tags.get("contact:phone")
        ),
        "address": (
            tags.get("addr:street")
            or tags.get("addr:full")
        ),
        "commune": (
            tags.get("addr:city")
            or tags.get("addr:suburb")
            or tags.get("addr:district")
        ),
        "website": (
            tags.get("website")
            or tags.get("contact:website")
        ),
        "latitude": latitude,
        "longitude": longitude,
        "source": "OpenStreetMap",
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat()
    })


# ==========================================
# 6. Create DataFrame
# ==========================================

df = pd.DataFrame(records)


# ==========================================
# 7. Basic data quality checks
# ==========================================

if not df.empty:
    df = df.drop_duplicates(
        subset=["osm_type", "osm_id"]
    )
    print("\nMissing values:")
    print(df.isnull().sum())
else:
    print("Warning: No medical places found.")


# ==========================================
# 8. Create output folder
# ==========================================

output_folder = Path("data/raw")
output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 9. Export CSV
# ==========================================

output_file = output_folder / "algiers_medical_places.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. Display results
# ==========================================

print("\nFirst 5 records:")
print(df.head())

print("\nTotal records:", len(df))

print("\nSaved successfully to:", output_file)
