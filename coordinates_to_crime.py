import geopandas as gpd
from shapely.geometry import Point

# Load the Toronto neighborhoods GeoJSON
toronto_neighborhoods = gpd.read_file("neighbourhoods.geojson")

# Example coordinate (latitude, longitude)
latitude, longitude = 43.6532, -79.3832  # Example: Downtown Toronto

# Convert to a Point geometry
point = Point(longitude, latitude)

# Find the neighborhood that contains the point
neighborhood = toronto_neighborhoods[toronto_neighborhoods.contains(point)]

# Print the result
if not neighborhood.empty:
    print(f"The coordinate is in: {neighborhood.iloc[0]['AREA_NAME']}")
else:
    print("Coordinate is not within any known Toronto neighborhood.")
