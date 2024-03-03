import pandas as pd
from geopy.geocoders import MapQuest
from geopy.exc import GeocoderTimedOut

# Load the dataset
file_path = "TG_XLSX_20234_caka2_train.xlsx"
df = pd.read_excel(file_path)

# Specify the column containing addresses
address_column = "Adreses pieraksts"

# Initialize geocoder
geolocator = MapQuest(api_key="EewT3a7KMMODr0Bi5xjPShjyM0R03DMl")

# Function to get latitude and longitude from address
def get_lat_long(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return get_lat_long(address)  # Retry in case of timeout

# Apply the function to get latitude and longitude for each address
df["Latitude"], df["Longitude"] = zip(*df[address_column].apply(get_lat_long))

# Save the updated dataframe to a new Excel file
output_file_path = "Caka_train_longLat.xlsx"
df.to_excel(output_file_path, index=False)

print("Latitude and longitude coordinates added and file saved successfully.")
