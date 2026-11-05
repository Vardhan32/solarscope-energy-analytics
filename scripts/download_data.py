import os
import urllib.request
import sys

BASE_URL_MAIN = "https://raw.githubusercontent.com/Rakshitha-Ireddi/Solar-Power-Generation-Forecast/master/"
BASE_URL_ALT = "https://raw.githubusercontent.com/louisgthier/solar-power-generation-prediction/main/"
BASE_URL_ALT2 = "https://raw.githubusercontent.com/kaivalpanchal/Solar-Panel-Power-Generation/main/"

FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(filename):
    dest = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"File already exists: {dest} ({os.path.getsize(dest)} bytes)")
        return True

    urls_to_try = [
        BASE_URL_MAIN + filename,
        BASE_URL_ALT + filename,
        BASE_URL_ALT2 + filename,
        f"https://raw.githubusercontent.com/eshan-kaul/Regression-Voting-Ensemble-for-Solar-Power-Prediction/main/{filename}"
    ]

    for url in urls_to_try:
        try:
            print(f"Attempting download for {filename} from {url}...")
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            if size > 1000:
                print(f"Successfully downloaded {filename}: {size} bytes")
                return True
            else:
                print(f"Downloaded file too small ({size} bytes), removing...")
                os.remove(dest)
        except Exception as e:
            print(f"Failed from {url}: {e}")
            if os.path.exists(dest):
                os.remove(dest)
    return False

def main():
    success_count = 0
    for f in FILES:
        if download_file(f):
            success_count += 1
        else:
            print(f"Error: Could not download {f}")
    
    print(f"Downloaded {success_count}/{len(FILES)} files successfully.")
    if success_count < len(FILES):
        sys.exit(1)

if __name__ == "__main__":
    main()
