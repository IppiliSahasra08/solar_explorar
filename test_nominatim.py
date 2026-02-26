from geopy.geocoders import Nominatim
import time

def test_indian_addresses():
    geolocator = Nominatim(user_agent="solar_explorar_test")
    
    addresses = [
        "Taj Mahal, Agra",
        "Cyber Hub, Gurgaon",
        "Marine Drive, Mumbai",
        "Hampi, Karnataka",
        "Munnar, Kerala"
    ]
    
    print(f"{'Address':<30} | {'Latitude':<10} | {'Longitude':<10}")
    print("-" * 55)
    
    for addr in addresses:
        try:
            location = geolocator.geocode(addr)
            if location:
                print(f"{addr:<30} | {location.latitude:<10.6f} | {location.longitude:<10.6f}")
            else:
                print(f"{addr:<30} | Not Found")
            # Nominatim policy: 1 request per second
            time.sleep(1.1)
        except Exception as e:
            print(f"Error geocoding {addr}: {e}")

if __name__ == "__main__":
    test_indian_addresses()
