import os
import requests

output_folder = "cv_stress_test_images"
os.makedirs(output_folder, exist_ok=True)

# 20 curated edge-case images using production asset IDs
images = [
    # People
    {"name": "people_extreme_crowd.jpg", "id": "photo-1506157786151-b8491531f063"},
    {"name": "people_heavy_motion_blur.jpg", "id": "photo-1502224562085-639556652f33"},
    {"name": "people_low_light_silhouette.jpg", "id": "photo-1517457373958-b7bdd4587205"},
    {"name": "people_heavy_occlusion.jpg", "id": "photo-1519671482749-fd09be7ccebf"},
    {"name": "people_reflection_glass.jpg", "id": "photo-1522158673376-3c851142ef4d"},
    {"name": "people_overhead_drone_view.jpg", "id": "photo-1539635278303-d4002c07eae3"},
    {"name": "people_underwater_distortion.jpg", "id": "photo-1544551763-46a013bb70d5"},
    # Vehicles
    {"name": "vehicle_heavy_rain_night.jpg", "id": "photo-1492144534655-ae79c964c9d7"},
    {"name": "vehicle_dense_traffic_fog.jpg", "id": "photo-1473841623126-734d5415c9ff"},
    {"name": "vehicle_abstract_motion_blur.jpg", "id": "photo-1508962914676-134849a727f0"},
    {"name": "vehicle_extreme_close_up.jpg", "id": "photo-1552519507-da3b142c6e3d"},
    {"name": "vehicle_lens_flare_glare.jpg", "id": "photo-1568605117036-5fe5e7bab0b7"},
    {"name": "vehicle_stacked_junkyard.jpg", "id": "photo-1533234433722-e221598cc004"},
    {"name": "vehicle_heavy_snow_covered.jpg", "id": "photo-1482862549707-f63cb32c5fd9"},
    # Animals
    {"name": "animal_perfect_camouflage.jpg", "id": "photo-1575550959106-5a7defe28b56"},
    {"name": "animal_fast_action_blur.jpg", "id": "photo-1534447677768-be436bb09401"},
    {"name": "animal_underwater_murky.jpg", "id": "photo-1546026423-cc4642628d2b"},
    {"name": "animal_night_infrared_look.jpg", "id": "photo-1549488344-1f9b8d2bd1f3"},
    {"name": "animal_extreme_macro_occlusion.jpg", "id": "photo-1561037404-61cd46aa615b"},
    {"name": "animal_herd_high_density.jpg", "id": "photo-1516426122078-c23e76319801"}
]

# Adding generic browser headers + forcing the raw photo parameter layout
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
}

print(f"🚀 Downloading 20 images directly into './{output_folder}'...")

for i, img in enumerate(images, 1):
    file_path = os.path.join(output_folder, img["name"])
    
    # Constructing a clean download link using Unsplash's dynamic file-rendering structure
    direct_download_url = f"https://unsplash.com{img['id']}?q=80&w=800&auto=format&fit=crop"
    
    try:
        res = requests.get(direct_download_url, headers=headers, timeout=15)
        if res.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(res.content)
            print(f"[{i}/20] Saved: {img['name']}")
        else:
            print(f"❌ Failed: {img['name']} (HTTP {res.status_code})")
    except Exception as e:
        print(f"❌ Error downloading {img['name']}: {e}")

print("\n✅ Done! Check your local folder now.")
