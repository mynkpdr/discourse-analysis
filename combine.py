import json
import glob

combined_items = []

for file in sorted(glob.glob("data/page_*.json")):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    combined_items.extend(data.get("directory_items", []))

with open("combined.json", "w", encoding="utf-8") as f:
    json.dump({"directory_items": combined_items}, f, indent=2)

print(f"Saved {len(combined_items)} items to combined.json")
