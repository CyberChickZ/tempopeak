import json
import os

d = json.load(open("/Users/harryzhang/git/tempopeak/outputs/sam3_mask_extractor/00001.json"))
keys = [int(k) for k in d.keys()]
print(f"JSON has {len(d)} frames.")
print(f"First 10 keys: {sorted(keys)[:10]}")
