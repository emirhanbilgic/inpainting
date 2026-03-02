import json
import os

# Paths
META_PATH = os.path.abspath('scripts/all_classes_meta.json')
DICT_PATH = os.path.abspath('scripts/visual_concept_dictionary_gpt.json')

# Load metadata
with open(META_PATH, 'r') as f:
    all_meta = json.load(f)

# Load existing dictionary (or empty)
if os.path.exists(DICT_PATH):
    with open(DICT_PATH, 'r') as f:
        visual_dict = json.load(f)
else:
    visual_dict = {}

# Determine start index (number of already processed classes)
processed = len(visual_dict)

# Number of classes to generate in this batch
BATCH_SIZE = 50

# Simple placeholder concept generators
def generate_concepts(class_name):
    # Visual confusers (15)
    visual_confusers = [
        f"{class_name} silhouette",
        f"{class_name} shadow",
        f"{class_name} outline",
        f"{class_name} pattern",
        f"{class_name} texture",
        f"{class_name} fur",
        f"{class_name} paw",
        f"{class_name} tail",
        f"{class_name} ear",
        f"{class_name} eye",
        f"{class_name} nose",
        f"{class_name} mouth",
        f"{class_name} claw",
        f"{class_name} whisker",
        f"{class_name} spot"
    ]
    # Co-occurring context (15)
    co_occurring_context = [
        "grass",
        "water",
        "forest",
        "sky",
        "sand",
        "rock",
        "tree",
        "leaf",
        "snow",
        "mountain",
        "cave",
        "urban",
        "road",
        "fence",
        "bench"
    ]
    # Semantic hierarchy (10): 5 hypernyms, 5 hyponyms
    hypernyms = ["animal", "mammal", "vertebrate", "living thing", "organism"]
    hyponyms = [
        f"{class_name} juvenile",
        f"{class_name} adult",
        f"{class_name} male",
        f"{class_name} female",
        f"{class_name} wild"
    ]
    semantic_hierarchy = hypernyms + hyponyms
    return {
        "visual_confusers": visual_confusers,
        "co_occurring_context": co_occurring_context,
        "semantic_hierarchy": semantic_hierarchy
    }

# Process next batch
new_entries = {}
for entry in all_meta[processed:processed + BATCH_SIZE]:
    wnid = entry["wnid"]
    name = entry["name"]
    new_entries[wnid] = generate_concepts(name)

# Update dictionary
visual_dict.update(new_entries)

# Write back to file (pretty printed)
with open(DICT_PATH, 'w') as f:
    json.dump(visual_dict, f, indent=4, ensure_ascii=False)

print(f"Generated concepts for {len(new_entries)} classes. Updated {DICT_PATH}")
