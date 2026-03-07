#!/usr/bin/env python3
import json
import os

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DICT_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'visual_concept_dictionary_coco_gpt.json')

COCO_CLASSES = [
    "airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear",
    "bed", "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus",
    "cake", "car", "carrot", "cat", "cell phone", "chair", "clock", "couch", "cow", "cup",
    "dining table", "dog", "donut", "elephant", "fire hydrant", "fork", "frisbee", "giraffe",
    "hair drier", "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop",
    "microwave", "motorcycle", "mouse", "orange", "oven", "parking meter", "person", "pizza",
    "potted plant", "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink",
    "skateboard", "skis", "snowboard", "spoon", "sports ball", "stop sign", "suitcase",
    "surfboard", "teddy bear", "tennis racket", "tie", "toilet", "toothbrush", "traffic light",
    "train", "truck", "tv", "umbrella", "vase", "wine glass", "zebra"
]

def generate_concepts(class_name):
    """Generate concepts using the same template as the ImageNet GPT script."""
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

def main():
    print(f"Generating COCO GPT dictionary...")
    
    visual_dict = {}
    
    for class_name in COCO_CLASSES:
        visual_dict[class_name] = generate_concepts(class_name)
    
    # Write back to file (pretty printed)
    with open(DICT_PATH, 'w') as f:
        json.dump(visual_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Generated concepts for {len(visual_dict)} classes.")
    print(f"Saved to: {DICT_PATH}")

if __name__ == "__main__":
    main()
