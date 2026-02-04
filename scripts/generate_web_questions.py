import os
import json

data_dir = "/Users/emirhan/Desktop/LeGrad-1/web_application/data"
output_file = "/Users/emirhan/Desktop/LeGrad-1/web_application/data.js"

questions = []

image_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

for img_id in image_ids:
    img_path = os.path.join(data_dir, img_id)
    
    # Get names
    target_1_name = ""
    target_2_name = ""
    target_fake_name = ""
    
    def get_name(target_dir):
        files = os.listdir(os.path.join(img_path, target_dir))
        for f in files:
            if f.startswith("target_") and f.endswith(".txt"):
                return f[7:-4].capitalize()
        return "Unknown"

    target_1_name = get_name("Target_1")
    target_2_name = get_name("Target_2")
    target_fake_name = get_name("Target_fake")

    targets = [
        {"folder": "Target_1", "index": 0, "name": target_1_name},
        {"folder": "Target_2", "index": 1, "name": target_2_name},
        {"folder": "Target_fake", "index": 2, "name": "None of them"}
    ]

    for target in targets:
        target_path = os.path.join(img_path, target["folder"])
        rel_img_path = f"data/{img_id}/{target['folder']}/original.png"
        
        heatmaps = [
            {"file": "legrad.png", "method": "LeGrad"},
            {"file": "legrad_omp.png", "method": "LeGrad OMP"}
        ]
        
        for hm in heatmaps:
            rel_hm_path = f"data/{img_id}/{target['folder']}/{hm['file']}"
            
            questions.append({
                "id": len(questions),
                "image_id": img_id,
                "target_type": target["folder"],
                "original_image": rel_img_path,
                "heatmap_image": rel_hm_path,
                "options": [
                    target_1_name,
                    target_2_name,
                    "None of them"
                ],
                "correct_index": target["index"],
                "correct_name": target["name"],
                "method": hm["method"]
            })

with open(output_file, "w") as f:
    f.write("const QUESTIONS_DATA = ")
    json.dump(questions, f, indent=2)
    f.write(";")

print(f"Generated {len(questions)} questions in data.js")
