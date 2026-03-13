import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ensure the output directory exists
output_dir = os.path.join('results', 'visualise')
os.makedirs(output_dir, exist_ok=True)

# Model name mapping
model_names = {
    'baseline': 'Baseline',
    'LogicH': 'LogicH'
}

# Paths
base_path = 'results'
folders = [
    ('test', 'baseline'),
    ('test', 'LogicH'),
    ('val', 'baseline'),
    ('val', 'LogicH')
]

# Pre-select gestures and clips for each split to ensure consistency
split_selections = {}
for split in ['test', 'val']:
    # Get all files for this split across models to find common gestures
    all_files = []
    for model in ['baseline', 'LogicH']:
        folder_path = os.path.join(base_path, split, model)
        files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f.count('__') >= 2 and not f.startswith('confusion_matrix')]
        all_files.extend(files)
    
    # Extract gestures and clips
    gesture_clip = {}
    for f in all_files:
        parts = f.split('__')
        gesture = parts[0]
        clip = parts[1]
        key = f"{gesture}__{clip}"
        if gesture not in gesture_clip:
            gesture_clip[gesture] = []
        if key not in gesture_clip[gesture]:
            gesture_clip[gesture].append(key)
    
    # Randomly select 2 different gestures
    gestures = list(gesture_clip.keys())
    if len(gestures) < 2:
        raise ValueError(f"Not enough different gestures in {split}")
    selected_gestures = random.sample(gestures, 2)
    
    # For each selected gesture, randomly select one clip
    selected_keys = []
    for gesture in selected_gestures:
        clips = gesture_clip[gesture]
        selected_key = random.choice(clips)
        selected_keys.append(selected_key)
    
    split_selections[split] = selected_keys

for split, model in folders:
    folder_path = os.path.join(base_path, split, model)
    model_name = model_names[model]
    output_path = os.path.join(output_dir, f"{split}_{model_name}.png")
    
    # Get the selected keys for this split
    selected_keys = split_selections[split]
    
    # Find the corresponding files in this model's folder
    selected_files = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f.count('__') >= 2 and not f.startswith('confusion_matrix')]
    for key in selected_keys:
        matching_files = [f for f in files if f.startswith(key)]
        if matching_files:
            selected_files.append(os.path.join(folder_path, matching_files[0]))
    
    if len(selected_files) != 2:
        raise ValueError(f"Could not find matching files for {split} {model}")
    
    # Create a 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, file_path in enumerate(selected_files):
        img = mpimg.imread(file_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        # Add title with gesture and clip
        parts = os.path.basename(file_path).split('__')
        title = f"{parts[0]} {parts[1]}"
        axes[i].set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Generated {output_path}")
