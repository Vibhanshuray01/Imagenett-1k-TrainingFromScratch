import os
import shutil
from tqdm import tqdm

def organize_validation_data(val_dir):
    """Organizes validation data into class folders matching training structure."""
    # Get class names from training directory
    train_dir = os.path.join(os.path.dirname(val_dir), 'train')
    class_names = sorted(os.listdir(train_dir))
    
    print(f"Found {len(class_names)} classes in training data")
    
    # Create temporary directory for organization
    temp_dir = val_dir + '_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move all validation files to temp directory
    print("Moving files to temporary directory...")
    for file in os.listdir(val_dir):
        if file.endswith('.JPEG'):
            shutil.move(os.path.join(val_dir, file), os.path.join(temp_dir, file))
    
    # Create class directories
    print("Creating class directories...")
    for class_name in tqdm(class_names):
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # Read ground truth file and move images
    print("Moving files to class directories...")
    with open('ILSVRC2012_validation_ground_truth.txt', 'r') as f:
        ground_truth = [int(line.strip()) for line in f.readlines()]
    
    for idx, class_idx in enumerate(tqdm(ground_truth)):
        img_name = f'ILSVRC2012_val_{idx+1:08d}.JPEG'
        src = os.path.join(temp_dir, img_name)
        dst = os.path.join(val_dir, class_names[class_idx-1], img_name)
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print("Validation data organization complete!")

if __name__ == "__main__":
    val_dir = "/media/data/ILSVRC/Data/CLS-LOC/val"
    organize_validation_data(val_dir) 