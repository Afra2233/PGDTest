import os
import json
import shutil

# path ####
val_dir = "/nobackup/projects/bdlan08/jzhang89/PGDTest/data/imagenet1k/val/val_images"
ground_truth_dir ="/nobackup/projects/bdlan08/jzhang89/PGDTest/data/imagenet1k/val/ILSVRC2012_validation_ground_truth.txt"
json_dir = "/nobackup/projects/bdlan08/jzhang89/PGDTest/data/imagenet1k/val/imagenet_class_index.json"
output_dir ="/nobackup/projects/bdlan08/jzhang89/PGDTest/data/imagenet1k/val/sorted_images"

## mapping load ##
with open(ground_truth_dir,'r') as f:
    ground_truth = [int(x.strip()) for x in  f.readlines()]

with open(json_dir,'r') as f :

    class_index = json.load(f)

index_id ={int(k):v[0] for k,v in class_index.items()}
img_files = sorted(os.listdir(val_dir))
assert len(img_files) == len(ground_truth), "the lenght of image fils is not same as that of ground truth file "

os.makedirs(output_dir, exist_ok=True)

for idx, (img_file,label) in enumerate (zip(img_files,ground_truth)):
    synset =index_id[label]
   
    synset_dir = os.path.join(output_dir, synset)
    os.makedirs(synset_dir, exist_ok=True)

    
    src_path = os.path.join(val_dir, img_file)
    dst_path = os.path.join(output_dir, img_file)
    shutil.move(val_dir, dst_path) 
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1} images...")

print("Done!")





# img_files = sorted ([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])

# assert len(img_files) == len(ground_truth), "Mismatch between number of images and labels"

# class_to_images = {}

# for i, img_file in enumerate(img_files):
#     class_idx = ground_truth[i] - 1  # labels in file are 1-based
#     class_id = idx_to_class[class_idx]
    
#     if class_id not in class_to_images:
#         class_to_images[class_id] = []
    
#     class_to_images[class_id].append(img_file)

# with open (output_dir )

                   
