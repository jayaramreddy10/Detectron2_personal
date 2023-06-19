import numpy as np
import json
from pycocotools import mask

def load_instance_masks(file_path):
    with open(file_path, 'r') as file:
        coco_annotations = json.load(file)
    
    masks = []
    
    for annotation in coco_annotations['annotations']:
        rle_mask = annotation['segmentation']
        mask_ = rle_to_binary_mask(rle_mask)
        masks.append(mask_)
    
    return masks

def rle_to_binary_mask(rle):
    compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    return mask.decode(compressed_rle)
    
# Example usage:
file_path = '/home/jayaram/research/research_tracks/multibody_slam/instance_segmentation_detectron2/detectron2/demo_images/masks_info_folder/messi_wc.json'
retrieved_masks = load_instance_masks(file_path)

# Printing the retrieved masks
for i, mask_ in enumerate(retrieved_masks):
    print(f'Mask {i+1}:')
    print(mask_.shape)
    ones_indices = np.argwhere(mask_ == 1)
    print('first and last indices where elements are 1: {}, {}'.format(ones_indices[0], ones_indices[-1]))
    print()