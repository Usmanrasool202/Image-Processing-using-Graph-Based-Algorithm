import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform image segmentation using grabCut
def grabcut_segmentation(image_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 2: Initialize mask and background/foreground models
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Step 3: Define rectangular ROI (can be fine-tuned)
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)
    
    # Step 4: Apply grabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Step 5: Prepare binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Step 6: Create segmented image
    segmented_img = img_rgb * mask2[:, :, np.newaxis]
    
    return img_rgb, segmented_img  # Return both for display externally
