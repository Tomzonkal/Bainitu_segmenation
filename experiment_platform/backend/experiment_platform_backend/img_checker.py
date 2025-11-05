import cv2
import numpy as np

def check_image_opencv(path):
    img = cv2.imread(path)  # Reads as BGR by default
    if img is None:
        print(f"Error: could not read {path}")
        return
    
    height, width = img.shape[:2]
    if len(img.shape) == 2:
        color_type = "Grayscale"
    elif img.shape[2] == 3:
        color_type = "RGB"
    else:
        color_type = f"Other ({img.shape[2]} channels)"

    if np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
        print("Image visually grayscale but stored as RGB.")
    else:
        print("Image has real color differences.")
    
    print(f"File: {path}")
    print(f" → Size: {width} x {height}")
    print(f" → Type: {color_type}")

# Example usage:
check_image_opencv(r"C:\Users\ja\Desktop\Bainitu_segmenation\data_labeling\src\results\slic_f5e1203b6c7a\797-DQ-full_superpixel_segment0.png")
