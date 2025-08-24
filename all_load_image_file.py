import cv2 as cv
import os

valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def all_load_image_file(path):
    images = []
    image_names = []

    if not os.path.exists(path):
        print(f"📂 Path not found: {path}")
        return (images, image_names)

    if os.path.isfile(path):
        file_name, file_extension = os.path.splitext(os.path.basename(path))
        if file_extension.lower() in valid_extensions:
            image = cv.imread(path)
            if image is None:
                print(f"🔴 Failed to load image: {file_name}")
            else:
                images.append(image)
                image_names.append(file_name)
                print(f"🟢 Loaded image: {file_name}")
        return (images, image_names)
    
    for root, _, files in os.walk(path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension.lower() in valid_extensions:
                full_path = os.path.join(root, file)
                image = cv.imread(full_path)
                if image is None:
                    print(f"🔴 Failed to load image: {file_name}")
                else:
                    images.append(image)
                    image_names.append(file_name)
                    print(f"🟢 Loaded image: {file_name}")

    return (images, image_names)