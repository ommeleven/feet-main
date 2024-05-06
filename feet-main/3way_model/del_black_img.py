import os
from PIL import Image

def is_black_image(image_path, threshold=0.95):
    """
    Check if an image is predominantly black.
    """
    image = Image.open(image_path).convert("L")
    width, height = image.size
    pixels = image.load()

    black_count = 0
    total_pixels = width * height

    for y in range(height):
        for x in range(width):
            if pixels[x, y] < 25:  # Check if pixel is almost black
                black_count += 1

    if black_count / total_pixels > threshold:
        return True  # Image is predominantly black
    else:
        return False  # Image is not predominantly black

def delete_black_images(folder):
    """
    Delete black images from the specified folder.
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            image_path = os.path.join(root, file)
            if is_black_image(image_path):
                os.remove(image_path)
                print(f"Deleted black image: {image_path}")

def main():
    healthy_folder = '/Users/HP/src/feet_fracture_data/864/all_images/healthy'
    delete_black_images(healthy_folder)

if __name__ == '__main__':
    main()
