import os
from footfracture.config import DATA_DIR, RESULTS_DIR
from PIL import Image, ImageDraw

def draw_bounding_box(image_path, x, y, width, height, output_path):
    # Open an image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    print(img_width, img_height)
    
    # Calculate bounding box coordinates
    x = int(x * img_width)/100
    y = int(y * img_height)/100
    width = int(width * img_width)/100
    height = int(height * img_height)/100

    # Define the bounding box coordinates
    top_left = (x, y)
    bottom_right = (x + width, y + height)
    
    # Draw the red bounding box
    draw.rectangle([top_left, bottom_right], outline="red", width=4)
    
    # Save the image with the bounding box
    image.save(output_path)
    print(f"Image saved with bounding box at {output_path}")

if __name__ == "__main__":
    # Usage example (Label Studio coordinates are percentages of image size)
    draw_bounding_box(os.path.join(DATA_DIR, "P020 SAGT1/study/P020 SAGT1_016.jpg"), x=45.81372637, y=58.18604906, width=16.2479062, height=10.72026801, output_path=os.path.join(RESULTS_DIR, 'bbox_example.jpg'))
