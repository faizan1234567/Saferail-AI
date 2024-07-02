import os
import sys

def write_image_names_to_file(image_directory, output_file):
    # Define common image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'}

    with open(output_file, 'w') as file:
        for filename in os.listdir(image_directory):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                file.write(filename + '\n')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python write_image_names.py <image_directory> <output_file>")
        sys.exit(1)

    image_directory = sys.argv[1]

    if not os.path.isdir(image_directory):
        print(f"Error: The directory {image_directory} does not exist.")
        sys.exit(1)

    # create meta dir
    meta_dir_path = os.path.join(image_directory, 'meta')
    os.makedirs(meta_dir_path, exist_ok = True)

    # pred.txt path
    if os.path.exists(meta_dir_path):
        pred_file_path = os.path.join(meta_dir_path, 'pred.txt')
        
        vi_images_dir = os.path.join(image_directory, 'vi')
        write_image_names_to_file(vi_images_dir, pred_file_path)

        print(f"Image names have been written to {pred_file_path}")
    else:
        print('Incorrect file path')
