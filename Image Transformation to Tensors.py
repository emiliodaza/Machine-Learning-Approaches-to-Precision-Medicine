from PIL import Image
import numpy as np
import os

def convert_images_to_matrices(input_folder, output_folder, size = (512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) # this creates the output_folder
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("L") # Converts image to grayscale "L" means Luminance 0 means black and 255 pure white
            # The following ensures the image is 512 x 512. If it's already 512 x 512 there will be no operation performed otherwise it will be resized
            img = img.resize(size, Image.LANCZOS) # Remember, this is img.resize not img.thumbnail
            # The following converts image to numpy array
            img_matrix = np.array(img)

            matrix_filename = os.path.splitext(filename)[0] + ".npy" # os.path.splitext(filename) splits the filename into the filename without the extension and the second part is the extension. [0] selects the first item, ".npy" appends it to the filename without extension
            np.save(os.path.join(output_folder, matrix_filename), img_matrix) # the fist argument of np.save is the path where it will be saved and the second argument is what to save

            print(f"Converted and saved {filename} to {matrix_filename}")

input_folder = r"C:\Users\emida\OneDrive\Escritorio\Research Project\Medical Imaging Artificial Intelligence Interpreter\Resized MRI Brain Tumour Dataset\Testing\glioma_tumor"
output_folder = r"C:\Users\emida\OneDrive\Escritorio\Research Project\Medical Imaging Artificial Intelligence Interpreter\Numpy Arrays of Resized MRI Brain Tumour Dataset\Testing\glioma_tumor"

# The process was done for all the folders of categories

convert_images_to_matrices(input_folder, output_folder)