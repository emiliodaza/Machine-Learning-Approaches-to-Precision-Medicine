from PIL import Image
import os

def resize_and_pad_image(input_folder, output_folder, size = (512, 512), bg_color = (0,0,0)): # size = (512, 512) means 512 pixels by 512 pixels as the desired resized images # bg_color means background color
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) # if the mentioned output_folder does not exist it is created

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path) # loads the image

            img.thumbnail(size, Image.LANCZOS) # Image.LANCZOS is Lanczos interpolation which is a mathematical technique created by Cornelius Lanczos which helps to maintain the details of an image when downsampling (reduced in size) an image. This code makes the image fit within the specified size keeping its original aspect ratio

            new_img = Image.new("RGB", size, bg_color) # Creates a new image in RGB format and applies the specified size and bg_color
            new_img.paste( # new_img.paste pastes an image on top of another image
                img, ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2) # the // operator performs the division and rounds down to the nearest integer
            ) # ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2) indicates the position where the img will pasted on top of the new_img. It has to components because it represents the x and y coordinates of the top_left corner of img

            new_img.save(os.path.join(output_folder, filename)) # saves the image in the path specified by os.path.join(output_folder, filename) and also creates the file inside the folder
            print(f"Resized and saved {filename}")

input_folder = r"C:\Users\emida\OneDrive\Escritorio\Research Project\Medical Imaging Artificial Intelligence Interpreter\Original MRI Brain Tumour Dataset\Testing\glioma_tumor"
output_folder = r"C:\Users\emida\OneDrive\Escritorio\Research Project\Medical Imaging Artificial Intelligence Interpreter\Resized MRI Brain Tumour Dataset\Testing\glioma_tumor"

# The process was done for all the folders of categories

resize_and_pad_image(input_folder, output_folder)