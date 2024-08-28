import numpy as np
import os

def one_hot_encode_labels(input_folder):
    categories = {
        "glioma_tumor": [1,0,0,0],
        "meningioma_tumor": [0,1,0,0],
        "no_tumor": [0,0,1,0],
        "pituitary_tumor": [0,0,0,1]
    } # Defines One-Hot enconding for all possible categories
    
    images = []
    labels = []

    for category, one_hot in categories.items(): # categories.items() returns a list of 2-tuples containing a label/category and its respective one-hot encoding. The for loop means that a category and its one-hot enconding label are taken once at a time sequentially
        category_folder = os.path.join(input_folder, category) # adds to the input_folder path the name of the category which is the same as the name of the sub_folder that contains the necessary npy files
        if not os.path.exists(category_folder): # if it does not exist, it continues to a new iteration in the loop
            continue

        for filename in os.listdir(category_folder):
            if filename.endswith(".npy"):
                img_path = os.path.join(category_folder, filename)
                img_matrix = np.load(img_path)

                images.append(img_matrix) # Adds the matrix representation of an image to the images' list
                labels.append(one_hot) # Adds the respective one_hot encoded label to the labels' list

                print(f"Loaded {filename} from {category_folder}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

input_folder = r"C:\Users\emida\OneDrive\Escritorio\Research Project\Medical Imaging Artificial Intelligence Interpreter\Numpy Arrays of Resized MRI Brain Tumour Dataset\Training"

images, labels = one_hot_encode_labels(input_folder)

if images.size == 0 or labels.size == 0: # x.size gives you the cardinality of the set x
    print("No images or labels were loaded. Check folder structure and file names")
else:
    np.save("training_images.npy", images) # first argument specifies the name of the file to be saved and the second argument mentions the data that will be stored in this file.
    np.save("training_labels.npy", labels)
    print("One-hot encoding complete. Arrays saved to the Workspace")