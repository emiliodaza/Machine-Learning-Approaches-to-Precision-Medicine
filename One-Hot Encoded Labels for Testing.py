import numpy as np
import os

# the code works in the same way as the one in One-Hot Encoded Labels for Training.py but in this case it is for testing.

def one_hot_encode_labels(input_folder):
    categories = {
        "glioma_tumor": [1,0,0,0],
        "meningioma_tumor": [0,1,0,0],
        "no_tumor": [0,0,1,0],
        "pituitary_tumor": [0,0,0,1]
    }
    
    images = []
    labels = []

    for category, one_hot in categories.items():
        category_folder = os.path.join(input_folder, category)
        if not os.path.exists(category_folder):
            continue

        for filename in os.listdir(category_folder):
            if filename.endswith(".npy"):
                img_path = os.path.join(category_folder, filename)
                img_matrix = np.load(img_path)

                images.append(img_matrix)
                labels.append(one_hot)

                print(f"Loaded {filename} from {category_folder}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

input_folder = r"C:\Users\emida\OneDrive\Escritorio\Research Project\Medical Imaging Artificial Intelligence Interpreter\Numpy Arrays of Resized MRI Brain Tumour Dataset\Testing"

images, labels = one_hot_encode_labels(input_folder)

if images.size == 0 or labels.size == 0:
    print("No images or labels were loaded. Check folder structure and file names")
else:
    np.save("testing_images.npy", images)
    np.save("testing_labels.npy", labels)
    print("One-hot encoding complete. Arrays saved to disk")