import tensorflow as tf
import tensorflow_datasets as tfds

# Load Entire KITTI dataset to start working with the data 
kitti = tfds.load(
    'kitti',
    split='train[:1000]',  # Only load the first 1000 examples
    as_supervised=False,
    shuffle_files=True
)



# Explore the dataset structure
for example in kitti.take(1):
    print(example.keys())
    
    # Access the image and labels
    image = example['image']
    labels = example['objects']
    
    # Display an example image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.show()


