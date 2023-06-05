import os
import cv2

from random import sample
from matplotlib import pyplot as plt
from numpy import ceil
from seaborn import barplot

from utils import detect_obj


def plot_train_images(images_path, labels_path, num_images=32):
    # Get a list of all the image files in the training images directory
    image_files = os.listdir(images_path)

    # Choose 16 random image files from the list
    random_images = sample(image_files, num_images)

    num_rows = int(ceil(num_images / 4))
    num_cols = min(num_images, 4)

    # Set up the plot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))

    # Loop over the random images and plot the object detections
    for i, image_file in enumerate(random_images):
        row = i // num_cols
        col = i % num_cols

        # Load the image
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)

        # Load the labels for this image
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")

        # Loop over the labels and plot the object detections
        for label in labels:
            if len(label.split()) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, label.split())
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        # Show the image with the object detections
        if i < len(random_images):
            axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[row, col].grid(False)
            axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_metrics(metrics):
    # Create the barplot
    ax = barplot(x=['mAP50-95', 'mAP50', 'mAP75'], y=[metrics.box.map, metrics.box.map50, metrics.box.map75])
    # Set the title and axis labels
    ax.set_title('YOLO Evaluation Metrics')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')

    # Set the figure size
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    # Add the values on top of the bars
    for p in ax.patches:
        ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center',
                    va='bottom')

    # Show the plot
    plt.show()


def evaluate_model_on_test_images(model, images_path, num_images=64):
    # Define the directory where the custom images are stored
    custom_image_dir = images_path

    # Get the list of image files in the directory
    image_files = os.listdir(custom_image_dir)

    # Select 16 random images from the list
    selected_images = sample(image_files, num_images)

    num_rows = int(ceil(num_images / 4))
    num_cols = min(num_images, 4)

    # Create a figure with subplots for each image
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 16))

    # Iterate over the selected images and plot each one
    for i, img_file in enumerate(selected_images):
        # Compute the row and column index of the current subplot
        row = i // num_cols
        col = i % num_cols

        # Load the current image and run object detection
        img_path = os.path.join(custom_image_dir, img_file)
        detect_img = detect_obj(model, img_path)

        # Plot the current image on the appropriate subplot
        axes[row, col].imshow(detect_img)
        axes[row, col].axis('off')

    # Adjust the spacing between the subplots
    plt.tight_layout()
    plt.show()
