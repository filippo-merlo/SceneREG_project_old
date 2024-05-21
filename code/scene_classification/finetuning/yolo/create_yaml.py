import os
import shutil

def create_annotation_file(image_name, class_name, output_dir):
    annotation_file_path = os.path.join(output_dir, image_name + '.txt')
    with open(annotation_file_path, 'w') as annotation_file:
        annotation_file.write(class_name)

def transform_dataset(source_dir, target_dir):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Create train and validation directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # Create images and labels directories inside train and val directories
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    val_images_dir = os.path.join(val_dir, 'images')
    val_labels_dir = os.path.join(val_dir, 'labels')
    os.makedirs(train_images_dir)
    os.makedirs(train_labels_dir)
    os.makedirs(val_images_dir)
    os.makedirs(val_labels_dir)

    # Transform training data
    for root, dirs, files in os.walk(os.path.join(source_dir, 'images/training')):
        for directory in dirs:
            class_name = directory
            class_dir_path = os.path.join(root, directory)
            for subdir, _, img_files in os.walk(class_dir_path):
                for img_file in img_files:
                    img_src_path = os.path.join(subdir, img_file)
                    img_dst_path = os.path.join(train_images_dir, img_file)
                    shutil.move(img_src_path, img_dst_path)
                    create_annotation_file(img_file, class_name, train_labels_dir)

    # Transform validation data
    for root, dirs, files in os.walk(os.path.join(source_dir, 'images/validation')):
        for directory in dirs:
            class_name = directory
            class_dir_path = os.path.join(root, directory)
            for subdir, _, img_files in os.walk(class_dir_path):
                for img_file in img_files:
                    img_src_path = os.path.join(subdir, img_file)
                    img_dst_path = os.path.join(val_images_dir, img_file)
                    shutil.move(img_src_path, img_dst_path)
                    create_annotation_file(img_file, class_name, val_labels_dir)
                    
# Example usage:
transform_dataset('/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26', '/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted')
