import os
import shutil

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
    for root, dirs, files in os.walk(os.path.join(source_dir, 'training')):
        for directory in dirs:
            if directory not in ['misc', 'outliers']:
                class_name = directory
                for img_root, _, img_files in os.walk(os.path.join(root, directory)):
                    for img_file in img_files:
                        img_src_path = os.path.join(img_root, img_file)
                        img_dst_path = os.path.join(train_images_dir, img_file)
                        shutil.copyfile(img_src_path, img_dst_path)
                        label_dst_path = os.path.join(train_labels_dir, img_file.replace('.', '_') + '.txt')
                        with open(label_dst_path, 'w') as label_file:
                            label_file.write(class_name)

    # Transform validation data
    for root, dirs, files in os.walk(os.path.join(source_dir, 'validation')):
        for directory in dirs:
            if directory not in ['misc', 'outliers']:
                class_name = directory
                for img_root, _, img_files in os.walk(os.path.join(root, directory)):
                    for img_file in img_files:
                        img_src_path = os.path.join(img_root, img_file)
                        img_dst_path = os.path.join(val_images_dir, img_file)
                        shutil.copyfile(img_src_path, img_dst_path)
                        label_dst_path = os.path.join(val_labels_dir, img_file.replace('.', '_') + '.txt')
                        with open(label_dst_path, 'w') as label_file:
                            label_file.write(class_name)

# Example usage:
transform_dataset('/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26', '/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted')
