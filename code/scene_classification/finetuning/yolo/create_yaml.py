import os
import shutil

def create_class_directories(base_dir, class_names):
    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def move_images_and_create_annotations(source_root, target_root, phase):
    for root, dirs, files in os.walk(os.path.join(source_root, f'images/{phase}')):
        for class_dir in dirs:
            class_path = os.path.join(root, class_dir)
            for subdir, _, img_files in os.walk(class_path):
                for img_file in img_files:
                    img_src_path = os.path.join(subdir, img_file)
                    class_target_dir = os.path.join(target_root, phase, class_dir)
                    img_dst_path = os.path.join(class_target_dir, img_file)
                    shutil.move(img_src_path, img_dst_path)

def get_class_names(source_dir):
    class_names = set()

    for root, dirs, files in os.walk(os.path.join(source_dir, 'images/training')):
        for directory in dirs:
            class_names.add(directory)

    for root, dirs, files in os.walk(os.path.join(source_dir, 'images/validation')):
        for directory in dirs:
            class_names.add(directory)

    return class_names

def transform_dataset(source_dir, target_dir):
    # Get all class names
    class_names = get_class_names(source_dir)

    # Create target directory structure
    for phase in ['train', 'val']:
        phase_dir = os.path.join(target_dir, phase)
        if not os.path.exists(phase_dir):
            os.makedirs(phase_dir)
        create_class_directories(phase_dir, class_names)

    # Move images and create annotations
    for phase in ['train', 'val']:
        move_images_and_create_annotations(source_dir, target_dir, phase)


# Example usage:
transform_dataset('/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26', '/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted')

#import os
#
#def get_class_names(source_dir):
#    class_names = set()
#
#    # Get class names from training data
#    for root, dirs, files in os.walk(os.path.join(source_dir, 'images/training')):
#        for directory in dirs:
#            class_names.add(directory)
#
#    # Get class names from validation data
#    for root, dirs, files in os.walk(os.path.join(source_dir, 'images/validation')):
#        for directory in dirs:
#            class_names.add(directory)
#
#    return class_names
#
## Example usage:
#source_dir = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26'
#class_names = get_class_names(source_dir)
#for class_name in sorted(class_names):
#    print(class_name)