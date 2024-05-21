import os
import shutil

# Define the source and target directories
source_dir = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26/images'
target_dir = '/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted'

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_category_mapping(src):
    # Get all first-level subdirectories in the training set as categories
    category_mapping = {}
    training_dir = os.path.join(src, 'training')
    for letter_dir in os.listdir(training_dir):
        letter_path = os.path.join(training_dir, letter_dir)
        if os.path.isdir(letter_path):
            for category in os.listdir(letter_path):
                category_path = os.path.join(letter_path, category)
                if os.path.isdir(category_path):
                    category_mapping[category] = category
    return category_mapping

def copy_files(src, dst, category_mapping):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith('.png'):
                category = os.path.basename(root)
                if category in category_mapping:
                    target_category = category_mapping[category]
                    target_path = os.path.join(dst, target_category)
                    ensure_dir_exists(target_path)
                    shutil.move(os.path.join(root, file), os.path.join(target_path, file))

def main():
    # Directories to process
    dirs_to_process = {
        'training': 'train',
        'validation': 'test'
    }

    category_mapping = get_category_mapping(source_dir)

    for src_subdir, dst_subdir in dirs_to_process.items():
        src_path = os.path.join(source_dir, src_subdir)
        dst_path = os.path.join(target_dir, dst_subdir)
        ensure_dir_exists(dst_path)
        copy_files(src_path, dst_path, category_mapping)

if __name__ == '__main__':
    main()
