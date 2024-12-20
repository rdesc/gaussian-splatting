import os
import shutil
import cv2
import yaml
import numpy as np


# Target directory to save the selected images
save_dir = 'duckiebot_images_HD'
os.makedirs(save_dir, exist_ok=True)

# Source directories and frame intervals
ego_view = ['/Users/rodriguedeschaetzen/Downloads/1080p/pngs', '/Users/rodriguedeschaetzen/Downloads/1080p/ccw_run']
ego_keep_every = 1  # Keep every 5th frame
overhead_view = []
overhead_keep_every = 2  # Keep every 5th frame
camera_intrinsic_filename = 'camera_intrinsic.yaml'

def load_camera_intrinsics(filename):
    if not os.path.isfile(filename):
        print(('Intrinsic calibration for {} does not exist.'.format(filename)))
        exit(3)
    with open(filename) as f:
        contents = f.read()
        data = yaml.safe_load(contents)
    intrinsics = {}
    intrinsics['K'] = np.array(data['camera_matrix']['data']).reshape(3, 3)
    intrinsics['D'] = np.array(data['distortion_coefficients']['data']).reshape(1, 5)
    intrinsics['R'] = np.array(data['rectification_matrix']['data']).reshape(3, 3)
    intrinsics['P'] = np.array(data['projection_matrix']['data']).reshape((3, 4))
    intrinsics['distortion_model'] = data['distortion_model']
    print('Loaded camera intrinsics for {}'.format(filename))
    return intrinsics

def rectify_images(image, intrinsics):
    '''Undistort image'''
    height, width, _ = image.shape
    rectified_image = np.zeros(np.shape(image))
    mapx = np.ndarray(shape=(height, width, 1), dtype='float32')
    mapy = np.ndarray(shape=(height, width, 1), dtype='float32')
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics['K'],
                                             intrinsics['D'],
                                             intrinsics['R'],
                                             intrinsics['P'],
                                             (width, height), cv2.CV_32FC1, mapx, mapy)
    return cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC, rectified_image)

# Function to copy selected images
def copy_images(source_dirs, keep_every, destination, prefix_file_name='', rectify=False, intrinsics=None):
    if rectify:
        intrinsics = load_camera_intrinsics(filename=intrinsics)

    count = 0
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory '{source_dir}' does not exist.")
            continue

        # List files in the directory
        files = sorted(os.listdir(source_dir))  # Sort for consistency
        for i, file_name in enumerate(files):
            if '.ini' in file_name: continue
            count += 1
            # Keep every nth image
            if count % keep_every == 0:
                source_path = os.path.join(source_dir, file_name)
                if os.path.isfile(source_path):  # Ensure it's a file
                    destination_path = os.path.join(destination, prefix_file_name + '_' + str(count) + '.png')

                    if rectify:
                        image = cv2.imread(source_path)
                        if image is not None:
                            rectified_image = rectify_images(image, intrinsics)
                            cv2.imwrite(destination_path, rectified_image)
                            print(f"Rectified and saved: {source_path} -> {destination_path}")
                        else:
                            print(f"Failed to load image: {source_path}")

                    else:
                        shutil.copy(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")

# Copy images from both sets of directories
copy_images(ego_view, ego_keep_every, save_dir, prefix_file_name='egoview') # , rectify=True, intrinsics='camera_intrinsic.yaml')
# copy_images(overhead_view, overhead_keep_every, save_dir, prefix_file_name='overhead') # , rectify=True, intrinsics='camera_intrinsic.yaml')
