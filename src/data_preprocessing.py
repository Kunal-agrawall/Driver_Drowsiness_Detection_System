import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def organize_eye_data(source_path, destination_path):
    # Create directory structure
    os.makedirs(os.path.join(destination_path, 'train', 'open'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'train', 'closed'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'validation', 'open'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'validation', 'closed'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'test', 'open'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'test', 'closed'), exist_ok=True)
    
    # Get list of files
    open_files = [f for f in os.listdir(os.path.join(source_path, 'Open')) if f.endswith(('.png', '.jpg', '.jpeg'))]
    closed_files = [f for f in os.listdir(os.path.join(source_path, 'Closed')) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split data
    open_train, open_temp = train_test_split(open_files, test_size=0.3, random_state=42)
    open_val, open_test = train_test_split(open_temp, test_size=0.5, random_state=42)
    
    closed_train, closed_temp = train_test_split(closed_files, test_size=0.3, random_state=42)
    closed_val, closed_test = train_test_split(closed_temp, test_size=0.5, random_state=42)
    
    # Copy files to destination
    for file in open_train:
        shutil.copy(os.path.join(source_path, 'Open', file), os.path.join(destination_path, 'train', 'open', file))
    for file in open_val:
        shutil.copy(os.path.join(source_path, 'Open', file), os.path.join(destination_path, 'validation', 'open', file))
    for file in open_test:
        shutil.copy(os.path.join(source_path, 'Open', file), os.path.join(destination_path, 'test', 'open', file))
    
    for file in closed_train:
        shutil.copy(os.path.join(source_path, 'Closed', file), os.path.join(destination_path, 'train', 'closed', file))
    for file in closed_val:
        shutil.copy(os.path.join(source_path, 'Closed', file), os.path.join(destination_path, 'validation', 'closed', file))
    for file in closed_test:
        shutil.copy(os.path.join(source_path, 'Closed', file), os.path.join(destination_path, 'test', 'closed', file))

def organize_yawn_data(source_path, destination_path):
    # Create directory structure
    os.makedirs(os.path.join(destination_path, 'yawn'), exist_ok=True)
    os.makedirs(os.path.join(destination_path, 'no_yawn'), exist_ok=True)
    
    # Get list of files
    yawn_files = [f for f in os.listdir(os.path.join(source_path, 'yawn')) if f.endswith(('.png', '.jpg', '.jpeg'))]
    no_yawn_files = [f for f in os.listdir(os.path.join(source_path, 'no_yawn')) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split data
    yawn_train, yawn_test = train_test_split(yawn_files, test_size=0.2, random_state=42)
    yawn_train, yawn_val = train_test_split(yawn_train, test_size=0.1, random_state=42)
    
    no_yawn_train, no_yawn_test = train_test_split(no_yawn_files, test_size=0.2, random_state=42)
    no_yawn_train, no_yawn_val = train_test_split(no_yawn_train, test_size=0.1, random_state=42)
    
    # Copy files to destination
    for file in yawn_train:
        shutil.copy(os.path.join(source_path, 'yawn', file), os.path.join(destination_path, 'yawn', file))
    for file in yawn_val:
        shutil.copy(os.path.join(source_path, 'yawn', file), os.path.join(destination_path, 'yawn', file))
    for file in yawn_test:
        shutil.copy(os.path.join(source_path, 'yawn', file), os.path.join(destination_path, 'yawn', file))
    
    for file in no_yawn_train:
        shutil.copy(os.path.join(source_path, 'no_yawn', file), os.path.join(destination_path, 'no_yawn', file))
    for file in no_yawn_val:
        shutil.copy(os.path.join(source_path, 'no_yawn', file), os.path.join(destination_path, 'no_yawn', file))
    for file in no_yawn_test:
        shutil.copy(os.path.join(source_path, 'no_yawn', file), os.path.join(destination_path, 'no_yawn', file))
