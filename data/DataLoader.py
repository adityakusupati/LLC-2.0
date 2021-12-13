import torch
import os
import numpy as np
from PIL import Image

class InstanceLoader:
    def __init__(self, data_root, label_map_path, flat_struct = False):
        self.image_paths = []
        self.label_map = np.load(label_map_path, allow_pickle = True).item()
        if flat_struct == False:
            for folder in os.listdir(data_root):
                folder_path = os.path.join(data_root, folder)
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    self.image_paths.append(file_path)
        else:
            for folder in os.listdir(data_root):
                    folder_path = os.path.join(data_root, folder)
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        self.image_paths.append(file_path)
            
    def len(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path)
        label = self.label_map[image_name]
        image = Image.open(image_path).convert('RGB')
        return image, label



def create_label_map(data_root, save_path,flat_struct = False):
    #Creates hash map from data set at data_root and saves to save_path.
    label_map = dict()
    i = 0
    if flat_struct == False:
        for folder in os.listdir(data_root):
            folder_path = os.path.join(data_root, folder)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                label_map[file_name] = i
                i += 1
    else:
        for file_name in os.listdir(data_root):
            file_path = os.path.join(data_root, file_name)
            label_map[file_name] = i
            i += 1
    print("Saving label map to: " + save_path)
    np.save(save_path, label_map)

