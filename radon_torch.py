### Author: Hojune Kim
### Date: June 15, 2023
### Last Updated: Jun 15, 2023
### Description: Radon Transform (obj to sinogram)

import torch
import os
import time
import numpy as np

from torchvision.utils import save_image
from sinogram_class import ImageRotator, ImgToSinogram

def main():
    ###############################################################
    # Specify the file paths
    sinogram_path = os.path.join(os.pardir, 'dataset_foam', 'x_train_sinograms.npy')
    obj_path = os.path.join(os.pardir,  'dataset_foam' ,'foam_training.npy')

    # Load the .npy files
    sinogram_data = np.load(sinogram_path)
    obj_data = np.load(obj_path)

    # Convert the data to PyTorch tensors
    sinogram_tensor = torch.from_numpy(sinogram_data)
    obj_tensor = torch.from_numpy(obj_data)

    # Preprocess
    image_batch = obj_tensor.unsqueeze(1) # Size = (n,y,x)
    angles = np.arange(91, 271, dtype=float) # Angle list
    
    # Create instances of the classes
    rotata = ImageRotator()
    sinooo = ImgToSinogram()

    # Pass the image batch and angles through the model
    rotata.forward(image_batch, angles) 
    sinooo.forward(rotata.rotated_batch) 

    # Define Variables to save results
    rotated_images_batch = rotata.rotated_batch # torch.Size([50, 180, 128, 128])
    sinogram_batch = sinooo.sinogram_batch # torch.Size([50, 180, 128])
    
    # Create the data folder if it doesn't exist
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Save sinogram_batch as .npy file
    sinogram_file = os.path.join(data_folder, 'sinogram_batch.npy')
    np.save(sinogram_file, sinogram_batch)
    
    ###############################################################
    

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')