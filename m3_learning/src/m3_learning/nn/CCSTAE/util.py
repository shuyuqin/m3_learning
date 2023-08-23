import numpy as np
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from .nnModel import center_of_mass


def mask_function(img,radius=7,center_coordinates=(100,100)):
    image = np.copy(img.squeeze())
    thickness = -1
    color = 100
    image_2 = cv2.circle(image, center_coordinates, radius,color, thickness)
    image_2 = np.array(image_2)
    mask = (image_2==100)
    mask = np.array(mask)

    return mask


def make_folder(folder, **kwargs):
    """Function that makes new folders

    :param folder: folder where to save
    :type folder: string
    :return: folder where to save
    :rtype: string
    """


    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return (folder)


def Show_Process(model,
                 test_iterator,
                 mask_list,
                 name_of_file,
                 device,
                 up_inp,
                 ):

    model.eval()
    
    number_loss = len(mask_list)

    for i, x_value in enumerate(tqdm(test_iterator,leave=True,total=len(test_iterator))):
        with torch.no_grad():
            
            if type(x_value) != list:
                
                x = x_value.to(self.device, dtype=torch.float)
                y= None
            else:
                x,y = x_value
                x = x.to(self.device, dtype= torch.float)
                y = y.to(self.device, dtype= torch.float)
            
            if up_inp:
                predicted_x,predicted_base,predicted_input,kout,theta_1,theta_2,adj_mask,new_list,x_inp= model(x,y)
                
                mask_list = upsample_mask(mask_list,test_value.shape[-1], x_inp.shape[-1])
                
            else:
                predicted_x,predicted_base,predicted_input,kout,theta_1,theta_2,adj_mask,new_list= model(x,y)
          
          
        if i ==0:
            break
            
    predicted_base = predicted_base[0].cpu().detach().numpy()
    
    h5f = h5py.File(name_of_file+'.h5', 'w')

    h5f.create_dataset('base', data=predicted_base)
    
    # stack the list to torch tensor for saving in the h5 format
    gou_list  = torch.cat(mask_list)
    gou_list = gou_list.reshape(len(mask_list),mask_list[0].shape[-2],mask_list[0].shape[-1])
    
    h5f.create_dataset('mask_list', data=gou_list)
    h5f.close()


def inverse_base(name_of_file, input_mask_list, coef=2, radius = 7):
    
    load_file = h5py.File(name_of_file+'.h5','r')
    load_base = load_file['base'][0].squeeze()
    
    
    
    base_ = torch.tensor(load_base,dtype=torch.float).reshape(1,1,load_base.shape[-1],load_base.shape[-2])
    
    center_mask_list,rotate_center = center_mask_list_function(base_,input_mask_list,coef,radius=radius)
    
    
    
    return center_mask_list, rotate_center


def upsample_mask(mask_list,input_size, up_size):
    
    if mask_list[0].shape[-1]==up_size:
        return mask_list
    
    mask_with_inp = []
    for mask_ in mask_list:
        temp_mask = torch.tensor(mask_.reshape(1,1,input_size,input_size),dtype=torch.float)
        temp_mask = F.interpolate(temp_mask, size=(up_size,up_size),mode = 'bicubic')
        temp_mask[temp_mask<0.5]=0
        temp_mask[temp_mask>=0.5]=1
        temp_mask = torch.tensor(temp_mask.squeeze(),dtype=torch.bool)
        mask_with_inp.append(temp_mask)

    return mask_with_inp


def center_mask_list_function(image,mask_list,coef,radius=7):
    
    center_mask_list = []
    mean_ = np.zeros([image.shape[-2],image.shape[-1]])
    
    input_size = mask_list[0].shape[-1]
    up_size = image.shape[-1]
    
    if input_size!=up_size:
        
        mask_list = upsample_mask(mask_list,input_size,up_size)
    
    for j, mask in enumerate(mask_list):
        

        mask_ = mask.reshape(1,1,mask.shape[-2],mask.shape[-1])

        new_image = image*mask_

        center_x,center_y = center_of_mass(new_image.squeeze(),mask_.squeeze(),coef)
        
        center_x = int(np.round(np.array(center_x)))
        center_y = int(np.round(np.array(center_y)))
        print(center_x,center_y)
        
        small_mask = mask_function(mean_,radius=radius,center_coordinates=(center_y,center_x))
        
        small_mask = torch.tensor(small_mask,dtype = torch.bool)
        
        center_mask_list.append(small_mask)
        
    if input_size!=up_size:
        
        center_mask_list = upsample_mask(center_mask_list,up_size,input_size)
        
    rotate_mask_up = torch.clone(center_mask_list[0])
    
    for i in range(1,len(center_mask_list)):
        rotate_mask_up+=center_mask_list[i]
        
    return center_mask_list,rotate_mask_up