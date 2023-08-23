import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable



# TODO build a class for the accumulated loss, this should take loss functions as an input
class AcumulatedLoss(nn.Module):

    def __init__(self, 
                 device = torch.device('cpu'),
                 reg_coef = 0,
                 scale_coef = 0,
                 shear_coef = 0,
                 norm_order = 1,
                 scale_penalty=0.04,
                 shear_penalty=0.03,
                 mask_list = None,
                 batch_para = 2,
                 loss_type = 'custom',
                 weight_coef = 2,
                 upgrid_img = False,
                 soft_threshold = 1.5,
                 hard_threshold = 3,
                 con_div = 15
                 ):
        super(AcumulatedLoss, self).__init__()

        self.device = device
        self.reg_coef = reg_coef
        self.scale_coef = scale_coef
        self.shear_coef = shear_coef
        self.norm_order = norm_order
        self.scale_penalty = scale_penalty
        self.shear_penalty = shear_penalty
        self.mask_list = mask_list
        self.loss_type = loss_type
        self.weight_coef = weight_coef
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.con_div = con_div
#        self.loss_function = loss_function
        
    def __call__(self,
                 model,
                 data_iterator,
                 optimizer,
                 batch_para = 128,
                 cycle_consistent = True,
                 upgrid_img = False,
                 dynamic_mask_region = False
                 ):

        self.cycle_consistent = cycle_consistent
        self.upgrid_img = upgrid_img
        self.batch_para = batch_para
        self.dynamic_mask_region = dynamic_mask_region
        
        train_loss = 0
        L2_loss = 0 
        Scale_Loss = 0
        Shear_Loss = 0 

        model.train()

        optimizer.zero_grad()
    
        NUM_ACCUMULATION_STEPS = self.batch_para
        # NUM_ACCUMULATION_STEPS = div # Full gradient

        #for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
        for batch_idx, x_value in enumerate(tqdm(data_iterator)):
            
            if type(x_value) != list:
                x = x_value.to(self.device, dtype=torch.float)
                y= None
            else:
                x,y = x_value
                x = x.to(self.device, dtype= torch.float)
                y = y.to(self.device, dtype= torch.float)
            
            if self.upgrid_img:

                predicted_x,predicted_base,predicted_input,kout,theta_1,theta_2,adj_mask,new_list,x_inp = model(x,y)
            
            else:
            
                predicted_x,predicted_base,predicted_input,kout,theta_1,theta_2,adj_mask,new_list = model(x,y)
            
            l2_loss = self.reg_coef*torch.norm(predicted_base.squeeze(), p=self.norm_order)/x.shape[0]
            
            scale_loss = self.scale_coef*(torch.mean(F.relu(abs(theta_1[:,0,0]-1)-self.scale_penalty))\
                                    +torch.mean(F.relu(abs(theta_1[:,1,1]-1)-self.scale_penalty)))
            
            shear_loss = self.shear_coef*torch.mean(F.relu(abs(theta_1[:,0,1])-self.shear_penalty)) 
            
            initial_loss = l2_loss + scale_loss + shear_loss

            if self.dynamic_mask_region:

                
                loss = self.dynamic_mask_list(x_inp,predicted_x,predicted_base,predicted_input,
                                              new_list,initial_loss,con_div=self.con_div)

            else:
                loss = self.fix_mask_list(x,predicted_x,predicted_base,predicted_input,initial_loss)

            
            
            # TODO: these need to be changed to no be hard coded, can be default but should be able to change it.         
            
            # TODO: Dictionary of the loss functions
            # Update vars
            train_loss += loss.detach().cpu().numpy()
            L2_loss += l2_loss.detach().cpu().numpy()
            Scale_Loss += scale_loss.detach().cpu().numpy()
            Shear_Loss += shear_loss.detach().cpu().numpy()

            # Backward pass
            loss = loss / NUM_ACCUMULATION_STEPS
            loss.backward()

            # Update the weights
            if ((batch_idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(data_iterator)):
                optimizer.step()
                optimizer.zero_grad()  

        train_loss = train_loss/len(data_iterator)
        L2_loss =  L2_loss/len(data_iterator)
        Scale_Loss =  Scale_Loss/len(data_iterator)
        Shear_Loss =  Shear_Loss/len(data_iterator)

        loss_dictionary = { "train_loss": train_loss,
                       "l2_loss": L2_loss,
                       "scale_loss": Scale_Loss,
                       "shear_loss": Shear_Loss
                       }

        return loss_dictionary
    
    def weighted_difference_loss(self,
                                 x, 
                                 y, 
                                 n=3, 
                                 reverse = True
                                 ):
        """Adds a weight to the MSE loss based in the difference is positive or negative.

        Args:
            x (torch.tensor): _description_
            y (_type_): _description_
            n (int, optional): _description_. Defaults to 3.
            reverse (bool, optional): _description_. Defaults to True.
        """
    
        # switches the order of the arguments when calculating the difference
        if reverse:
            diff = x-y
        else:
            diff = y-x
        
        # determines the positions to weight
        index_pos = torch.where(diff>0)
        index_neg = torch.where(diff<0)
        
        value = (torch.sum(diff[index_pos]**2) + \
                n*torch.sum(diff[index_neg]**2))/torch.numel(x)
            
        return value

    def fix_mask_list(self,
                      x,
                      predicted_x,
                      predicted_base,
                      predicted_input,
                      initial_loss
                      ):
        """_summary_

        Args:
            x (_type_): _description_
            predicted_x (_type_): _description_
            predicted_base (_type_): _description_
            predicted_input (_type_): _description_
            initial_loss (_type_): _description_

        Returns:
            _type_: _description_
        """

        loss = initial_loss + 0

        for i,mask in enumerate(self.mask_list):
                
            if self.loss_type == 'custom':

                loss += self.weighted_difference_loss(predicted_x.squeeze()[:,mask],predicted_base.squeeze()[:,mask],n = self.weight_coef)\
                    + self.weighted_difference_loss(x.squeeze()[:,mask], predicted_input.squeeze()[:,mask],n = self.weight_coef)
            else:
                
                loss += F.mse_loss(predicted_base.squeeze()[:,mask], predicted_x.squeeze()[:,mask], reduction='mean')\
                + F.mse_loss(predicted_input.squeeze()[:,mask], x.squeeze()[:,mask], reduction='mean')
            
        if loss > self.soft_threshold:

            loss = initial_loss + 0 

            for i,mask in enumerate(self.mask_list):

                loss += F.l1_loss(predicted_base.squeeze()[:,mask], predicted_x.squeeze()[:,mask], reduction='mean')\
                + F.l1_loss(predicted_input.squeeze()[:,mask], x.squeeze()[:,mask], reduction='mean')

            loss-=1

        if loss > self.hard_threshold:

            loss = initial_loss + self.hard_threshold



        return loss
    
    def dynamic_mask_list(self,
                          x,
                          predicted_x,
                          predicted_base,
                          predicted_input,
                          new_list,
                          initial_loss,
                          con_div=15):
        """_summary_

        Args:
            x (_type_): _description_
            predicted_x (_type_): _description_
            predicted_base (_type_): _description_
            predicted_input (_type_): _description_
            new_list (_type_): _description_
            initial_loss (_type_): _description_
            con_div (int, optional): _description_. Defaults to 15.

        Returns:
            _type_: _description_
        """

        loss = len(self.mask_list)*initial_loss 

        for i,mask in enumerate(self.mask_list):

            if self.cycle_consistent:

                loss += F.mse_loss(predicted_base.squeeze()[:,mask], predicted_x.squeeze()[:,mask], reduction='mean')
            #set the loss for the generated input and input
            sub_loss = 0
            for k in range(x.shape[0]):
                
                sub_loss += F.mse_loss(predicted_input[k].squeeze()[new_list[i][k]], x[k].squeeze()[new_list[i][k]], reduction='mean')


            loss += (sub_loss/x.shape[0])

        loss = loss/(len(self.mask_list)*con_div)

        if loss > self.soft_threshold:

            loss = len(self.mask_list)*initial_loss 
            
            for i, mask in enumerate(self.mask_list):

                if self.cycle_consistent:

                    loss += F.l1_loss(predicted_base.squeeze()[:,mask], predicted_x.squeeze()[:,mask], reduction='mean')

                #set the loss for the generated input and input
                sub_loss = 0

                for k in range(x.shape[0]):
                    
                    sub_loss += F.l1_loss(predicted_input[k].squeeze()[new_list[i][k]], x[k].squeeze()[new_list[i][k]], reduction='mean')

                loss += (sub_loss/x.shape[0])

            loss = loss/(len(self.mask_list)*con_div)

            loss-=1
        
        if loss > self.hard_threshold:

            loss = initial_loss + self.hard_threshold

        return loss