import torch

def weighted_difference_loss(x, y, n=3, reverse = True):
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
