import torch
import torch.nn.functional as F
import random, json
import pystrum.pynd.ndutils as nd
from scipy.ndimage import generate_binary_structure, distance_transform_edt, binary_erosion
import scipy.ndimage as ndimage
import numpy as np

def read_txt_file(path):
    try:
        with open(path,'r',encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0,len(lines)):
                lines[i] = lines[i].strip('\n')
            return lines
    except FileExistsError:
        print("File not found. Please check the path.")
        return[]
    except Exception as e:
        print("An error occurred:".str(e))
        return []

def MotionVAELoss_weighted(recon_x, x, mu, logvar, beta=1e-2):
    BCE = mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), 1))
    result = BCE + beta * KLD
    return result

def MotionVAELoss_weighted_batch(recon_x, x, mu, logvar, beta=1e-2):
    result_list = []
    logvar = logvar.unsqueeze(1)
    mu = mu.unsqueeze(1)
    for i in range(len(recon_x)):
        BCE = mse_loss(recon_x[i], x[i])
        logvar = logvar[i]
        mu = mu[i]
        KLD = -0.5 * torch.mean(torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), 1))
        result = BCE + beta * KLD
        result_list.append(result)
    return torch.mean(torch.stack(result_list))

def mse_loss(input, target):
    return torch.mean(torch.sum((input - target) ** 2, (3, 2, 1)))

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def nonzero_centroid(arr):
    nonzero_indices = np.nonzero(arr)
    #centroid = [int(np.mean(nonzero_indices[i])) for i in range(len(nonzero_indices))]
    centroid = [np.mean(nonzero_indices[i]) for i in range(len(nonzero_indices))]
    return centroid


def TRE(y_ture,y_pred,spacing=0.8):
    coord1 = nonzero_centroid(y_ture)
    coord2 = nonzero_centroid(y_pred)
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    try:
        distance = np.linalg.norm(coord1-coord2)  #欧式
        #distance = np.mean(np.abs(np.subtract(coord1, coord2)))  # 平均绝对误差MAE
    except IndexError:  #处理标签缺失等异常
        distance = 100000.0
    return distance *spacing

def DSC( prediction, target):
    smooth = 1e-6  # Smoothing factor to avoid division by zero
    prediction = prediction.squeeze(0)  # Remove batch dimension
    target = target.squeeze(0)  # Remove batch dimension
    try:
        # Compute intersection and union
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)

        # Compute Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
    except IndexError:  #处理标签缺失等异常
        dice = 0.0
    return dice

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)  #输入【C，D，H, W】--- [D,H,W,C]
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def Dice_loss( prediction, target):
    smooth = 1e-6  # Smoothing factor to avoid division by zero
    prediction = prediction.squeeze(0)  # Remove batch dimension
    target = target.squeeze(0)  # Remove batch dimension

    # Compute intersection and union
    intersection = torch.sum(prediction * target)
    union = torch.sum(prediction) + torch.sum(target)

    # Compute Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Compute Dice loss
    loss = 1.0 - dice

    return loss









