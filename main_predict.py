import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import *
from network.VAE_new import *
from network.STN import *
from utils import *
from torch.utils.data.dataloader import DataLoader

n_worker = 1
bs = 1
z_dim = 64
img_size = [128,128,128]
spacing= [0.8,0.8,0.8]
mu = 1e-3
max_norm = 6

set_seeds(42)
VAE_model_load_path = './models/1e-4_bn_4_beta_0.1_zdim_64.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = '/home/dasong/ProstateReg-main/biopsy_prostate'
test_set = Dataset_prostate_for_biopsy_predict(root)
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,batch_size=bs, shuffle=False)

def test():
    dsc_all = []
    tre_all = []
    jd_all = []
    before_dsc_all=[]
    before_tre_all = []

    Trans_model = VAE_new(img_size=128, z_dim=z_dim, nf=32).to(device)
    Trans_model.load_state_dict(torch.load(VAE_model_load_path))

    spatial_trans = SpatialTransformer(img_size, mode='bilinear').to(device)
    spatial_trans_label = SpatialTransformer(img_size, mode='nearest').to(device)

    for batch_idx, batch in tqdm(enumerate(testing_data_loader,1), total=len(testing_data_loader)):
        us_msk, mr_msk, us_img, mr_img, us_label, mr_label= batch

        us_msk = us_msk[0].to(torch.float32, device=device)
        us_label = us_label[0].to(torch.float32, device=device)
        ori_mr_label = mr_label[0].to(torch.float32, device=device)
        mr_msk = mr_msk.to(torch.float32, device=device).unsqueeze(1)
        mr_label = mr_label.to(torch.float32, device=device).unsqueeze(1)

        z_recall = torch.tensor(np.random.normal(0.0, 0.8, (1, z_dim)), dtype=torch.float, requires_grad=True,device=device)
        optimizer = optim.Adam([{'params': z_recall}], lr=0.1, weight_decay=mu)

        delta_loss = 1
        loss_ex = 100
        count = 0

        while delta_loss > 1e-4:
            z_recall.retain_grad()
            optimizer.zero_grad()
            predicted_ddf = Trans_model.decode(z_recall)
            predicted_ddf = predicted_ddf * max_norm

            after_mr_msk = spatial_trans_label(mr_msk, predicted_ddf)
            loss = Dice_loss(after_mr_msk, us_msk)

            dsc = 1-loss.cpu().detach().numpy()
            #print("Dice:", dsc)
            loss.backward(retain_graph=True)
            optimizer.step()
            delta_loss = np.abs(loss_ex - loss.item())
            loss_ex = loss.item()
            count += 1

        print("Predict finished!")
        print("Dice:", dsc)

        after_mr_label = spatial_trans_label(mr_label, predicted_ddf)
        after_mr_label = torch.squeeze(after_mr_label)
        after_mr_msk = spatial_trans(mr_msk, predicted_ddf)

        dsc = DSC(after_mr_msk, us_msk).cpu().detach()
        before_dsc = DSC(mr_msk, us_msk).cpu().detach()
        print("Before Dice: %.4f" % before_dsc)
        print("Dice: %.4f" % dsc)
        dsc_all.append(dsc)
        before_dsc_all.append(before_dsc)

        tre = TRE(after_mr_label.cpu().detach().numpy(), us_label.cpu().detach().numpy())
        before_tre = TRE(ori_mr_label.cpu().detach().numpy(), us_label.cpu().detach().numpy())
        print(f'Before TRE : %.4f' % before_tre)
        print(f'TRE : %.4f' % tre)
        tre_all.append(tre)
        before_tre_all.append(before_tre)

        jd = jacobian_determinant_vxm(predicted_ddf.squeeze().cpu().detach().numpy())
        jd_result = np.sum(jd <= 0) / np.prod(us_msk.shape)
        print(f'JD: %.5f' %jd_result)
        jd_all.append(jd_result)

    print("Before Dice_mean: {:.4f} and std is {:.4f}" .format(np.mean(before_dsc_all), np.std(before_dsc_all)) )
    print("Before TRE_mean: {:.4f} and std is {:.4f}".format(np.mean(before_tre_all), np.std(before_tre_all)))
    print("Dice_mean: {:.4f} and std is {:.4f}".format(np.mean(dsc_all), np.std(dsc_all)))
    print("TRE_mean: {:.4f} and std is {:.4f}".format(np.mean(tre_all), np.std(tre_all)))
    print("JD_mean: %.4f " % np.mean(jd_all))

if __name__ == '__main__':
    test()
