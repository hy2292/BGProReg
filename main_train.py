import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import *
from network.VAE_new import *
from torch.utils.data.dataloader import DataLoader
import time

lr = 1e-5
n_worker = 0
bs = 4
n_epoch = 200
beta = 0.1
alpha = 0
weight_decay = 0
max_norm = 6

set_seeds(42)
model_save_path = './models/1e-4_bn_4_beta_0.1_zdim_64.pth'
VAE_model = VAE_new(img_size=128, z_dim=64, nf=32).cuda()

writer = SummaryWriter('log/1e-4_bn_4_beta_0.1_zdim_64')

# model_load_path = './models/main_prior_model_pretrained.pth'
# VAE_model.load_state_dict(torch.load(model_load_path))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, VAE_model.parameters()), lr=lr, weight_decay=weight_decay)


def train(epoch):
    VAE_model.train()
    epoch_loss = []

    for batch_id, flow in tqdm(enumerate(training_data_loader,1), total=len(training_data_loader)):

        flow = flow.cuda()
        optimizer.zero_grad()
        recon, mu, logvar = VAE_model(flow,max_norm)

        df_loss = MotionVAELoss_weighted(recon, flow,  mu, logvar, beta)
        loss = df_loss

        loss.backward()

        optimizer.step()
        epoch_loss.append(loss.item())

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,  np.mean(epoch_loss)))
    writer.add_scalar('train_loss', np.mean(epoch_loss), epoch)


def test():
    VAE_model.eval()
    test_loss = []
    global base_err

    for batch_idx, flow in tqdm(enumerate(testing_data_loader,1),total=len(testing_data_loader)):

        flow = flow.cuda()
        recon, mu, logvar = VAE_model(flow, max_norm)

        df_loss = MotionVAELoss_weighted(recon, flow, mu, logvar, beta)
        loss = df_loss

        test_loss.append(loss.item())

    print('Base Loss: {:.6f}'.format(base_err))
    print('Test Loss: {:.6f}'.format(np.mean(test_loss)))
    writer.add_scalar('test_loss', np.mean(test_loss), epoch)

    if np.mean(test_loss) < base_err:
        torch.save(VAE_model.state_dict(), model_save_path)
        print("Checkpoint saved to {}".format(model_save_path))
        base_err = np.mean(test_loss)

root = '/data/new-MRI-US'
train_set = Dataset_prostate_DDF_for_biopsy(root, 'train')
val_set = Dataset_prostate_DDF_for_biopsy(root, 'val')

# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker,batch_size=bs, shuffle=True)
testing_data_loader = DataLoader(dataset=val_set, num_workers=n_worker,batch_size=bs, shuffle=False)
base_err = 1000000

for epoch in range(0, n_epoch + 1):
    print('Epoch {}'.format(epoch))
    start = time.time()
    test()
    end = time.time()
    print("testing took {:.8f}s".format(end - start))

    start = time.time()
    train(epoch)
    end = time.time()
    print("training took {:.8f}s".format(end - start))