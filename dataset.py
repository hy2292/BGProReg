import torch.utils.data as data
import os
from utils import *
import SimpleITK as sitk

class Dataset_prostate_DDF_for_biopsy(data.Dataset):
    def __init__(self, root, split, img_size=128):
        super(Dataset_prostate_DDF_for_biopsy, self).__init__()
        self.img_size = img_size
        self.root = root

        if split == 'train':
            self.file_list = read_txt_file(os.path.join(root, 'train.txt'))
            print(self.file_list)
        elif split == 'val':
            self.file_list = read_txt_file(os.path.join(root, 'val.txt'))
            print(self.file_list)
        elif split == 'test':
            self.file_list = read_txt_file(os.path.join(root, 'test.txt'))
            print(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        ddf_path = os.path.join(self.root, 'DDFs_npy', f'ID{filename[:5]}_DDF_FEM.npy')
        ddf = np.load(ddf_path)
        ddf = np.transpose(ddf,(3,0,1,2))
        return ddf


class Dataset_prostate_for_biopsy_predict(data.Dataset):
    def __init__(self, root, img_size=128):
        super(Dataset_prostate_for_biopsy_predict, self).__init__()
        self.img_size = img_size
        self.root = root
        self.file_list = read_txt_file(os.path.join(root, 'test.txt'))
        print(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]

        # load prostate mask
        mr_msk_path = os.path.join(self.root, 'processed', f'ID{filename[:5]}_MR_msk.nrrd')
        mr_msk = sitk.ReadImage(mr_msk_path)
        mr_msk = sitk.GetArrayFromImage(mr_msk).astype(np.uint8)
        us_msk_path = os.path.join(self.root, 'processed', f'ID{filename[:5]}_US_msk.nrrd')
        us_msk = sitk.ReadImage(us_msk_path)
        us_msk = sitk.GetArrayFromImage(us_msk).astype(np.uint8)

        mr_label_path = os.path.join(self.root, 'processed', f'ID{filename[:5]}_MR_target.nrrd')
        us_label_path = os.path.join(self.root, 'processed', f'ID{filename[:5]}_US_target.nrrd')

        mr_label = sitk.ReadImage(mr_label_path)
        mr_label = sitk.GetArrayFromImage(mr_label).astype(np.uint8)
        us_label = sitk.ReadImage(us_label_path)
        us_label = sitk.GetArrayFromImage(us_label).astype(np.uint8)

        #load img
        mr_img_path = os.path.join(self.root, 'processed', f'ID{filename[:5]}_MR_img.nrrd')
        mr_img = sitk.ReadImage(mr_img_path)
        mr_img = sitk.GetArrayFromImage(mr_img).astype(np.float32)
        us_img_path = os.path.join(self.root, 'processed', f'ID{filename[:5]}_US_img.nrrd')
        us_img = sitk.ReadImage(us_img_path)
        us_img = sitk.GetArrayFromImage(us_img).astype(np.float32)

        return us_msk, mr_msk , us_img, mr_img, us_label, mr_label








