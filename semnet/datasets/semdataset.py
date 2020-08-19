import os.path as osp
import numpy as np
from PIL import Image
import skimage.io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
HERE = osp.dirname(osp.abspath(__file__))
DATASETS_DIR = osp.join(HERE, '../../sem_data')

class SEMDataset(Dataset):
    class_names = np.array([
        'FA',
        'OPC'
    ])

    def __init__(self, root=None, split='train', augmentations=None):
        super(SEMDataset, self).__init__()
        self.root = root
        self.split = split
        self.augmentations = augmentations

        if self.root is None:
            self.root = DATASETS_DIR
        if not osp.exists(self.root):
            self.download()

        self.files = []
        imgsets_file = osp.join(self.root, 'train.txt')
        for did in open(imgsets_file):
            did = did.strip()
            type, idx = did.split(',')
            img_file = osp.join(self.root, 'SEM-%s-2020/%s-0-%03d.tif' % (type, type, int(idx)))
            al_file = osp.join(self.root, 'SEM-%s-2020/%s-Al-%03d.tif' % (type, type, int(idx)))
            ca_file = osp.join(self.root, 'SEM-%s-2020/%s-Ca-%03d.tif' % (type, type, int(idx)))
            si_file = osp.join(self.root, 'SEM-%s-2020/%s-Si-%03d.tif' % (type, type, int(idx)))
            self.files.append({
                'idx': idx,
                'type': type,
                'img_file': img_file,
                'al_file': al_file,
                'ca_file': ca_file,
                'si_file': si_file,
            })

    def __len__(self):
        if self.split == "train":
            return 155
        else:
            return 39

    def __getitem__(self, index):
        if self.split == "train":
            index = index
        else:
            index = index + 155
        data_file = self.files[index]
        img_file = data_file['img_file']
        img = Image.open(img_file).convert('L')
        img = np.array(img, dtype=np.uint8)[:512, 131:643]
        al_file = data_file['al_file']
        al = skimage.io.imread(al_file)[:256, 65:321, 0]
        ca_file = data_file['ca_file']
        ca = skimage.io.imread(ca_file)[:256, 65:321, 0]
        si_file = data_file['si_file']
        si = skimage.io.imread(si_file)[:256, 65:321, 1]
        al = resize(al, (512, 512))
        ca = resize(ca, (512, 512))
        si = resize(si, (512, 512))
        img = self.transform(img, al, ca, si)
        lbl = torch.tensor([0.0]) if data_file['type'] == 'FA' else torch.tensor([1.0])
        sample = {'idx': data_file['idx'], 'lbl': lbl , 'img': img}
        return sample

    def transform(self, img, al, ca, si):
        img = img.astype(np.float64)
        img = img / 255.0
        img = torch.from_numpy(img).float()
        al = torch.from_numpy(al).float()
        ca = torch.from_numpy(ca).float()
        si = torch.from_numpy(si).float()
        img = torch.stack((img, al, ca, si), dim=0)
        return img

    @staticmethod
    def download():
        raise NotImplementedError

if __name__ == '__main__':
    # dataset = SEMDataset()
    # sample = dataset[0]
    # print(sample['lbl'], sample['idx'])
    # import matplotlib.pyplot as plt
    # plt.subplot(221)
    # plt.imshow(sample['img'].numpy()[0,:,:])
    # plt.subplot(222)
    # plt.imshow(sample['img'].numpy()[1,:,:])
    # plt.subplot(223)
    # plt.imshow(sample['img'].numpy()[2,:,:])
    # plt.subplot(224)
    # plt.imshow(sample['img'].numpy()[3,:,:])
    # plt.show()
    batch_size = 3
    from semnet.models.resnet import SEMNet
    from torch.autograd import Variable
    model = SEMNet()
    data_loader = torch.utils.data.DataLoader(SEMDataset(split='val'), batch_size=batch_size, shuffle=False)
    batch_idx, sample = next(enumerate(data_loader))
    img, lbl = sample['img'], sample['lbl']
    # import matplotlib.pyplot as plt
    # plt.subplot(221)
    # plt.imshow(sample['img'].numpy()[0,0,:,:])
    # plt.subplot(222)
    # plt.imshow(sample['img'].numpy()[0,1,:,:])
    # plt.subplot(223)
    # plt.imshow(sample['img'].numpy()[0,2,:,:])
    # plt.subplot(224)
    # plt.imshow(sample['img'].numpy()[0,3,:,:])
    # plt.show()
    cuda = torch.cuda.is_available()
    if cuda:
        img, lbl = img.cuda(), lbl.cuda()
        model = model.cuda()
    img, lbl = Variable(img), Variable(lbl)
    with torch.no_grad():
        pred = model(img)
    lbl = lbl.data.cpu().squeeze(dim=1).numpy()
    pred = np.where(pred.data.cpu().squeeze(dim=1).numpy() > 0.5, 1.0, 0.0)
    print(lbl, pred)





