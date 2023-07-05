import torch
from PIL import Image
import os
import torchvision
import math
import natsort


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               [".png", ".jpg", ".jpeg", ".JPEG", ".tif", ".bmp"])


class RC_Dataset(torch.utils.data.Dataset):
    """
    Dataset for Random Coupling
    """

    def __init__(self, root_dir="data",
                 filespath="/ILSVRC2012_img_val_noisy", multiplicity=8,
                 transform=torchvision.transforms.ToTensor()):
        self.filespath = filespath
        self.root_dir = root_dir
        self.transform = transform
        self.multiplicity = multiplicity
        names = self.__dict__

        for i in range(1, self.multiplicity + 1):
            names['fullpath' + str(i)] = os.path.join(self.root_dir,
                                                      self.filespath) + str(i)
            names['files' + str(i)] = natsort.natsorted(
                [self.filespath + str(i) + '/' + x for x in
                 os.listdir(eval('self.fullpath' + str(i))) if
                 is_image_file(x)])

    def __getitem__(self, idx):
        src = math.ceil(torch.rand(1)[0] * self.multiplicity)
        tar = math.ceil(torch.rand(1)[0] * self.multiplicity)
        if src == 0:
            src = 1
        if tar == 0:
            tar = 1

        srcImgpath = self.root_dir + eval(
            'self.files' + str(src) + '[' + str(idx) + ']')
        tarImgpath = self.root_dir + eval(
            'self.files' + str(tar) + '[' + str(idx) + ']')

        srcImg = self.transform(Image.open(srcImgpath).convert('RGB'))
        tarImg = self.transform(Image.open(tarImgpath).convert('RGB'))

        srcImg = srcImg[:, 0:256, 0:256]
        tarImg = tarImg[:, 0:256, 0:256]
        if srcImg.shape[0] == 1:
            srcImg = torch.cat([srcImg, srcImg, srcImg], 0)
        if tarImg.shape[0] == 1:
            tarImg = torch.cat([tarImg, tarImg, tarImg], 0)

        return [srcImg, tarImg]

    def __len__(self):
        i = math.ceil(torch.rand(1)[0] * self.multiplicity)
        j = math.ceil(torch.rand(1)[0] * self.multiplicity)
        exec("assert len(self.files{}) == len(self.files{})".format(i, j))
        return len(eval('self.files' + str(i)))

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class AL_Dataset(torch.utils.data.Dataset):
    """
    Dataset for Alienation Loss
    """

    def __init__(self, root_dir="data",
                 filespath="/ILSVRC2012_img_val_noisy", multiplicity=8,
                 transform=torchvision.transforms.ToTensor()):
        self.filespath = filespath
        self.root_dir = root_dir
        self.transform = transform
        self.multiplicity = multiplicity
        names = self.__dict__

        for i in range(1, self.multiplicity + 1):
            names['fullpath' + str(i)] = os.path.join(self.root_dir,
                                                      self.filespath) + str(i)
            names['files' + str(i)] = natsort.natsorted(
                [self.filespath + str(i) + '/' + x for x in
                 os.listdir(eval('self.fullpath' + str(i))) if
                 is_image_file(x)])

    def __getitem__(self, idx):
        for i in range(1, self.multiplicity + 1):
            self.names['srcImgpath' + str(i)] = self.root_dir + eval(
                'self.files' + str(i) + '[' + str(idx) + ']')
            self.names['srcImg' + str(i)] = self.transform(
                Image.open(eval('self.srcImgpath' + str(i))).convert('RGB'))
            eval('self.srcImg' + str(i) + ' = self.srcImg' + str(
                i) + '[:, 0:256, 0:256]')
            if eval('self.srcImg' + str(i) + '.shape[0] == 1'):
                eval('self.srcImg' + str(i) + ' = torch.cat([self.srcImg' + str(
                    i) + ', self.srcImg' + str(i) + ', self.srcImg' + str(
                    i) + '], 0)')

        return [eval('self.srcImg' + str(i)) for i in
                range(1, self.multiplicity + 1)]

    def __len__(self):
        exec("assert len(self.files{}) == len(self.files{})".format(0, 1))
        return len(eval('self.files' + str(0)))

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class Test_Dataset(torch.utils.data.Dataset):
    """
    Dataset for testing
    """

    def __init__(self, root_dir="data", srcfilespath="BSD300_noisy",
                 tarfilespath="BSD300",
                 transform=torchvision.transforms.ToTensor()):
        self.srcfilespath = srcfilespath
        self.tarfilespath = tarfilespath
        self.root_dir = root_dir
        self.transform = transform

        self.srcfullpath = os.path.join(root_dir, srcfilespath)
        self.tarfullpath = os.path.join(root_dir, tarfilespath)

        self.srcfiles = [os.path.join(srcfilespath, x) for x in
                         os.listdir(self.srcfullpath) if is_image_file(x)]
        self.srcfiles = natsort.natsorted(self.srcfiles)
        self.tarfiles = [os.path.join(tarfilespath, x) for x in
                         os.listdir(self.tarfullpath) if is_image_file(x)]
        self.tarfiles = natsort.natsorted(self.tarfiles)

    def __getitem__(self, idx):

        srcImgpath = os.path.join(self.root_dir, self.srcfiles[idx])
        tarImgpath = os.path.join(self.root_dir, self.tarfiles[idx])

        srcImg = Image.open(srcImgpath).convert('RGB')
        tarImg = Image.open(tarImgpath).convert('RGB')

        srcImg = self.transform(srcImg)
        tarImg = self.transform(tarImg)

        if srcImg.shape[0] == 1:
            srcImg = torch.cat([srcImg, srcImg, srcImg], 0)
        if tarImg.shape[0] == 1:
            tarImg = torch.cat([tarImg, tarImg, tarImg], 0)

        return [srcImg, tarImg]

    def __len__(self):
        assert len(self.srcfiles) == len(self.tarfiles)
        return len(self.srcfiles)

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])
