from dataset import RC_Dataset, AL_Dataset
from torch.utils.data import DataLoader
import os
from utils.network.unet_dc import UNet
from utils.trainRC import train_RC
from utils.trainAL import train_AL
import torch
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train modified U-Net with the ImageNet validation set (50000 images) under OPD strategy.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strategy", default="RC", help="OPD-RC or OPD-AL")
    parser.add_argument("--batchsize", type=int, default=64,
                        help="The batch size")
    parser.add_argument("--epochnum", type=int, default=300,
                        help="The number of epochs")
    parser.add_argument("--channelnum", type=int, default=3,
                        help="The number of the channels of input images")
    parser.add_argument("--lr", default=0.0072, help="The learning rate")
    parser.add_argument("--weightdecay", default=0.,
                        help="The weight decay of optimizer")
    parser.add_argument("--stepsize", type=int, default=10,
                        help="The step size of lr scheduler")
    parser.add_argument("--gamma", default=0.5,
                        help="The gamma of lr scheduler")
    parser.add_argument("--gpu", default='0',
                        help="Available or visible cuda devices")
    parser.add_argument("--multiplicity", type=int, default=8,
                        help="The number of multiple frames")
    parser.add_argument("--dataroot", default="data",
                        help="The root directory of training dataset")
    parser.add_argument("--datadir", default="ILSVRC2012_img_val_noisy",
                        help="The directory of training dataset")
    parser.add_argument("--modelroot", default="models",
                        help="The root directory of trained models")
    parser.add_argument("--trainsize", default=0.9,
                        help="The proportion of the data used for training")
    parser.add_argument("--valsize", default=0.1,
                        help="The proportion of the data used for validation")
    parser.add_argument("--balance", default=1.,
                        help="The hyperparameter to balance between MSE and MSA in alienation loss")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_prefix = "OPD-{}_ILSVRC2012val_AWGN".format(args.strategy)
    model_dir = os.path.join(args.modelroot, model_name_prefix)
    try:
        os.makedirs(model_dir, exist_ok=False)
    except:
        pass

    transformer = transforms.Compose([
        transforms.ToTensor()
    ])

    total_trainData = None
    if args.strategy == 'RC':
        total_trainData = RC_Dataset(root_dir=args.dataroot,
                                     filespath=args.datadir,
                                     multiplicity=args.multiplicity,
                                     transform=transformer)
    elif args.strategy == 'AL':
        total_trainData = AL_Dataset(root_dir=args.dataroot,
                                     filespath=args.datadir,
                                     multiplicity=args.multiplicity,
                                     transform=transformer)
    train_size = int(args.trainsize * len(total_trainData))
    val_size = int(args.valsize * len(total_trainData))

    abandon_data = len(total_trainData) - train_size - val_size
    trainData, valData, abandon_data = torch.utils.data.random_split(
        total_trainData, [train_size, val_size, abandon_data])

    dataloaders = {
        'train': DataLoader(trainData, batch_size=args.batchsize, shuffle=True,
                            num_workers=1),
        'val': DataLoader(valData, batch_size=args.batchsize, shuffle=True,
                          num_workers=1)
    }

    start = time.time()

    model = UNet(args.channelnum, args.channelnum).to(device)

    print("model", model)
    print('# Unet parameters:',
          sum(param.numel() for param in model.parameters()))

    optimizer_ft = optim.Adam(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.999), eps=1e-8,
                              weight_decay=args.weightdecay, amsgrad=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                           step_size=args.stepsize,
                                           gamma=args.gamma)

    best_epoch = 0
    if args.strategy == 'RC':
        model, best_epoch = train_RC(dataloaders=dataloaders, model=model,
                                     optimizer=optimizer_ft,
                                     scheduler=exp_lr_scheduler,
                                     num_epochs=args.epochnum,
                                     model_dir=model_dir,
                                     model_name_prefix=model_name_prefix,
                                     device=device)
    if args.strategy == 'AL':
        model, best_epoch = train_AL(dataloaders=dataloaders, model=model,
                                     optimizer=optimizer_ft,
                                     scheduler=exp_lr_scheduler,
                                     num_epochs=args.epochnum,
                                     model_dir=model_dir,
                                     model_name_prefix=model_name_prefix,
                                     device=device,
                                     balance=args.balance)

    model_out_path = "{}/{}_epoch{}_best.pth".format(model_dir,
                                                     model_name_prefix,
                                                     best_epoch)
    torch.save(model, model_out_path)

    toend = time.time() - start
    print('{:.0f}h {:.0f}m {:.0f}s'.format(toend // 3600, (toend % 3600) // 60,
                                           (toend % 3600) % 60))
