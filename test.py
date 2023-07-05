import torch
from dataset import Test_Dataset
from torch.utils.data import DataLoader
from libtiff import TIFF
import os
from skimage import io
import numpy as np
import time
import argparse
import sys
sys.path.append("utils/network")


def tifwrite(tif, tifpath):
    out_tiff = TIFF.open(tifpath, mode='w')
    tif_num, tif_depth, tif_width = tif.shape
    for i in range(0, tif_num):
        out_tiff.write_image(tif[i], compression=None)
    out_tiff.close()
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Test the model trained under OPD strategy.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strategy", default="RC", help="OPD-RC or OPD-AL")
    parser.add_argument("--batchsize", type=int, default=1,
                        help="The batch size")
    parser.add_argument("--gpu", default='0',
                        help="Available or visible cuda devices")
    parser.add_argument("--dataroot", default="data",
                        help="The root directory of training dataset")
    parser.add_argument("--srcpath", default="BSD300_noisy",
                        help="The directory of the testing data")
    parser.add_argument("--tarpath", default="BSD300",
                        help="The directory of the clean reference data")
    parser.add_argument("--modelroot", default="models",
                        help="The root directory of trained models")
    parser.add_argument("--outputroot", default="img",
                        help="The root directory of output images")
    parser.add_argument("--testdata", default="BSD300",
                        help="BSD300 or KODAK or SET14")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_prefix = "OPD-{}_ILSVRC2012val_AWGN".format(args.strategy)

    testData = Test_Dataset(root_dir=args.dataroot, srcfilespath=args.srcpath,
                            tarfilespath=args.tarpath)
    testDataloader = DataLoader(testData, batch_size=args.batchsize,
                                shuffle=False, num_workers=1)

    since = time.time()

    model_dir = os.path.join(args.modelroot, model_name_prefix)
    checkpoint = "{}/{}.pth".format(model_dir, model_name_prefix)
    model = torch.load(checkpoint, map_location=device)
    model.eval()

    output_dir = os.path.join(args.outputroot,
                              model_name_prefix + '_' + args.testdata)
    try:
        os.makedirs(output_dir, exist_ok=False)
    except:
        pass

    count = 0
    for noisy, _ in testDataloader:
        noisy = noisy.to(device)

        count += 1
        with torch.no_grad():
            pred = model(noisy)
            pred = pred.squeeze()
            pred = pred.permute(1, 2, 0).cpu().numpy()
            pred = np.uint8(np.clip(pred * 255, 0, 255))
        io.imsave(output_dir + "/output{}.jpg".format(count - 1), pred)
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
