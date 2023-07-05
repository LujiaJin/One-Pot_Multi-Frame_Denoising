# -*- coding: UTF-8 -*-
import numpy as np
from skimage import io
import os
import natsort
from tqdm import tqdm
import argparse
import string
from PIL import Image, ImageDraw, ImageFont


def add_text_noise(image, noise_param):
    """
    Add random text strings with random font size and random color, placed in random locations within the image.
    :param image: the image to be corrupted
    :param noise_param: the probability of corrupted pixels, a static value or a variable range
    :return: the corrupted image with text noise
    """

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    height, width = image.shape[:2]

    # Set the range of the content, length and font size of random text strings
    str_len_range = (5, 30)
    char_pool = list(string.ascii_letters + string.digits)
    font_size_range = (15, 40)

    for y in range(height):
        for x in range(width):
            if np.random.rand() * 1200 < noise_param:
                # the average pixel number that each added text string corrupts equals to about 1,200
                # Randomly synthesize the text string
                str_len = np.random.randint(*str_len_range)
                font_size = np.random.randint(*font_size_range)
                font = ImageFont.truetype('./utils/times.ttf', size=font_size)
                text = ''.join(np.random.choice(char_pool) for _ in range(str_len))
                # calculate the size and location of the text to be added
                text_width, text_height = draw.textsize(text, font=font)
                text_x = x - text_width // 2
                text_y = y - text_height // 2 + font_size // 4
                # draw the text to the image
                draw.text((text_x, text_y), text, font=font, fill=tuple(np.random.randint(0, 256, size=3)))

    # transfer PIL image back to a numpy array
    noisy_img = np.array(pil_img)

    return noisy_img


def add_noise(root_dir="data/ILSVRC2012_img_val/",
              tar_dir="data/ILSVRC2012_img_val_AWGN/", noise_type='AWGN', noise_var=False, noise_param=25.):
    """
    Add noise of various types and levels to a given image dataset.
    :param root_dir: directory containing clean images
    :param tar_dir: directory containing the output noisy images
    :param noise_type: noise type to add, chosen from: AWGN (additive white Gaussian noise), SDPN (signal-dependent
    Poisson noise), MBN (multiplicative Bernoulli noise), MB (mixed-blind noise), RVIN (random-valued impulse noise),
    and TR (text removal)
    :param noise_var: noise level static or variable
    :param noise_param: noise parameter of the noise to be added. For example, standard deviration of the AWGN. For
    static MB, three noise parameters should be inputed. For variable noise except MB, two noise parameters should be
    inputed. For variable MB noise, six noise parameters should be inputed
    """

    files_Instance = os.listdir(root_dir)
    files_Instance = natsort.natsorted(files_Instance)

    for fi_Instance in tqdm(files_Instance, desc=tar_dir):
        fi_d_Instance = os.path.join(root_dir, fi_Instance)
        img = io.imread(fi_d_Instance)
        # Additive White Gaussian Noise
        if noise_type == 'AWGN':
            noise_scale = np.random.uniform(low=noise_param[0], high=noise_param[1]) if noise_var else noise_param
            GaussianNoise = np.array(np.random.normal(loc=0., scale=noise_scale, size=img.shape), dtype='float32')
            noisyImage = np.uint8(np.clip(img + GaussianNoise, 0, 255))
        # Signal-Dependent Poisson Noise
        elif noise_type == 'SDPN':
            noise_scale = np.random.uniform(low=noise_param[0], high=noise_param[1]) if noise_var else noise_param
            noisyImage = np.random.poisson(lam=((img / 255.) * noise_scale), size=None) / noise_scale
            noisyImage = np.uint8(np.clip(noisyImage * 255., 0, 255))
        # Multiplicative Bernoulli Noise
        elif noise_type == 'MBN':
            noise_scale = np.random.uniform(low=noise_param[0], high=noise_param[1]) if noise_var else noise_param
            BernoulliNoise = 1 - np.random.binomial(1, noise_scale, img.shape)
            noisyImage = np.uint8(np.clip(img * BernoulliNoise, 0, 255))
        # Mixed-Blind Noise
        elif noise_type == 'MB':
            noise_scale = [np.random.uniform(low=noise_param[0], high=noise_param[1]),
                           np.random.uniform(low=noise_param[2], high=noise_param[3]),
                           np.random.uniform(low=noise_param[4], high=noise_param[5])] if noise_var else noise_param
            # 1st step: add AWGN
            GaussianNoise = np.array(np.random.normal(loc=0., scale=noise_scale[0], size=img.shape), dtype='float32')
            Gauss_noisyImage = img + GaussianNoise
            Gauss_noisyImage[Gauss_noisyImage < 0.] = 0.
            # 2nd step: add SDPN
            Poisson_noisyImage = np.random.poisson(lam=((Gauss_noisyImage / 255.) * noise_scale[1]), size=None) / \
                                 noise_scale[1]
            # 3rd step: add MBN
            BernoulliNoise = 1 - np.random.binomial(1, noise_scale[2], img.shape)
            noisyImage = np.uint8(np.clip(Poisson_noisyImage * 255. * BernoulliNoise, 0, 255))
        # Random-Valued Impulsed Noise
        elif noise_type == 'RVIN':
            noise_scale = np.random.uniform(low=noise_param[0], high=noise_param[1]) if noise_var else noise_param
            random_matrix = np.random.rand(*img.shape)
            replace_mask = random_matrix < noise_scale
            noise_color = np.random.rand(*img.shape) * 255.
            new_color = img * (1 - noise_scale) + noise_color * noise_scale
            noisyImage = np.uint8(np.clip(np.where(replace_mask, new_color, img), 0, 255))
        # Text Removal
        elif noise_type == 'TR':
            noise_scale = np.random.uniform(low=noise_param[0], high=noise_param[1]) if noise_var else noise_param
            noisyImage = add_text_noise(img, noise_scale)

        io.imsave(os.path.join(tar_dir, fi_Instance), noisyImage)


def main():
    parser = argparse.ArgumentParser(
        description='Add noise to ImageNet validation set.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-rd", "--root_dir", type=str, default="data/ILSVRC2012_img_val/",
                        help="Directory containing clean images")
    parser.add_argument("-td", "--tar_dir", type=str, default="data/ILSVRC2012_img_val_AWGN/",
                        help="Directory containing the output noisy images")
    parser.add_argument("-nt", "--noise_type", type=str, choices=['AWGN', 'SDPN', 'MBN', 'MB', 'RVIN', 'TR'],
                        default="AWGN",
                        help="Noise type to add, chosen from: AWGN (additive white Gaussian noise), SDPN \
                             (signal-dependent Poisson noise), MBN (multiplicative Bernoulli noise), MB \
                             (mixed-blind noise), RVIN (random-valued impulse noise), and TR (text removal)")
    parser.add_argument("-nv", "--noise_var", action='store_true', help="Noise level static or variable")
    parser.add_argument("-np", "--noise_param", type=float, nargs='+', default=[25.],
                        help="Noise parameter of the noise to be added. For example, standard deviration of the AWGN. \
                             For static MB, three noise parameters should be inputed. For variable noise except MB, \
                             two noise parameters should be inputed. For variable MB noise, six noise parameters \
                             should be inputed.")
    args = parser.parse_args()

    # Create the target directory if not existing
    try:
        os.makedirs(args.tar_dir, exist_ok=True)
    except:
        pass

    if len(args.noise_param) == 1:  # noise level not variable
        add_noise(args.root_dir, args.tar_dir, args.noise_type, args.noise_var, args.noise_param[0])
    else:  # noise level variable
        add_noise(args.root_dir, args.tar_dir, args.noise_type, args.noise_var, args.noise_param)


if __name__ == "__main__":
    main()
