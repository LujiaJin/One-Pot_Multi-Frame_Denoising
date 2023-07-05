import torch
from utils.loss import *
import copy
from collections import defaultdict
import time


def print_metrics(metrics, epoch_samples, phase):
    """
    Print temp metrics during training to terminal
    :param metrics: metrics to be printed, including loss, PSNR...
    :param epoch_samples: sample number of the processed epoch
    :param phase: train or val
    """

    outputs = []
    outputs_txt = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        outputs_txt.append("{}".format(metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_RC(dataloaders, model, optimizer, scheduler, num_epochs=25,
             model_dir='', model_name_prefix='', device='cuda'):
    """
    Training under OPD-RC strategy
    :param dataloaders: dataloaders
    :param model: model
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param num_epochs: number of epochs
    :param model_dir: directory path to save models
    :param model_name_prefix: model name
    :param device: device can be used
    :return: the best model and the best epoch number
    """

    best_loss = 1e10
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict)
    step = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for input_lowdose, input_stddose in dataloaders[phase]:

                input_lowdose = input_lowdose.to(device)
                input_stddose = input_stddose.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output_stddose = model(input_lowdose)
                    loss = calc_loss_RC(output_stddose, input_stddose, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += input_lowdose.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 1 == 0:
            model_out_path = "{}/{}_epoch{}.pth". \
                format(model_dir, model_name_prefix, epoch)
            torch.save(model, model_out_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val epoch: {:4f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch
