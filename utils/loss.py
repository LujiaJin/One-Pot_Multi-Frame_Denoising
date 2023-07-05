import torch.nn as nn


def calc_loss_RC(pred, target, metrics):
    """
    Calculate loss under OPD-RC strategy
    :param pred: the predicted clan image
    :param target: the reference clean ground truth
    :param metrics: the container to hold loss
    :return: loss
    """

    criterion_MSE = nn.MSELoss(reduction='mean')
    loss = criterion_MSE(pred, target)
    metrics['loss'] += loss

    return loss


def calc_loss_AL(output, input, metrics, balance):
    """
    Calculate loss under OPD-AL strategy
    :param output: the output result of networks
    :param input: the input noisy images
    :param metrics: the container to hold loss
    :param balance: hyperparameter to balance between MSE and MSA
    :return: loss
    """

    criterion_MSE = nn.MSELoss(reduction='mean')
    loss_MSE = 0
    loss_MSA = 0
    for i in range(input.size(0)):
        for j in range(input.size(1)):
            for k in range(input.size(1)):
                if j != k:
                    loss_MSE += criterion_MSE(output[i][j], input[i][k])
                loss_MSA += criterion_MSE(output[i][j], output[i][k])
    loss = loss_MSE / input.size(1) / (
                input.size(1) - 1) + balance * loss_MSA / 2 / input.size(1) ** 2
    metrics['loss'] += loss

    return loss
