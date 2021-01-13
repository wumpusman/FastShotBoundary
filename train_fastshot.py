"""main component for trianing"""
import torch
from torch import nn
import torch.nn.functional as F


class TrainFastShot:
    """
    Training wrapper to evaluate model, can be extended
    """
    def __init__(self, model, loss_function, skew_loss_function=False):
        self._loss_function = loss_function
        self._model = model

        if skew_loss_function:
            self.skew_loss_function(self._loss_function)

    def get_model(self):
        return self._model

    def skew_loss_function(self):
        """
        Changes loss function to handle for class imbalanace
        To DO, downweight the negative class (no shot boundary)
        :return:
        """
        pass

    def set_loss_function(self, loss_function, skew_loss_function=False):
        """
        sets the loss function
        """
        self._loss_function = loss_function

    def prepare_loss(self, batch):
        """
        adaptable way to handle different loss functions, or extending it for specific use cases
        :param batch:
        :return:
        """
        x = batch[0]
        y = batch[1]

        result = self.get_model()(x)
        #align shapes for the output with class labels of y
        result = result.permute(0, 2, 3, 4, 1)
        result = result.reshape(
            result.shape[0] * result.shape[1] * result.shape[2] *
            result.shape[3], 2)
        return self._loss_function(result, y.long())

    def train(self, data, epochs, optimizer, is_training):
        """
        :param data: data to be iterated over, a tuple of x and y, should really be a dataloader object in the future
        :param epochs: how many interations
        :param optimizer: optimizer for training
        :param is_training: is it training or evaluation
        :return:
        """
        loss_vals = []
        for time in range(epochs):
            for i in range(len(data)):
                x = data[i][0]
                y = data[i][1]
                loss = 0
                if is_training == True:
                    loss = self.prepare_loss((x, y))
                else:
                    with torch.no_grad():
                        loss = self.prepare_loss((x, y))

                optimizer.zero_grad()

                loss_vals.append(loss.cpu().detach().numpy())

                if is_training:
                    loss.backward()
                    optimizer.step()

        return loss_vals

    def score(self, predictions):
        """
        provides a score evaluation of performance
        :param predictions:
        :return:
            float
        """
        pass
