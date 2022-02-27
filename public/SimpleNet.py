# import the data loader
import os
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from DatasetParser import DatasetParser


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 4000),
            nn.ReLU(),
            nn.Linear(4000, 594),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits.view(-1, 99, 6)


class SimpleNet:
    def __init__(self, model=None, graph_name=None):
        print("\n cuda available?")
        print(torch.cuda.is_available())
        print("\n Torch version:")
        print(torch.__version__)

        self.epochs = 5000
        self.batch_size = 200
        self.output_folder = "../../output/"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.model = DNN()
        # set the number of thread
        """torch.set_num_threads(4)
        print (" ----Number of thread set to 4 ----")"""

        # save te current script and the data loader for the log
        copyfile("SimpleNet.py", self.output_folder + "SimpleNet.py")
        copyfile("DatasetParser.py", self.output_folder + "DatasetParser.py")

        # load data
        self.train_dataset = DatasetParser(False)
        self.test_dataset = DatasetParser(True)
        # # to save the pred
        self.save = True
        #
        # # create test and train loader
        self.test_loader = DataLoader(
             self.test_dataset, batch_size=len(self.test_dataset), drop_last=False
         )
        self.train_loader = DataLoader(
             self.train_dataset, batch_size=self.batch_size, drop_last=True
         )

        ### Neural network definition
        # fully connected layer

        self.n_h = 1000
        self.n_out = 1

        self.criterion = torch.nn.L1Loss()
        # Construct the optimizer (Adam in this case)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.loss_list = []
        self.mse_list = []

    def train_epoch(self):
        # Gradient Descent
        for i, (X, y) in enumerate(tqdm(self.train_loader)):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(X.float())

            # Compute and print loss
            loss = self.criterion(y_pred, y.float())
            # print("epoch: ", epoch, " loss: ", loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            self.optimizer.step()

    def test_epoch(self, epoch):
        print("--- test epoch ---")
        # Test
        size = 0
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for (X, y) in tqdm(self.test_loader):
                pred = self.model(X.float())
                test_loss += self.criterion(
                    pred, y.float()
                ).item()
                batch_pred = pred.detach().numpy()
                batch_y = y.detach().numpy()

        if epoch % 50 == 0:
            fig = plt.figure()
            for i in range(6):
                plt.subplot(3, 2, i + 1)
                plt.plot(
                    batch_pred[0, :, i] - batch_y[0, :, i]
                )
            fig.savefig(self.output_folder + "pred vs test" + str(epoch) + ".png")

        if self.save:
            np.save(self.output_folder + "y_pred", batch_pred)
            np.save(self.output_folder + "y_test", batch_y)

        self.loss_list.append(test_loss)

        print(
            f"Test Error: \n  Avg loss: {test_loss:>8f} \n"
        )


    def train(self):

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            self.train_epoch()
            self.test_epoch(epoch)

            if epoch % 100 == 0:
                print(self.output_folder)
                torch.save(self.model, self.output_folder + str(epoch) + ".pt")

            if epoch % 50 == 0:
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(self.loss_list)
                plt.title(" loss over epoch " + str(epoch))
                ax.legend()
                fig.savefig(self.output_folder + "loss" + str(epoch) + ".png")

        print("Done!")

if __name__ == '__main__':
    net = SimpleNet()
    net.train()
