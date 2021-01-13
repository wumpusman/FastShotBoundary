"""example module demonstrating what this code does """
import torch
from fast_shot_boundary import FastShot
from train_fastshot import TrainFastShot

if __name__ == '__main__':
    fake_img = torch.rand(
        (5, 3, 20, 64,
         64))  #single batch of 5 blocks that are 20 long in dpeth
    ground_truth = torch.ones(3610).float()  #hard coded corresponding class

    fake_shot_net = FastShot()

    optimizer = torch.optim.SGD(fake_shot_net.parameters(), lr=.7)

    train_fake_shot = TrainFastShot(fake_shot_net, torch.nn.NLLLoss())

    final_result = train_fake_shot.train([(fake_img, ground_truth)],
                                         2,
                                         optimizer,
                                         is_training=True)

    print(final_result)
