import torch
import torchvision
import argparse
from torchvision import datasets, transforms
import math
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 100)
        self.dense1_bn = nn.BatchNorm1d(100) if args.batch_norm else None
        self.fc2 = nn.Linear(100, 50)
        self.dense2_bn = nn.BatchNorm1d(50) if args.batch_norm else None
        self.fc3 = nn.Linear(50, 10)
        nn.init.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2.weight, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc3.weight, gain=np.sqrt(2.0))


    def forward(self, x, args):
        x = x.view(-1, 28 * 28 * 1)
        x = F.relu(self.dense1_bn(self.fc1(x))) if args.batch_norm else F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=args.dropout_prob)
        x = F.relu(self.dense2_bn(self.fc2(x))) if args.batch_norm else F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=args.dropout_prob)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=2)  # input dims: 1 x 28 x 28
        self.conv1_drop = nn.Dropout2d(p = args.dropout_prob)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=2)  # input dims: 32 x 30 x 30
        self.conv2_drop = nn.Dropout2d(p = args.dropout_prob)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=2)  # input dims: 32 x 32 x 32
        self.conv3_drop = nn.Dropout2d(p=args.dropout_prob)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)  # input dims: 64 x 17 x 17
        self.conv4_drop = nn.Dropout2d(p=args.dropout_prob)
        self.pool = nn.MaxPool2d(2, 2)  # downsampling by half
        self.fc1 = nn.Linear(64 * 15 * 15, 100)  # 1x28X28 (conv->) 32x30X30 (conv->) 32x32x32 (conv->) 64x34x34 (pol->) 64x17x17 (conv->) 64x15x15
        self.dense1_bn = nn.BatchNorm1d(100) if args.batch_norm else None
        self.fc2 = nn.Linear(100, 50)
        self.dense2_bn = nn.BatchNorm1d(50) if args.batch_norm else None
        self.fc3 = nn.Linear(50, 10)
        nn.init.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2.weight, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc3.weight, gain=np.sqrt(2.0))


    def forward(self, x, args):
        x = F.relu(self.conv1_drop(self.conv1(x)))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = self.pool(F.relu(self.conv3_drop(self.conv3(x))))
        x = F.relu(self.conv4_drop(self.conv4(x)))
        x = x.view(-1, 64 * 15 * 15)
        x = F.relu(self.dense1_bn(self.fc1(x))) if args.batch_norm else F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=args.dropout_prob)
        x = F.relu(self.dense2_bn(self.fc2(x))) if args.batch_norm else F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=args.dropout_prob)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def read_data(args):

    # user python data loaders to create iterators over fashion mnist data
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=
                       transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]))

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True,
                      transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # get validation set from train set
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(math.floor(num_train*0.2))

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, sampler=train_sampler,
                                               num_workers=args.num_workers)
    # Create the validation loader
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.test_batch_size, sampler=validation_sampler,
                                                    num_workers=args.num_workers)

    return train_loader, validation_loader, test_loader


def plot_image(trainloader, args):
    def imshow(img):
        img = img / 2 + 0.5   # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(args.train_batch_size)))
    np.set_printoptions(threshold=np.nan)
    print(str(images.numpy()))


def train(args, model, train_loader, validation_loader, optimizer):

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        cumm_loss = correct = 0.0
        num_examples = 0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # zero the gradients
            output = model(data, args)  # get prediction
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            loss = F.nll_loss(output, target)  # the average batch loss
            loss.backward()
            optimizer.step()

            num_examples += len(target)
            cumm_loss += loss  # get the average batch loss
            correct += pred.eq(target.data.view_as(pred)).sum().item()  # sum correct predictions

        cumm_loss /= len(train_loader)  # get average loss over whole dataset
        correct /= num_examples
        val_cumm_loss, val_accuracy, _ = test(args, model, validation_loader)  # run model on validation set

        print('train Epoch: {}\ttrain Loss: {:.6f}\ttrain accuracy: {:.6f}\t'
          'val loss: {:.6f}\tval accuracy: {:.6f}'.format(
        epoch, cumm_loss, correct,val_cumm_loss, val_accuracy))


def test(args, model, loader, is_val = True):

    cumm_loss = correct = 0.0
    model.eval()
    num_examples = 0

    for batch_idx, (data, target) in enumerate(loader):
        output = model(data, args)  # get prediction
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        predictions = torch.cat((predictions, pred), 0) if batch_idx > 0 else pred.clone()  # get a vector with all the predictions
        cumm_loss += F.nll_loss(output, target, size_average=False).item()  # sum losses
        correct += pred.eq(target.view_as(pred)).sum().item()
        num_examples += len(target)

    cumm_loss /= num_examples  # get average loss over whole dataset
    correct /= num_examples

    if not is_val:
        print('test Loss: {:.6f}\ttest accuracy: {:.6f}\t'.format(
            cumm_loss, correct))
    return cumm_loss, correct, predictions


def run():
    # configurations
    parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--dropout-prob', type=float, default=0.0, metavar='N',
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--batch-norm', type=bool, default=False, metavar='N',
                        help='add batch normalization (default: False)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--convolution', type=bool, default=False, metavar='N',
                        help='True for convolution network and False for MLP (default: False)')
    parser.add_argument('--seed', type=int, default=222, metavar='S',
                        help='random seed (default: 111)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='S',
                        help='number of workers (default: 1)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # read data to data loader objects
    train_loader, validation_loader, test_loader = read_data(args)
    # plot_image(train_loader, args)

    len(train_loader.dataset)
    # initialize network
    if args.convolution:
        model = CNN(args)
    else:
        model = MLP(args)

    # set loss function and optimizer
    if args.momentum == 0.0:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(args, model, train_loader, validation_loader, optimizer)
    _, accuracy, predictions = test(args, model, test_loader, False)

    num_preds = predictions.numpy()
    np.savetxt("./predictions/test.pred.{:.6f}".format(accuracy), num_preds, fmt='%d')


if __name__ == '__main__':
    run()