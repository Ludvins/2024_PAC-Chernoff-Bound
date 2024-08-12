import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm



import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl


def non_latex_format():
    """ Set the matplotlib style to non-latex format """
    mpl.rcParams.update(mpl.rcParamsDefault)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.figsize"] = (16, 9)
    fontsize = 26
    matplotlib.rcParams.update({"font.size": fontsize})


def latex_format():
    """ Set the matplotlib style to latex format """
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{bm}",
        }
    )
    mpl.rc("font", family="Times New Roman")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.figsize"] = (16, 9)
    fontsize = 30
    matplotlib.rcParams.update({"font.size": fontsize})


def createmodel(k):
    """ Create a LeNet5 model with k times the number of channels. 
    
    Arguments
    ---------
    k : int
        Multiplies the number of channels in the layers of LeNet-5.
    random_seed : int
                  Random number for reproducibility.
    n_classes : int
                Number of classes in the dataset.
    n_channels : int
                 Number of channels in the input data.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    return LeNet5(n_classes, n_channels, k)


class LeNet5(nn.Module):
    def __init__(self, n_classes, input_channels, k):
        """ Initialize the LeNet-5 model with k times the number of channels.
        
        The model has 3 convolutional layers, 2 pooling layers, and 2 fully connected layers.
        The first convolutional layer has int(6k) output channels.
        The second convolutional layer has int(16k) output channels.
        The third convolutional layer has int(120k) output channels.
        
        Arguments
        ---------
        n_classes : int
            Number of classes in the dataset.
        input_channels : int
            Number of channels in the input data.
        k : int
            Multiplicative factor of the number of channels.
        
        """
        super(LeNet5, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=int(6 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(6 * k),
                out_channels=int(16 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.part3 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(16 * k),
                out_channels=int(120 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(120 * k), out_features=int(84)),
            nn.ReLU(),
            nn.Linear(in_features=int(84), out_features=n_classes),
        )

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, train_loader):
    es = EarlyStopper(patience = 2)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose = False)
    data_iter = iter(train_loader)
    iters_per_epoch = len(data_iter)
    aux_loss = 1
    tq = tqdm(range(N_ITERS))
    for it in tq:

            model.train()

            try:
                inputs, target = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(train_loader)
                inputs, target = next(data_iter)


            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            logits = model(inputs) # forward pass

            loss = criterion(logits, target) # supervised loss
            aux_loss += loss.detach().cpu().numpy()

            tq.set_postfix({'Train cce': loss.detach().cpu().numpy(),
                            "Patience": es.counter})


            loss.backward() # computes gradients
            optimizer.step()

            if it % iters_per_epoch == 0 and it != 0:
              scheduler.step()
              if aux_loss/iters_per_epoch < 0.01 or es.early_stop(aux_loss):
                break
              aux_loss = 0



    return model
    
def eval(device, model, loader, criterion):
    correct = 0
    total = 0
    losses = 0
    model.eval()
    with torch.no_grad():
        for data, targets in loader:
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, 1)
            correct += (predicted == targets).sum().detach().cpu().numpy()

            loss = criterion(logits, targets) # supervised loss
            losses += loss.detach().cpu().numpy() * targets.size(0)

    return correct, total, losses/total

def get_log_p(device, model, loader):
    cce = nn.CrossEntropyLoss(reduction="none")  # supervised classification loss
    aux = []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            logits = model(data)
            log_p = -cce(logits, targets)  # supervised loss
            aux.append(log_p)
    return torch.cat(aux)


#Binary Search for lambdas
def rate_function_BS(model, s_value):
  if (s_value<0):
    min_lamb=torch.tensor(-10000).to(device)
    max_lamb=torch.tensor(0).to(device)
  else:
    min_lamb=torch.tensor(0).to(device)
    max_lamb=torch.tensor(10000).to(device)

  s_value=torch.tensor(s_value).to(device)
  log_p = get_log_p(device, model, test_loader_batch)
  return aux_rate_function_TernarySearch(log_p, s_value, min_lamb, max_lamb, 0.001)

def eval_log_p(log_p, lamb, s_value):
  jensen_val=(torch.logsumexp(lamb * log_p, 0) - torch.log(torch.tensor(log_p.shape[0], device = device)) - lamb *torch.mean(log_p))
  return lamb*s_value - jensen_val

def aux_rate_function_BinarySearch(log_p, s_value, low, high, epsilon):

  while (high - low) > epsilon:
      mid = (low + high) / 2
      print(mid)
      print(eval_log_p(log_p, low, s_value))
      print(eval_log_p(log_p, mid, s_value))
      print(eval_log_p(log_p, high, s_value))
      print("--")
      if eval_log_p(log_p, mid, s_value) < eval_log_p(log_p, high, s_value):
          low = mid
      else:
          high = mid

  # Return the midpoint of the final range
  mid = (low + high) / 2
  return [eval_log_p(log_p, mid, s_value).detach().cpu().numpy(), mid.detach().cpu().numpy(), (mid*s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy()]


def aux_rate_function_TernarySearch(log_p, s_value, low, high, epsilon):

  while (high - low) > epsilon:
    mid1 = low + (high - low) / 3
    mid2 = high - (high - low) / 3

    if eval_log_p(log_p, mid1, s_value) < eval_log_p(log_p, mid2, s_value):
        low = mid1
    else:
        high = mid2

  # Return the midpoint of the final range
  mid = (low + high) / 2
  return [eval_log_p(log_p, mid, s_value).detach().cpu().numpy(), mid.detach().cpu().numpy(), (mid*s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy()]

import math
def aux_rate_function_golden_section_search(log_p, s_value, a, b, epsilon):
    """
    Maximizes a univariate function using the golden section search algorithm.

    Parameters:
        f (function): The function to minimize.
        a (float): The left endpoint of the initial search interval.
        b (float): The right endpoint of the initial search interval.
        tol (float): The error tolerance value.

    Returns:
        float: The x-value that minimizes the function f.
    """
    # Define the golden ratio
    golden_ratio = (torch.sqrt(torch.tensor(5).to(device)) - 1) / 2

    # Define the initial points
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    # Loop until the interval is small enough
    while abs(c - d) > epsilon:
        # Compute the function values at the new points
        fc = eval_log_p(log_p, c, s_value)
        fd = eval_log_p(log_p, d, s_value)

        # Update the interval based on the function values
        if fc > fd:
            b = d
            d = c
            c = b - golden_ratio * (b - a)
        else:
            a = c
            c = d
            d = a + golden_ratio * (b - a)

    # Return the midpoint of the final interval
    mid = (a + b) / 2
    return [eval_log_p(log_p, mid, s_value).detach().cpu().numpy(), mid.detach().cpu().numpy(), (mid*s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy()]



def eval_jensen(model, lambdas):
  log_p = get_log_p(device, model, test_loader_batch)
  return np.array(
      [
          (torch.logsumexp(lamb * log_p, 0) - torch.log(torch.tensor(log_p.shape[0], device = device)) - torch.mean(lamb * log_p)).detach().cpu().numpy() for lamb in lambdas
       ])

def inverse_rate_function(model, lambdas, rate_vals):
  jensen_vals = eval_jensen(model, lambdas)

  return np.array([ np.min((jensen_vals + rate)/lambdas) for rate in rate_vals])
