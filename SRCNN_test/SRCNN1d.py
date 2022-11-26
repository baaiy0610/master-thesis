# Test for 1D image Super-Resolution CNN 
import numpy as np
import torch
import torch.nn as nn
import random
from scipy.spatial import distance
import scipy
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the exponentiated quadratic 
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with sigma=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def gaussian_process(xmin, xmax, nb_of_samples, number_of_functions):
    # Sample from the Gaussian process distribution

    # Independent variable samples
    X = np.expand_dims(np.linspace(xmin, xmax, nb_of_samples), 1)
    Σ = exponentiated_quadratic(X, X)  # Kernel of data points

    # Draw samples from the prior at our data points.
    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=Σ, 
        size=number_of_functions)
    return np.squeeze(ys)

#---------------------------------------------------------------------------------------------------------------------------------------------
#Euler-Maruyama Method
def euler_maruyama(xmin, xmax, nb_of_samples, number_of_functions):
    y_true = np.zeros((number_of_functions, nb_of_samples))
    delta = (xmax-xmin) / nb_of_samples
    # Sample from the trigonometric_functions
    for i in range(number_of_functions):
        x = np.linspace(xmin, xmax, nb_of_samples)
        y0 = np.random.rand()
        r = np.random.rand()
        sigma = np.random.rand()
        for k in range(nb_of_samples):
            y = y0+r*delta*y0+y0*sigma*np.sqrt(delta)*np.random.normal(0,1)
            y0 = y
            y_true[i,k] = y0
    return y_true

#Stochastic Runge-Kuntta Method
def stochastic_rk(xmin, xmax, nb_of_samples, number_of_functions):
    y_true = np.zeros((number_of_functions, nb_of_samples))
    delta = (xmax-xmin) / nb_of_samples
    # Sample from the trigonometric_functions
    for i in range(number_of_functions):
        x = np.linspace(xmin, xmax, nb_of_samples)
        y0 = np.random.rand()
        r = np.random.rand()
        sigma = np.random.rand()
        for k in range(nb_of_samples):
            y_hat=y0+r*delta*y0+y0*sigma*np.sqrt(delta)*np.random.normal(0,1)
            y=y0+r*y0*delta+sigma*y0*np.sqrt(delta)*np.random.normal(0,1)+(sigma*y_hat-sigma*y0)*((np.sqrt(delta)*np.random.normal(0,1))**2-delta)/(2*np.sqrt(delta))
            y0=y
            y_true[i,k] = y0
    return y_true
#---------------------------------------------------------------------------------------------------------------------------------------------
# Define the sin function 
def f_sin(x):
    a = np.random.rand()
    b = np.random.rand()
    return b*np.sin(a*x)

# Define the cos function 
def f_cos(x):
    a = np.random.rand()
    b = np.random.rand()
    return b*np.cos(a*x)

def trigonometric_functions(xmin, xmax, nb_of_samples, number_of_functions):
    y_true = np.zeros((number_of_functions, nb_of_samples))
    x = np.linspace(xmin, xmax, nb_of_samples)
    f_list = [f_sin, f_cos]
    # Sample from the trigonometric_functions
    for i in range(number_of_functions):
        N = np.random.randint(1,1000)
        print(f"The number of trigonometric functions: {N}")
        c = np.random.rand()
        y_true[i] += c
        for n in range(N):
            f = random.choice(f_list)
            result = f(x)
            y_true[i] += result
    return y_true
#---------------------------------------------------------------------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, M, N, MID_CHANNEL, BATCH_SIZE) -> None:
        super(CNN, self).__init__()
        self.M = M
        self.N = N
        self.BATCH_SIZE = BATCH_SIZE
        self.MID_CHANNEL = MID_CHANNEL
        self.conv1 = nn.Sequential(
            nn.Conv1d(                      #(1, 1, 128)
                in_channels=1,
                out_channels=self.MID_CHANNEL,
                kernel_size=3,
                padding=1,
                padding_mode="circular"
            ),                              #->(1, 3, 128)
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv2 = nn.Sequential(         #(1, 3, 128)
            nn.Conv1d(MID_CHANNEL, self.MID_CHANNEL, 3, 
                      padding=1,
                      padding_mode="circular"),     
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv3 = nn.Sequential(         #(1, 3, 128)
            nn.Conv1d(MID_CHANNEL, self.MID_CHANNEL, 3, 
                      padding=1,
                      padding_mode="circular"),      
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv4 = nn.Sequential(         #(1, 3, 128)
            nn.Conv1d(MID_CHANNEL, self.MID_CHANNEL, 3, 
                      padding=1,
                      padding_mode="circular"),  
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv5 = nn.Sequential(         #(1, 3, 128)
            nn.Conv1d(MID_CHANNEL, self.MID_CHANNEL, 3, 
                      padding=1,
                      padding_mode="circular"),   
            nn.ReLU(),                      #->(1, 3, 128)
        )
       
        self.out = nn.Conv1d(self.MID_CHANNEL, self.N, 3,              #(1, 3, 128)
                      padding=1,
                      padding_mode="circular")     #->(1, N, 128)

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.conv1(input)                  
        x = self.conv2(x)                   
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)             
        output = self.out(x)
        output = output.transpose(1,2)
        output = torch.reshape(output, (batch_size , 1, self.M))
        return output

#---------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--x_min", default = -4.0, type=float)
    parser.add_argument("--x_max", default =  4.0, type=float)
    parser.add_argument("--resolution", default=1024, type=int)
    parser.add_argument("--reduction", default=2, type=int)  #Resolution reduction factor
    parser.add_argument("--number_f", default=1000,type=int)
    args = parser.parse_args()

    xmin = args.x_min
    xmax = args.x_max
    M = args.resolution
    N = args.reduction
    number_of_functions = args.number_f # Number of functions to sample
    # Hyper Parameters
    EPOCH = 100
    BATCH_SIZE = 100
    LR = 0.001
    MID_CHANNEL = 128
    try:
        assert M % N == 0
    except AssertionError:
        print("Error:Cannot scale to specified resolution")
    else:
        x_true = np.linspace(xmin, xmax, M)
        random_function = gaussian_process

        cnn = CNN(M, N, MID_CHANNEL, BATCH_SIZE)
        optimize = torch.optim.Adam(cnn.parameters(), lr = LR)
        loss_func = torch.nn.MSELoss()

        use_gpu = torch.cuda.is_available()
        print("GPU is avalilable:", use_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Which device is used
        print(device)


        nb_of_samples = M           # Number of points in each function 
        y_true = random_function(xmin, xmax, nb_of_samples=M, number_of_functions=number_of_functions)
        y = y_true[:,0::N]          # Reduce resolution
        assert y_true.shape[1] == y.shape[1]*N
        y_true = torch.from_numpy(y_true).float().unsqueeze(1)
        y = torch.from_numpy(y).float().unsqueeze(1)
        y_true = y_true.to(device)
        y = y.to(device)
        cnn.to(device)
        if torch.cuda.device_count() > 1:
            cnn = nn.DataParallel(cnn)

        # Initialize Dataloader
        torch_dateset = Data.TensorDataset(y, y_true)
        loader = Data.DataLoader(
                dataset=torch_dateset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )

        cnn.train()
        for epoch in range(EPOCH):
            loop = tqdm(enumerate(loader), total =len(loader), ascii=True, colour="#B266FF")
            for index,(x, target)in loop:
                x = x.to(device)
                target = target.to(device)
                predict = cnn(x)
                loss = loss_func(predict,target)
                optimize.zero_grad()
                loss.backward()
                optimize.step()
                # Update information
                loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
                loop.set_postfix(loss = loss.item())

        # Validation 
        Y_true = random_function(xmin, xmax, nb_of_samples=M, number_of_functions=3)
        Y = Y_true[:, 0::N]      # Reduce resolution
        Y_true = torch.from_numpy(Y_true).float().unsqueeze(1)
        Y = torch.from_numpy(Y).float().unsqueeze(1)
        Y_true = Y_true.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            cnn.eval()
            output = cnn(Y)
            output = output.cpu()
            Y_true = Y_true.cpu()
            # loss = loss_func(output, Y_true)
            for i in range(len(output)):
                    loss = loss_func(output[i], Y_true[i])
                    plt.cla()
                    plt.scatter(x_true, np.squeeze(Y_true.numpy()[i]), c='blue')
                    plt.scatter(x_true, np.squeeze(output.data.numpy()[i]), c='orange')
                    plt.text(0.5, 0.5, 'Epoch=%d Loss=%.4f' % (epoch, loss.data.numpy()), fontdict={'size': 15, 'color': 'red'}, horizontalalignment="center", verticalalignment="top",)
                    plt.title(f"{random_function.__name__}-{i}")
                    plt.savefig(f"./SRCNN1d-{i}.jpg")
                    plt.show()
                    # plt.pause(2)
            print('Validation Loss=%.8f' % loss.data.numpy())

