# Test for 2D image Super-Resolution CNN 
import numpy as np
import torch
import torch.nn as nn
import random
from scipy.spatial import distance
import scipy
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt
# import cv2
# from perlin_noise import PerlinNoise


# Define the exponentiated quadratic 
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
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
def random_function_2d(xmin, xmax, ymin, ymax, nb_of_samples, number_of_functions):
    mean = 0
    std = 1
    num_samples = (nb_of_samples, nb_of_samples)
    Z = np.zeros((number_of_functions, nb_of_samples, nb_of_samples))
    for i in range(number_of_functions):
        samples = np.random.normal(mean, std, size=num_samples)
        z = cv2.GaussianBlur(samples,(91,91),100)
        Z[i,:,:] = z
    return Z

#---------------------------------------------------------------------------------------------------------------------------------------------
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def perlin_noise(xmin, xmax, ymin, ymax, nb_of_samples, number_of_functions):
    Z = np.zeros((number_of_functions, nb_of_samples, nb_of_samples))
    for i in range(number_of_functions):
        pic = generate_perlin_noise_2d([nb_of_samples, nb_of_samples], [2,2])
        Z[i,:,:] = pic
    return Z

#---------------------------------------------------------------------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, M, N, MID_CHANNEL, BATCH_SIZE) -> None:
        super(CNN, self).__init__()
        self.M = M
        self.N = N
        self.BATCH_SIZE = BATCH_SIZE
        self.MID_CHANNEL = MID_CHANNEL
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      #(1, 1, 128)
                in_channels=1,
                out_channels=self.MID_CHANNEL,
                kernel_size=(3,3),
                padding=1,
                padding_mode="circular"
            ),                              #->(1, 3, 128)
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv2 = nn.Sequential(         #(1, 3, 128)
            nn.Conv2d(MID_CHANNEL, self.MID_CHANNEL, (3,3), 
                      padding=1,
                      padding_mode="circular"),     
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv3 = nn.Sequential(         #(1, 3, 128)
            nn.Conv2d(MID_CHANNEL, self.MID_CHANNEL, (3,3), 
                      padding=1,
                      padding_mode="circular"),      
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv4 = nn.Sequential(         #(1, 3, 128)
            nn.Conv2d(MID_CHANNEL, self.MID_CHANNEL, (3,3), 
                      padding=1,
                      padding_mode="circular"),  
            nn.ReLU(),                      #->(1, 3, 128)
        )

        self.conv5 = nn.Sequential(         #(1, 3, 128)
            nn.Conv2d(MID_CHANNEL, self.MID_CHANNEL, (3,3), 
                      padding=1,
                      padding_mode="circular"),   
            nn.ReLU(),                      #->(1, 3, 128)
        )
       
        self.out = nn.Conv2d(self.MID_CHANNEL, self.N * self.N, (3,3),              #(1, 3, 128)
                      padding=1,
                      padding_mode="circular")                             #->(1, N, 128)

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.conv1(input)                  
        x = self.conv2(x)                   
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)             
        output = self.out(x)
        output = output.moveaxis(1,-1)
        output = torch.reshape(output, (batch_size , 1, self.M//self.N, self.M//self.N, self.N, self.N))
        output = output.moveaxis(-1,-3)
        output = torch.reshape(output, (batch_size , 1, self.M, self.M))
        return output

#---------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--resolution", default=2048, type=int)
    parser.add_argument("--reduction", default=32, type=int)  #Resolution reduction factor
    parser.add_argument("--number_f", default=100,type=int)
    args = parser.parse_args()

    xmin, xmax = -4, 4
    ymin, ymax = -4, 4
    M = args.resolution
    N = args.reduction
    number_of_functions = args.number_f # Number of functions to sample
    # Hyper Parameters
    EPOCH = 100
    BATCH_SIZE = 20
    LR = 0.001
    MID_CHANNEL = 128

    x_true = np.linspace(xmin, xmax, M)
    random_function = perlin_noise

    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    
    nb_of_samples = M           # Number of points in each function 
    y_true = random_function(xmin, xmax, ymin, ymax, nb_of_samples=M, number_of_functions=number_of_functions)
    y = y_true[:, 0::N, 0::N]          # Reduce resolution
    print(y_true.shape, y.shape)
    cnn = CNN(M, N, MID_CHANNEL, BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn.to(device)

    if torch.cuda.device_count() > 1:
        cnn = nn.DataParallel(cnn)

    optimize = torch.optim.Adam(cnn.parameters(), lr = LR)
    loss_func = torch.nn.MSELoss()
    assert y_true.shape[1] == y.shape[1]*N
    assert y_true.shape[2] == y.shape[2]*N
    y_true = torch.from_numpy(y_true).float().unsqueeze(1)
    y = torch.from_numpy(y).float().unsqueeze(1)
    y_true = y_true.to(device)
    y = y.to(device)

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
    Y_true = random_function(xmin, xmax, ymin, ymax, nb_of_samples=M, number_of_functions=3)
    Y = Y_true[:, 0::N, 0::N]      # Reduce resolution
    Y_true = torch.from_numpy(Y_true).float().unsqueeze(1)
    Y = torch.from_numpy(Y).float().unsqueeze(1)
    Y_true = Y_true.to(device)
    Y = Y.to(device)
    with torch.no_grad():
        cnn.eval()
        output = cnn(Y)
        # loss = loss_func(output, Y_true)
        output = output.cpu()
        Y_true = Y_true.cpu()
        for i in range(len(output)):
                loss = loss_func(output[i], Y_true[i])
                fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=400)
                plt.cla()
                ax[0].imshow(np.squeeze(output.data.numpy()[i]))
                ax[0].set_title(f"y")
                ax[1].imshow(np.squeeze(Y_true.numpy()[i]))
                ax[1].set_title(f"y_true")
                plt.show()
                plt.savefig(f"./SRCNN2d-cnn2d{i}.jpg")
