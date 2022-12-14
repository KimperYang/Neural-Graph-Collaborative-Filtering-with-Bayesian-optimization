import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import linalg
from scipy.stats import norm
from scipy.stats import qmc
from NGCF import runNGCF
from utility.parser import changeArgs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
 
class GPR:
 
    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize
 
    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            return 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
 
        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]
        self.is_fit = True
 
    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
 
        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov
 
    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

def y_10d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y=np.empty((x.shape[0]))
    for i in range(x.shape[0]):
        X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = np.meshgrid(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][6],x[i][7],x[i][8],x[i][9])
        y[i]= ((X1*X1 + X2* X2 + X3*X3 + X4*X4 + X5* X5 + X6*X6+ X7*X7 + X8* X8 + X9*X9 + X10*X10).reshape(1,-1))[0][0]
    return -y

def ucb(x, gp):
    mu, std = gp.predict(x,return_std=True)
    return mu.ravel() + 0.5 * std

def acq_max(ac, gp, bounds):
    # Warm up with random points
    # x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
    #                              size=(1000, bounds.shape[0]))
    sampler4 = qmc.LatinHypercube(d=10)
    sample4 = sampler4.random(n=10000)
    x_tries=qmc.scale(sample4, l_bounds, u_bounds)
    

    ys = ac(x_tries, gp=gp)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    # x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
    #                             size=(250, bounds.shape[0]))

    sampler5 = qmc.LatinHypercube(d=10)
    sample5 = sampler5.random(n=250)
    x_seeds=qmc.scale(sample5, l_bounds, u_bounds)

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

sampler1 = qmc.LatinHypercube(d=6)
sample1 = sampler1.random(n=150)
l_bounds = [0.0001,1e-5,0,0,0,0]
u_bounds = [0.005,100,0.8,0.8,0.8,0.8]
train_X=[[0.0005,1e-5,0.1,0.1,0.1,0.1],
        [0.001,1,0.1,0.1,0.1,0.1],
        [0.0025,1e-5,0.2,0.1,0.1,0.1]]
# run NGCF:
# python NGCF.py --dataset amazon-book --regs trainX[1] --embed_size 64 --layer_size [64,64,64] --lr trainX[0] --save_flag 1 
# --pretain 0 --batch_size1024 --epoch 200 --verbose 50 --node_dropout trainX[2] --mess_dropout [trainX[3],trainX[4],trainX[5]]

train_y = [0.8444,0.7986,0.8127] #type in the result of NGCF (1-recall@20)
gpr = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        n_restarts_optimizer=25,
    )
gpr.fit(train_X, train_y)
Range=np.array([[0.0001,0.005],[1e-5,100],[0,0.8],[0,0.8],[0,0.8],[0,0.8]])
newSet=acq_max(ucb,gpr,Range).tolist()
print(newSet)