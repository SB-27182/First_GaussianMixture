import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader



k=3
class Gauss_nnMLE(nn.Module):
    def __init__(self):
        #For Sig and Pi, we have a scalar for each mixture kernel.
        #For Mu, we have a t size vector; a mean for every dimension of y, and one of those said vectors for each kernel.
        super(Gauss_nnMLE, self).__init__()
                           #from  #into
        self.lH1 = nn.Linear(1, 50, bias=True)
        self.lH2 = nn.Linear(50, 100, bias=True)
        self.lH3 = nn.Linear(100,50, bias=True)
        self.lPi = nn.Linear(50, k, bias=True)
        self.lMu = nn.Linear(50, k*1, bias=True)
        self.lSigSq = nn.Linear(50, k, bias=True) #Not using full covariance. lSigSq only for every kernel.



    def forward(self, x):
       # print("")
      #  print("This is the weights", list(self.lH1.named_parameters()))
        x = torch.tanh(self.lH1(x))
        x = torch.tanh(self.lH2(x))
        z = torch.tanh(self.lH3(x))

        pi = F.softmax(self.lPi(z), dim=1)
        sigSq = torch.exp(self.lSigSq(z))
        mu = self.lMu(z)

        return (pi, mu, sigSq)

def mixCoefs_3Pi(model, ll, ul):
 #Given a model, and input values (x values) ll, ul, generates (ul-ll)*10 mixing coefficient values
 #This has to load (ul-ll)*10 models ... it takes a bit to run..
 #returns a tuple of FOUR lists: x, pi0, pi1, pi2
    """MODEL MUST BE IN EVAL MODE AND MODEL MUST BE MOVED TO CUDA GPU BEFORE CALLING THIS"""
    xList = []
    pi0List = []
    pi1List = []
    pi2List = []
    i = ll
    while(i < ul):
        tens_i = torch.tensor([[i]], device="cuda:0")
        distr = model(tens_i)
        pi = distr[0][0]

        pi0List.append(pi[0].item())
        pi1List.append(pi[1].item())
        pi2List.append(pi[2].item())
        xList.append(i)

        i = i + 0.1

    return (xList, pi0List, pi1List, pi2List)




class Distrx():
    def __init__(self, pi:torch.Tensor, mu:torch.Tensor, sig2:torch.Tensor, x:float):
        self.m = torch.distributions.multinomial.Multinomial(total_count=1, probs=pi)
        self.x = x

        g0 = (torch.tensor([1.0, 0.0, 0.0], device="cuda:0"), torch.distributions.normal.Normal(mu[0], sig2[0]))
        g1 = (torch.tensor([0.0, 1.0, 0.0], device="cuda:0"), torch.distributions.normal.Normal(mu[1], sig2[1]))
        g2 = (torch.tensor([0.0, 0.0, 1.0], device="cuda:0"), torch.distributions.normal.Normal(mu[2], sig2[2]))
        #g3 = (torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda:0"), torch.distributions.normal.Normal(mu[1], sig2[2]))
        self.kernels = [g0, g1, g2]

    def outPuts(self, count):
        "WARNING; RETURNS A LIST OF FLOATS!"""
        outputs = []
        i = 0
        while(i < count):
            m = self.m.sample()

            for j in self.kernels:
                if torch.equal(j[0], m):
                    target = j[1].sample()
                    target = target.item()

                    outputs.append(target)

            i=i+1
        return outputs



def main():
    model = Gauss_nnMLE()
    paramDict = torch.load("./data/mixModel1.pt")
    model.load_state_dict(paramDict)

    model.to(device="cuda:0")
    model.eval()


    with torch.no_grad():
        x = -3.0
        testOutputs = []
        testInputs = []
        while(x < 3.0):
            tensX = torch.tensor([[x]], device="cuda:0")
            testM = model(tensX)
            testDist = Distrx(pi=testM[0][0], mu=testM[1][0], sig2=testM[2][0], x=x)
            testOutputs.append((testDist.outPuts(20)))
            testInputs.append([x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x])
            x = x + 0.025
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.plot(testInputs, testOutputs, 'ro')
    plt.show()


    with torch.no_grad():
        tuple = mixCoefs_3Pi(model, -3.0, 3.0)
        xVals = tuple[0]
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(xVals, tuple[1], label="pi_0")
    ax.plot(xVals, tuple[2], label="pi_1")
    ax.plot(xVals, tuple[3], label="pi_2")

    plt.show()
#From an x value, through the marginal PMF for pi, sample from the 3 kernels. Make 50 samples each. Output array.


   # i = 0
  #  while(i < 600):
  #      print(tuple[1][i])
  #      i = i + 1






if __name__ == '__main__':
    main()
