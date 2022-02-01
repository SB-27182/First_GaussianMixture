import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


#So we will make the vectors, forward these vectors
batch_size = 251
class testDataSet(Dataset):
    def __init__(self):
        #initializing data rep [data loading]
        xy = np.loadtxt("./data/test_2_values.txt", delimiter=",", dtype=np.float32, skiprows=1)
        x = xy[:, [0]]#We want all the rows, and first column
        y = xy[:, 1:3]#We want all the rows, and second column       # TWO BRACKETS MEANS IT WILL BE A (276 x 1) 2-dim TENSOR
        self.x = torch.from_numpy(x)                                  # xy[:, 0] MEANS IT WILL BE A (276) 1-dim TENSOR
        self.y = torch.from_numpy(y)
        self.n__samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[i] for indexing
        return (self.x[index], self.y[index])


    def __len__(self):
        return self.n__samples



k = 3
d = 2

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
        self.lMu = nn.Linear(50, k*d, bias=True)
        self.lVars = nn.Linear(50, k*d, bias=True) #Not using full covariance. lSigSq only for every kernel.



    def forward(self, x):
        x = torch.tanh(self.lH1(x))
        x = torch.tanh(self.lH2(x))
        z = torch.tanh(self.lH3(x))

        pi = F.softmax(self.lPi(z), dim=1)
        mu = self.lMu(z)
        vars = torch.exp(self.lVars(z))


        return (pi, mu, vars)


def logLoss_MLE(target:torch.Tensor, pi:torch.Tensor, mu:torch.Tensor, vars:torch.Tensor):
    """Input sizes:
       target.size() = [batch, dim]
       pi.size() = [batch, kernel]
       mu.size() = var.size() = [batch, outputDim*kernel]
       covar.size() = [batch, dim*kernel*(dim-1)/2]"""

    mu = mu.view(batch_size, k, d)
    vars = vars.view(batch_size, k, d)

    target = target.view(batch_size, 1, d)

    i = 0
    losses = torch.zeros(size=(1, batch_size), device='cuda:0')
    while(i < k):  # 3 values in PMF of K
        k_mu = mu[:, [i], :]  #<------------------- THIS [:, [i]] NOTATION MEANS WE WANT THE TRANSPOSE
        k_var = vars[:, [i], :]


        likelihood = gaussian_pdf(target, k_mu, k_var)
        k_p = pi[:, [i]]

        losses = torch.log(k_p) + likelihood + losses #Sum all k kernels
        i = i + 1

    loss = torch.mean(-losses)
    return loss


def gaussian_pdf(y, mu, vars):
    sub = torch.sub(y, mu)
    sigma = diagMatrixMake(vars)
    clone = torch.clone(vars)
    ln_det = torch.sum(torch.log(clone))

    sigmaInv = torch.linalg.inv(sigma)
    a = torch.bmm(input=sub, mat2=sigmaInv)
    nobis = torch.bmm(input=a, mat2=torch.transpose(a, dim0=1, dim1=2))

    return -0.5*(ln_det + nobis) -0.5*d*np.log(2*np.pi)

def diagMatrixMake(vars):
    """Given a d*1 tensor, outputs a d*d tensor with non-diagnol values of 0."""
    diag = torch.diag_embed(vars, dim1=2, dim2=1)
    return diag.reshape([batch_size, d, d])




def linnearLoss_MLE(target:torch.Tensor, pi:torch.Tensor, mu:torch.Tensor, vars:torch.Tensor):
    """Input sizes:
       target.size() = [batch, dim]
       pi.size() = [batch, kernel]
       mu.size() = var.size() = [batch, outputDim*kernel]
       covar.size() = [batch, dim*kernel*(dim-1)/2]"""

    mu = mu.view(batch_size, k, d)
    vars = vars.view(batch_size, k, d)

    target = target.view(batch_size, 1, d)

    i = 0
    losses = torch.zeros(size=(1, batch_size), device='cuda:0')
    while(i < k):  # 3 values in PMF of K
        k_mu = mu[:, [i], :]  #<------------------- THIS [:, [i]] NOTATION MEANS WE WANT THE TRANSPOSE
        k_var = vars[:, [i], :]


        likelihood = linr_gaussian_pdf(target, k_mu, k_var)
        k_p = pi[:, [i]]

        losses = k_p * likelihood + losses
        i = i + 1

    loss = torch.mean(-losses)
    return loss


def linr_gaussian_pdf(y, mu, vars):
    sub = torch.sub(y, mu)

    sigma = diagMatrixMake(vars)
    clone = torch.clone(sigma)

    sigmaInv = torch.linalg.inv(sigma)
    a = torch.bmm(input=sub, mat2=sigmaInv)
    nobis = torch.bmm(input=a, mat2=torch.transpose(a, dim0=1, dim1=2))
    fac1 = torch.exp(-0.5 * nobis)

    fac2 = torch.pow(torch.sqrt(torch.linalg.det(clone)), exponent=-1)
    fac3 = np.sqrt(np.pi*2) ** -d

    return fac3 * fac2 * fac1





def main():

    def piHook(self, grad_input, grad_output):
        #grad_output = grad_output*10
       #print("THIS IS GRAD INPUT", grad_input)
       # print("THIS IS GRAD OUTPUT", grad_output)
        return grad_output*10

    def forwardHook(self, input, output):#You can just print in the forward itself. Hooks are for dynamic actions.
        print("==================================================================")
        print("")
        print("input", input)
        print("==================================================================")
        print("")
        print("output", output)
        print("==================================================================")
        print("")


#==========[model]===================
    model = Gauss_nnMLE()
    model.to(device="cuda:0")

#==========[dataLoader]==============
    dataset = testDataSet()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) #I THINK BATCH SIZES SHOULD BE A FACTOR OF THE DATA SIZE FOR GRADIENT DESCENT!
                                                                #  The reason is otherwise you will have the one last batch using less data!
    #Full batch size = gradient descent                       #   And the issue is that the gradient step still listens to this "less informed" gradient step
    #Not full batch size = stochastic gradient descent       #    Which is interesting; if the step size for the "less informed" was smaller it'd be ok.
#=======[test the loader]=============================
 #   first_data = dataset.__getitem__(0)
 #   (xVal, yLab) = first_data
 #   print("features", xVal)
 #   print("labels", yLab)

                                                      #So this lossrate is good, DO NOT GO ANY Bigger
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)


#=============[THEHOOKS]====================================
   # linearModule = model.get_submodule("lin2")  # DEFINITELY USING THIS
   # handleL = linearModule.register_backward_hook(linearGradHook)  #CANT EVEN REGISTER FULL BACKWARD HOOK FOR A FIRST LAYER LINEAR MODULE
  #  handleL1 = linearModule.register_forward_hook(forwardHook)
    #pi_module = model.get_submodule("lPi")
    #pi_handle = pi_module.register_full_backward_hook(piHook)


# ===========[THEPARAMETERS_TO_TRACK]=========================
   # print("THESE ARE THE PARAMETERS BY NAME", list(model.named_parameters()))
  #  linWeight = model.get_parameter("lin.weight")
    #ARE GRADIENTS PRINTABLE WITHOUT SAVING GRAD?
  #  print(type(linWeight)) #<---- THIS IS NOT THE TENSOR BUT A PARAMETER. WRAPPED TENSOR.
  #  print("DATA OF PARAMETER", linWeight.data)#<---- THE ACTUALL TENSOR
  #  wHandle = linWeight.register_hook(lambda grad: print("THIS IS THE WEIGHT PARAMETER GRAD", grad)) #<---THIS IS RETURNING NO BATCH RELATED

# ==================[Training]=======================================
    terminatorCount = 0
    oldLoss = 100
    smallest = 5
    terminate = False


    epoch = 0
    while(epoch<2000 and terminate==False):
        batchIter = iter(dataloader)

        i = 0
        while(i < dataloader.__len__()):
            try:
                (features, targets) = batchIter.__next__()
                features = features.to(device="cuda:0")
                targets = targets.to(device="cuda:0")

            except StopIteration:
                break

            #if (i%10 == 0):
            #    handleL.remove()

            #if (i%17 == 0) :
            #    linearModule.register_backward_hook(linearGradHook)

            (pi, mu, var) = model.forward(features)

            if (epoch == 1000):
                optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.000005)
            if(epoch > 1000):
                totalLoss = linnearLoss_MLE(targets, pi, mu, var)
            else:
                totalLoss = logLoss_MLE(targets, pi, mu, var)


            optimizer.zero_grad()
            totalLoss.backward()

            optimizer.step()


            if(totalLoss.item() >= oldLoss):
                terminatorCount += 1
            else:
                terminatorCount = 0
            oldLoss = totalLoss.item()

            if (terminatorCount > 50):
                 terminate = True

           # if (totalLoss.item() < -6.53351688):
           #     terminate = True
            if (totalLoss.item() < smallest):
                smallest = totalLoss.item()

            if epoch%1 == 0:
                print("epoch:", epoch + 1, "total iterations. ", "loss: ", totalLoss.item(), "  TerminatorCount: ", terminatorCount)
                # .item() GETS THE ACTUAL VALUE, MUST BE A 1,1 TENSOR


        epoch = epoch + 1




#========================[Save Model]=====================================================
    print("Model's state parameters")
    for p in model.state_dict():
        print(p, "\t", model.state_dict()[p].size(), model.state_dict()[p].data)

    torch.save(model.state_dict(), "./data/mixModel2.pt")



if __name__ == '__main__':
    main()
