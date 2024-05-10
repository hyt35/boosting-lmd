import torch
import torchvision
import torchvision.transforms as transforms
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import time
import tqdm
device = 'cuda'
class CNNModelSmall(nn.Module):
    # 3x3 convolution
    def __init__(self):
        super(CNNModelSmall, self).__init__()
        
        self.filters = nn.ModuleList([nn.MaxPool2d(2,2),
                                      nn.Conv2d(1,8,3), #(14,14) -> (8,12,12)
                                      nn.ReLU(),
                                      nn.MaxPool2d(2,2), # (32, 26, 26) -> (32, 6,6)
                                      nn.Flatten(),
                                      nn.Linear(288, 10)
                                      ])

    def forward(self, x):
        for filter in self.filters:
            x = filter(x)
        out = F.log_softmax(x, dim=1)
        return out

class FiveLayerDense(nn.Module):
    def __init__(self):
        super(FiveLayerDense, self).__init__()
        self.filters = nn.ModuleList([nn.Flatten(),
                                      nn.Linear(784,50), nn.ReLU(),
                                      nn.Linear(50,40), nn.ReLU(),
                                      nn.Linear(40,30), nn.ReLU(),
                                      nn.Linear(30,20), nn.ReLU(),
                                      nn.Linear(20,10)])
    def forward(self, x):
        for filter in self.filters:
            x = filter(x)
        out = F.log_softmax(x, dim=1)
        return out
    
def test_acc(pred, target):
    return torch.sum(torch.argmax(pred, dim=1) == target) / len(target)


dataset = omniglot("data", ways=10, shots=5, test_shots=15, meta_train=True, download=True)
# dataset = omniglot("data", ways=10, shots=5, test_shots=15, meta_test=True, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)# print(len(dataloader))
# ctr = 2

lossFn = torch.nn.NLLLoss(reduction='mean')

ctr = 500
bsize = 1




dataloader = BatchMetaDataLoader(dataset, batch_size=bsize, num_workers=4)# print(len(dataloader))



num = ctr
num_optims = 20
avg_start, avg_end = 0,0
avgs = torch.zeros(num_optims).to(device)
start = time.time()
for _, batch in tqdm.tqdm(zip(range(num), dataloader)):
    train_inputs, train_targets = batch["train"]
    # print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
    # print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

    test_inputs, test_targets = batch["test"]
    # print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
    # print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)

    train_inputs, train_targets, test_inputs, test_targets = train_inputs.to(device), train_targets.to(device), test_inputs.to(device), test_targets.to(device)



    for task_idx, (train_input, train_target, test_input,
        test_target) in enumerate(zip(train_inputs, train_targets,
        test_inputs, test_targets)):

        # print(train_input.shape)
        model = CNNModelSmall().to(device)
        # model = FiveLayerDense().to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        avg_start += test_acc(model(test_input), test_target)
        for i in range(num_optims):
            
            opt.zero_grad()
            loss = lossFn(model(train_input), train_target)
            loss.backward()
            opt.step()

            avgs[i] += test_acc(model(test_input), test_target)
            # print(ctr, i, test_acc(model(test_inputs), test_targets))
        avg_end += test_acc(model(test_input), test_target)


    # ctr -= 1
    # if ctr == 0:
    #     break
end = time.time()
print('elapsed', end-start)
print(avg_start/(num*bsize) , avg_end/(num*bsize))
print(avgs/(num*bsize))
    # foo = make_grid(train_inputs[0])
    # save_image(foo, str(ctr)+'tr.png')
    # foo = make_grid(test_inputs[0])
    # save_image(foo, str(ctr)+'te.png')
    # ctr -= 1
    # if ctr == 0:
    #     break

# omniglot_dataset = torchvision.datasets.Omniglot('/local/scratch/public/hyt35/datasets/', background=True, 
#                                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((28,28))]))

