# %matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from torch.nn import init
from sklearn.model_selection import train_test_split
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.FloatTensor)
print(torch.__version__)

# device
device = torch.device("cpu")


# print(1)
import os
print(os.getcwd())

from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all" 

# define the NN model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.3),  # dropout训练
            nn.Linear(hidden_size, hidden_size),
            
            #nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            #nn.Dropout(p=0.3),
            
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=0.3),
            
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.fc1(x)       #fully connected layer 1
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out) #dropout也为hyperparameter
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(ResNetMLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer1 = ResidualBlock(hidden_size, dropout)
        self.layer2 = ResidualBlock(hidden_size, dropout)
        self.layer3 = ResidualBlock(hidden_size, dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)   #nn.Linear(input_size, hidden_size)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)     #dropout也为hyperparameter
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)      #经过三层ResidualBlock
        out = self.output_layer(out)
        return out


# loss function
loss_fn = torch.nn.MSELoss().to(device)

def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss_fn(clipped_preds.log(),labels.log()).mean())
    return rmse.item()

expr = "PROGENy" #input data name
x_data = pd.read_csv(f"{expr}_input.csv")
# print('>>>>>>> The input data is:', x_data.shape)
# print(x_data.head())
x_data_drop = x_data.iloc[:, 4:]
x_data_t = x_data_drop.T
print('>>>>>>> The input data is:', x_data_t.shape)
print(x_data_t.head())
x_data_t = x_data_t.apply(pd.to_numeric, errors='coerce')
print('>>>>>>> The input data is:', x_data_t.shape)
print(x_data_t.head())
X_test_Tensor = torch.tensor(x_data_t.values)
X_test_Tensor = X_test_Tensor.float()
X_test_Tensor = torch.nan_to_num(X_test_Tensor, nan=0.0)

#output result
net=torch.load("/Users/xincao/Library/Mobile Documents/com~apple~CloudDocs/Study/Research/2024.10_TFA/Sicmirkan - Trained/414_hkm_cuda3.pth",map_location=torch.device('cpu'))
net.eval()
# print(net)

output = net(X_test_Tensor)


#net = net.to("cpu")
#y_train_out = pd.DataFrame(net(X_train_Tensor).cpu().detach().numpy())  
y_test_out  = pd.DataFrame(net(X_test_Tensor).detach().numpy())   #prediction output
#'hsa-miR-335-5p', 
miRNAs = ['hsa-miR-335-5p', 'hsa-miR-338-3p', 'hsa-miR-324-3p', 'hsa-miR-18a-3p', 'hsa-miR-299-5p', 'hsa-miR-483-5p', 'hsa-miR-885-5p', \
'hsa-miR-193b-5p', 'hsa-miR-146b-3p', 'hsa-miR-885-3p', 'hsa-miR-590-3p', 'hsa-miR-625-3p', 'hsa-miR-628-5p', 'hsa-miR-539-3p', 'hsa-let-7a-3p',\
 'hsa-miR-584-3p', 'hsa-miR-365b-3p', 'hsa-miR-532-3p', 'hsa-miR-543', 'hsa-miR-937-3p', 'hsa-miR-518a-3p', 'hsa-miR-518d-3p', 'hsa-miR-517c-3p',\
 'hsa-miR-520h', 'hsa-miR-302d-3p', 'hsa-miR-365a-3p', 'hsa-miR-302b-3p', 'hsa-miR-302c-3p', 'hsa-miR-1224-5p', 'hsa-miR-1224-3p', 'hsa-miR-519a-5p',\
 'hsa-miR-518a-5p', 'hsa-miR-136-5p', 'hsa-miR-146a-5p', 'hsa-miR-126-5p', 'hsa-miR-126-3p', 'hsa-miR-127-3p', 'hsa-miR-134-5p', 'hsa-miR-191-5p',\
 'hsa-miR-9-5p', 'hsa-miR-9-3p', 'hsa-miR-125a-5p', 'hsa-miR-552-5p', 'hsa-miR-487b-5p', 'hsa-miR-520g-5p', 'hsa-miR-519d-5p', 'hsa-miR-510-3p',\
 'hsa-miR-629-5p', 'hsa-miR-671-3p', 'hsa-miR-22-5p', 'hsa-miR-21-3p', 'hsa-miR-20a-3p', 'hsa-miR-133a-5p', 'hsa-miR-200c-5p', 'hsa-miR-5584-5p',\
 'hsa-miR-3622a-3p', 'hsa-miR-186-5p', 'hsa-miR-185-5p', 'hsa-miR-154-3p', 'hsa-miR-150-5p', 'hsa-miR-149-5p', 'hsa-miR-5589-3p', 'hsa-miR-1295b-3p',\
 'hsa-miR-1295b-5p', 'hsa-miR-513c-3p', 'hsa-miR-1306-5p', 'hsa-let-7d-3p', 'hsa-let-7e-3p', 'hsa-miR-455-5p', 'hsa-miR-514a-5p', 'hsa-miR-410-3p', \
'hsa-miR-485-3p', 'hsa-miR-484', 'hsa-miR-485-5p', 'hsa-miR-487a-3p', 'hsa-miR-4728-3p', 'hsa-miR-381-3p', 'hsa-miR-382-5p', 'hsa-miR-380-5p', \
'hsa-miR-380-3p', 'hsa-miR-378a-3p', 'hsa-miR-379-5p', 'hsa-miR-378a-5p', 'hsa-miR-449c-5p', 'hsa-miR-223-3p', 'hsa-miR-224-5p', 'hsa-miR-519e-5p', \
'hsa-miR-519e-3p', 'hsa-miR-498-5p', 'hsa-miR-520e-3p', 'hsa-miR-515-5p', 'hsa-miR-515-3p', 'hsa-miR-181d-5p', 'hsa-miR-512-5p', 'hsa-miR-512-3p', \
'hsa-miR-338-5p', 'hsa-miR-194-5p', 'hsa-miR-195-5p', 'hsa-miR-770-5p', 'hsa-miR-542-3p', 'hsa-miR-1976', 'hsa-miR-4746-5p', 'hsa-miR-369-3p', \
'hsa-miR-376c-3p', 'hsa-miR-371a-3p', 'hsa-miR-370-3p', 'hsa-miR-373-5p', 'hsa-miR-372-3p', 'hsa-miR-374a-5p', 'hsa-miR-373-3p', 'hsa-miR-376a-3p',\
 'hsa-miR-375-3p', 'hsa-miR-525-3p', 'hsa-miR-525-5p', 'hsa-miR-519b-3p', 'hsa-miR-526b-3p', 'hsa-miR-520a-3p', 'hsa-miR-520a-5p', 'hsa-miR-519c-3p',\
 'hsa-miR-520f-3p', 'hsa-miR-200c-3p', 'hsa-miR-7706', 'hsa-miR-1298-3p', 'hsa-miR-668-3p', 'hsa-miR-671-5p', 'hsa-miR-136-3p', 'hsa-miR-138-1-3p', \
'hsa-miR-127-5p', 'hsa-miR-129-2-3p', 'hsa-miR-125a-3p', 'hsa-miR-766-3p', 'hsa-miR-145-3p', 'hsa-miR-517-5p', 'hsa-miR-524-3p', 'hsa-miR-16-2-3p',\
 'hsa-miR-101-5p', 'hsa-miR-96-3p', 'hsa-miR-99a-3p', 'hsa-miR-29b-2-5p', 'hsa-miR-425-5p', 'hsa-miR-1266-5p', 'hsa-miR-203a-3p', 'hsa-miR-204-5p',\
 'hsa-miR-205-5p', 'hsa-miR-210-3p', 'hsa-miR-183-5p', 'hsa-miR-199b-5p', 'hsa-miR-211-5p', 'hsa-miR-511-5p', 'hsa-miR-146b-5p', 'hsa-miR-488-5p',\
 'hsa-let-7c-5p', 'hsa-let-7b-5p', 'hsa-miR-15a-5p', 'hsa-miR-16-5p', 'hsa-miR-200b-3p', 'hsa-miR-150-3p', 'hsa-miR-193a-5p', 'hsa-miR-30a-3p',\
 'hsa-miR-31-5p', 'hsa-miR-27a-3p', 'hsa-miR-28-5p', 'hsa-miR-29a-3p', 'hsa-miR-30a-5p', 'hsa-miR-24-3p', 'hsa-miR-25-3p', 'hsa-miR-31-3p',\
 'hsa-miR-92a-1-5p', 'hsa-miR-27a-5p', 'hsa-miR-28-3p', 'hsa-miR-93-3p', 'hsa-miR-487b-3p', 'hsa-miR-128-2-5p', 'hsa-miR-335-3p', 'hsa-miR-219a-5p',\
 'hsa-miR-218-5p', 'hsa-miR-215-5p', 'hsa-miR-214-3p', 'hsa-miR-181a-3p', 'hsa-miR-222-3p', 'hsa-miR-221-3p', 'hsa-miR-432-5p', 'hsa-miR-495-3p', \
'hsa-miR-202-3p', 'hsa-miR-202-5p', 'hsa-miR-193b-3p', 'hsa-miR-496', 'hsa-miR-15b-5p', 'hsa-miR-1-3p', 'hsa-miR-27b-3p', 'hsa-miR-23b-3p', 'hsa-miR-17-3p', \
'hsa-miR-17-5p', 'hsa-miR-19a-3p', 'hsa-miR-18a-5p', 'hsa-miR-20a-5p', 'hsa-miR-19b-3p', 'hsa-miR-22-3p', 'hsa-miR-21-5p', 'hsa-miR-23a-3p', 'hsa-miR-101-3p',\
 'hsa-miR-100-5p', 'hsa-miR-99a-5p', 'hsa-miR-96-5p', 'hsa-miR-93-5p', 'hsa-miR-92a-3p', 'hsa-miR-32-5p', 'hsa-miR-509-3-5p', 'hsa-miR-934', 'hsa-miR-431-3p',\
'hsa-miR-223-5p', 'hsa-miR-200b-5p', 'hsa-miR-4677-3p', 'hsa-miR-6510-3p', 'hsa-miR-1251-5p', 'hsa-miR-129-5p', 'hsa-miR-148a-3p', 'hsa-miR-30d-5p', \
'hsa-miR-518e-3p', 'hsa-miR-527', 'hsa-miR-942-5p', 'hsa-miR-944', 'hsa-miR-522-3p', 'hsa-miR-519a-3p', 'hsa-miR-155-5p', 'hsa-miR-1273h-3p', \
'hsa-miR-135a-5p', 'hsa-miR-137-3p', 'hsa-miR-133a-3p', 'hsa-miR-128-3p', 'hsa-miR-124-3p', 'hsa-miR-125b-5p', 'hsa-miR-30b-5p', 'hsa-miR-122-5p',\
 'hsa-miR-552-3p', 'hsa-miR-92b-3p', 'hsa-miR-199b-3p', 'hsa-miR-219a-1-3p', 'hsa-miR-214-5p', 'hsa-miR-222-5p', 'hsa-miR-652-3p', 'hsa-miR-449b-5p',\
 'hsa-miR-411-5p', 'hsa-miR-378c', 'hsa-miR-514b-3p', 'hsa-miR-514b-5p', 'hsa-miR-3200-3p', 'hsa-miR-1180-3p', 'hsa-miR-1179', 'hsa-miR-501-5p',\
 'hsa-miR-500a-3p', 'hsa-miR-513a-5p', 'hsa-miR-507', 'hsa-miR-506-3p', 'hsa-miR-182-5p', 'hsa-miR-181c-5p', 'hsa-miR-34a-5p', 'hsa-miR-10b-5p', 'hsa-miR-181b-5p',\
 'hsa-miR-181a-5p', 'hsa-miR-139-5p', 'hsa-miR-10a-5p', 'hsa-miR-7-5p', 'hsa-miR-1226-3p', 'hsa-miR-513a-3p', 'hsa-miR-501-3p', 'hsa-miR-153-3p', 'hsa-miR-152-3p',\
 'hsa-miR-138-5p', 'hsa-miR-143-3p', 'hsa-miR-760', 'hsa-miR-145-5p', 'hsa-miR-509-5p', 'hsa-miR-181a-2-3p', 'hsa-miR-181c-3p', 'hsa-miR-7-2-3p', 'hsa-miR-139-3p',\
 'hsa-miR-30c-2-3p', 'hsa-miR-30d-3p', 'hsa-miR-140-5p', 'hsa-miR-142-5p', 'hsa-miR-92b-5p', 'hsa-miR-141-3p', 'hsa-miR-142-3p', 'hsa-miR-508-5p', 'hsa-miR-495-5p',\
 'hsa-miR-758-5p', 'hsa-miR-374b-5p', 'hsa-miR-24-2-5p', 'hsa-miR-128-1-5p', 'hsa-miR-655-3p', 'hsa-miR-19b-1-5p', 'hsa-miR-155-3p', 'hsa-miR-518c-3p',\
 'hsa-miR-524-5p', 'hsa-miR-520c-3p', 'hsa-miR-518c-5p', 'hsa-miR-518b', 'hsa-miR-518f-3p', 'hsa-miR-520b-3p', 'hsa-miR-523-3p', 'hsa-miR-518f-5p', 'hsa-miR-215-3p',\
 'hsa-miR-301a-3p', 'hsa-miR-99b-5p', 'hsa-miR-200a-3p', 'hsa-miR-302a-5p', 'hsa-miR-106b-5p', 'hsa-miR-29c-3p', 'hsa-miR-302a-3p', 'hsa-miR-192-5p', \
'hsa-miR-196a-5p', 'hsa-miR-197-3p', 'hsa-miR-513b-5p', 'hsa-miR-513c-5p', 'hsa-miR-744-5p', 'hsa-miR-488-3p', 'hsa-miR-1251-3p', 'hsa-miR-877-5p', 'hsa-miR-148a-5p', \
'hsa-miR-129-1-3p', 'hsa-miR-192-3p', 'hsa-miR-574-3p', 'hsa-let-7i-5p', 'hsa-miR-520g-3p', 'hsa-miR-516b-5p', 'hsa-miR-452-5p', 'hsa-miR-409-3p', 'hsa-miR-519d-3p', \
'hsa-miR-520d-5p', 'hsa-miR-520d-3p', 'hsa-miR-3922-3p', 'hsa-miR-130b-3p', 'hsa-miR-30e-5p', 'hsa-miR-199a-3p', 'hsa-miR-199a-5p', 'hsa-miR-1301-3p',\
 'hsa-miR-1323', 'hsa-miR-1283', 'hsa-miR-487a-5p', 'hsa-miR-329-5p', 'hsa-miR-153-5p', 'hsa-miR-370-5p', 'hsa-miR-664b-3p', 'hsa-miR-181b-3p', 'hsa-miR-382-3p', \
'hsa-miR-429', 'hsa-miR-1292-5p', 'hsa-miR-664a-3p', 'hsa-miR-4758-3p', 'hsa-miR-200a-5p', 'hsa-miR-433-3p', 'hsa-miR-329-3p', 'hsa-miR-323a-3p', 'hsa-miR-151a-3p',\
 'hsa-miR-340-3p', 'hsa-miR-328-3p', 'hsa-miR-342-3p', 'hsa-miR-135b-5p', 'hsa-miR-148b-3p', 'hsa-miR-889-3p', 'hsa-miR-541-3p', 'hsa-miR-708-3p', 'hsa-miR-190b-5p',\
 'hsa-miR-590-5p', 'hsa-miR-3614-5p', 'hsa-miR-3613-5p', 'hsa-miR-1307-3p', 'hsa-miR-1293', 'hsa-miR-191-3p', 'hsa-miR-423-3p', 'hsa-miR-424-5p', 'hsa-miR-425-3p',\
 'hsa-miR-874-3p', 'hsa-miR-29b-3p', 'hsa-miR-106a-5p', 'hsa-miR-122-3p', 'hsa-miR-124-5p', 'hsa-miR-135a-3p', 'hsa-miR-141-5p', 'hsa-miR-125b-2-3p', 'hsa-miR-371a-5p',\
 'hsa-miR-361-3p', 'hsa-miR-130b-5p', 'hsa-miR-374a-3p', 'hsa-miR-6499-5p', 'hsa-miR-584-5p', 'hsa-miR-4772-3p', 'hsa-miR-514a-3p', 'hsa-miR-508-3p', 'hsa-miR-509-3p',\
 'hsa-miR-203b-3p', 'hsa-miR-4709-3p', 'hsa-miR-345-5p', 'hsa-miR-346', 'hsa-miR-133b', 'hsa-miR-450b-5p', 'hsa-miR-758-3p', 'hsa-miR-330-5p', 'hsa-miR-342-5p', \
'hsa-miR-151a-5p', 'hsa-miR-135b-3p', 'hsa-miR-3127-5p', 'hsa-miR-99b-3p', 'hsa-miR-219a-2-3p', 'hsa-miR-29c-5p', 'hsa-miR-106b-3p', 'hsa-miR-194-3p', 'hsa-miR-6511b-3p',\
 'hsa-miR-30b-3p', 'hsa-let-7i-3p', 'hsa-miR-15b-3p', 'hsa-miR-629-3p', 'hsa-miR-196b-5p']

#name the result with miRNAs
y_test_out.columns=miRNAs
print('>>>>>>> The result is:', y_test_out.shape)
print(y_test_out.head())
y_test_out.index=x_data_t.index #change this with your sample name
print(y_test_out)
print('>>>>>>> Saved the result to csv file:', f"{expr}.csv !!!")
y_test_out.to_csv(f"{expr}.csv")
