import torch
from torch_geometric.nn import GraphSAGE
from Meshpoolscr import MeshPooling
# def cnntogra(pos,batch,datatomap):
#     imageindex=batch
#     posindexT=torch.round(pos/20*(datatomap.shape[-1]-1)).long()
#     posindex=torch.stack((posindexT[:,1],posindexT[:,0]),dim=-1)
#     # Select the right "image" using imageindex
#     selected_images = datatomap[imageindex]
#     # Now use posindex to index into the last two dimensions
#     result = selected_images[torch.arange(selected_images.shape[0]), :, posindex[:, 0], posindex[:, 1]]
#     return result

def cnntogra(pos, batch, datatomap):
    unique_indices, inverse_indices = torch.unique(batch, return_inverse=True)
    posindexT = torch.round(pos / 20 * (datatomap.shape[-1] - 1)).long()
    posindex = torch.stack((posindexT[:, 1], posindexT[:, 0]), dim=-1)

    selected_images = datatomap[unique_indices]
    result = selected_images[inverse_indices, :, posindex[:, 0], posindex[:, 1]]
    return result


def gratocnn(resultfromgnn,pos,batch,oldcnn):
    imageindex=batch
    # Example tensors for demonstration
    batchedimage =  torch.ones(oldcnn.shape[0], resultfromgnn.shape[-1], oldcnn.shape[2], oldcnn.shape[3]).to(device)*-10 # This can be of different sizes now
    pixelvalue = resultfromgnn  # Random values
    imageindex = imageindex.unsqueeze(dim=-1)  # Adjusted to batch size
    posindexT=torch.round(pos/20*(oldcnn.shape[-1]-1)).long()
    posindex=torch.stack((posindexT[:,1],posindexT[:,0]),dim=-1)
    # Flatten the batchedimage tensor
    batchedimage_flat = batchedimage.view(batchedimage.size(0), batchedimage.size(1), -1)
    # Get image dimensions
    img_height, img_width = batchedimage.size(2), batchedimage.size(3)
    # Calculate the linear indices
    linear_indices = posindex[:, 0] * img_width + posindex[:, 1]
    # Use advanced indexin to assign values
    batchedimage_flat[imageindex[:, 0], :, linear_indices] = pixelvalue
    # Reshape back to original shape
    batchedimage = batchedimage_flat.view(batchedimage.size(0), batchedimage.size(1), img_height, img_width)
    return batchedimage

def cnntogra_pad(pos_nopad,batch,datatomap):
    pos=pos_nopad+torch.tensor([20/8,20/8])
    imageindex=batch
    posindexT=torch.round(pos/(20*5/4)*(datatomap.shape[-1]-1)).long()
    posindex=torch.stack((posindexT[:,1],posindexT[:,0]),dim=-1)
    # Select the right "image" using imageindex
    selected_images = datatomap[imageindex]
    # Now use posindex to index into the last two dimensions
    result = selected_images[torch.arange(selected_images.shape[0]), :, posindex[:, 0], posindex[:, 1]]
    return result
def gratocnn_pad(resultfromgnn,pos_nopad,batch,oldcnn):
    pos=pos_nopad+torch.tensor([20/8,20/8])
    imageindex=batch
    # Example tensors for demonstration
    batchedimage =  torch.ones(oldcnn.shape[0], resultfromgnn.shape[-1], oldcnn.shape[2], oldcnn.shape[3])*-10 # This can be of different sizes now
    pixelvalue = resultfromgnn  # Random values
    imageindex = imageindex.unsqueeze(dim=-1)  # Adjusted to batch size
    posindexT=torch.round(pos/(20*5/4)*(oldcnn.shape[-1]-1)).long()
    posindex=torch.stack((posindexT[:,1],posindexT[:,0]),dim=-1)
    # Flatten the batchedimage tensor
    batchedimage_flat = batchedimage.view(batchedimage.size(0), batchedimage.size(1), -1)
    # Get image dimensions
    img_height, img_width = batchedimage.size(2), batchedimage.size(3)
    # Calculate the linear indices
    linear_indices = posindex[:, 0] * img_width + posindex[:, 1]
    # Use advanced indexin to assign values
    batchedimage_flat[imageindex[:, 0], :, linear_indices] = pixelvalue
    # Reshape back to original shape
    batchedimage = batchedimage_flat.view(batchedimage.size(0), batchedimage.size(1), img_height, img_width)
    return batchedimage

import matplotlib.pyplot as plt
import numpy as np
def plotedges(pos,edge_index,figindex):
    fig=plt.figure()
    posss=np.array(pos.cpu().detach())
    print(pos.shape)
    print(edge_index.shape)
    d = dict(enumerate(posss, 0))
    edgess=np.array(edge_index.cpu().detach()).T
    for i in edgess:
        plt.plot([d[i[0]][0],d[i[1]][0]],[d[i[0]][1],d[i[1]][1]],linewidth=0.1,c='b',alpha=0.5)
        # print(d[i[0]],d[i[1]])
    plt.axis('equal')
    plt.savefig(str(figindex)+'boundpoo.svg')
    plt.close()


from Stressnet import Modfiedunet4shrink, Modfiedunet3shrink,StressNetori,StressNetgit,StressNetgit4shrink,Modfiedunet
from torch_geometric.nn import GraphSAGE
import torch.nn as nn


class CNNadgnn(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Modfiedunet4shrink()
        self.g1=GraphSAGE(in_channels=int(8),hidden_channels=int(128), 
                                num_layers=4, out_channels=1)
    def forward(self, igra):
        x=igra.xdata128.float()
        gx,edge_index,posnew,pollinfor,batchinfo=igra.x,igra.edge_index,igra.pos.float(),igra.pollinfor,igra.batch
        gx=torch.cat((igra.x[:,0:2],igra.x[:,-3:],posnew),dim=-1).float()
        x1=self.c1(x)
        cnng=cnntogra(posnew,batchinfo,x1)
        # print(gx.shape,cnng.shape)
        gout=self.g1(torch.cat((gx,cnng),dim=-1),edge_index)
        return x1,gout
    
class CNNadgnn_cnn1ch(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Modfiedunet4shrink()
        self.g1=GraphSAGE(in_channels=int(8),hidden_channels=int(128), 
                                num_layers=4, out_channels=1)
    def forward(self, igra):
        x=igra.xdata128.float()
        first_channel = x[:, 0:1, :, :]
        x= first_channel.repeat(1, 6, 1, 1)
        gx,edge_index,posnew,pollinfor,batchinfo=igra.x,igra.edge_index,igra.pos.float(),igra.pollinfor,igra.batch
        gx=torch.cat((igra.x[:,0:2],igra.x[:,-3:],posnew),dim=-1).float()
        x1=self.c1(x)
        cnng=cnntogra(posnew,batchinfo,x1)
        # print(gx.shape,cnng.shape)
        gout=self.g1(torch.cat((gx,cnng),dim=-1),edge_index)
        return x1,gout


device='cuda'
class CNNadgnn_cnn1chv2(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Modfiedunet4shrink(chin=1)
        self.g1=GraphSAGE(in_channels=int(8),hidden_channels=int(128), 
                                num_layers=4, out_channels=1)
    def forward(self, igra):
        x=igra.xdata128.float()
        first_channel = x[:, 0:1, :, :]
        x= first_channel
        gx,edge_index,posnew,pollinfor,batchinfo=igra.x,igra.edge_index,igra.pos.float(),igra.pollinfor,igra.batch
        gx=torch.cat((igra.x[:,0:2],igra.x[:,-3:],posnew),dim=-1).float()
        x1=self.c1(x)
        # print(x1)
        cnng=cnntogra(posnew,batchinfo,x1)
        # print(gx.shape,cnng.shape)
        gout=self.g1(torch.cat((gx,cnng),dim=-1),edge_index)
        return x1,gout
 
class CNNadgnnIdent(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Modfiedunet4shrink()
        # self.g1=GraphSAGE(in_channels=int(8),hidden_channels=int(128), 
        #                         num_layers=4, out_channels=1)
    def forward(self, igra):
        x=igra.xdata128.float()
        gx,edge_index,posnew,pollinfor,batchinfo=igra.x,igra.edge_index,igra.pos.float(),igra.pollinfor,igra.batch
        gx=torch.cat((igra.x[:,0:2],igra.x[:,-3:],posnew),dim=-1).float()
        x1=self.c1(x)
        cnng=cnntogra(posnew,batchinfo,x1)
        # print(gx.shape,cnng.shape)
        # gout=self.g1(torch.cat((gx,cnng),dim=-1),edge_index)
        return x1,cnng


class FlexibleCNN(nn.Module):
    def __init__(self, num_layers=1, in_channels=1, out_channels=1, mid_channels=16, activation_fn=nn.ReLU):
        super(FlexibleCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation_fn()
        if num_layers > 1:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
                self.activation
            ))
        else:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                self.activation
            ))
        for _ in range(1, num_layers-1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1),
                self.activation
            ))
        if num_layers > 1:
            self.layers.append(nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1),
                nn.Identity()
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class CNNadgnnIdent_extrCNN(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Modfiedunet4shrink()
        self.cextra=FlexibleCNN(num_layers=4, mid_channels=64)
        # self.g1=GraphSAGE(in_channels=int(8),hidden_channels=int(128), 
        #                         num_layers=4, out_channels=1)
    def forward(self, igra):
        x=igra.xdata128.float()
        gx,edge_index,posnew,pollinfor,batchinfo=igra.x,igra.edge_index,igra.pos.float(),igra.pollinfor,igra.batch
        gx=torch.cat((igra.x[:,0:2],igra.x[:,-3:],posnew),dim=-1).float()
        x1=self.c1(x)
        x1=self.cextra(x1)
        cnng=cnntogra(posnew,batchinfo,x1)
        # print(gx.shape,cnng.shape)
        # gout=self.g1(torch.cat((gx,cnng),dim=-1),edge_index)
        return x1,cnng
    


def getcnnloss(Xtrain,predicted,Ytrain,highvalue):
    mask=Xtrain[:,0,:,:]==1
    predictedT=torch.transpose(predicted,1,3)
    predictedT=torch.transpose(predictedT,2,1)
    YtrainT=torch.transpose(Ytrain,1,3)
    YtrainT=torch.transpose(YtrainT,2,1)
    midpre=predictedT[mask]
    YtrainTmask=YtrainT[mask]
    errors = torch.abs(midpre-YtrainTmask)
    lossdif = torch.mean(errors)

    #####get error for highstress area
    highmask=YtrainTmask>=highvalue
    YtrainThighmask=YtrainTmask[highmask]
    midprehighmask=midpre[highmask]
    losshighstress=torch.mean(torch.abs(YtrainThighmask-midprehighmask))


    return losshighstress+lossdif, losshighstress





