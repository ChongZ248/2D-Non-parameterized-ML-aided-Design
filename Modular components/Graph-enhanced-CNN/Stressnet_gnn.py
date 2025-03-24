import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
# torch.set_default_dtype(torch.float16)

##### code for Stress net
class SEblock(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.lin1=nn.Linear(in_ch,in_ch//4)
        self.relu1  = nn.LeakyReLU()
        self.lin2=nn.Linear(in_ch//4,in_ch)
        self.sig2  = nn.LeakyReLU()

    def forward(self, x):
        xold=x
        avged=self.avgpool(x)
        viewed=avged.view(avged.shape[0:-2])
        red=self.relu1(self.lin1(viewed))
        sed=self.sig2(self.lin2(red))
        # print(avged.shape[0:-2])
        backtoshape=sed.view((xold.shape[0],xold.shape[1],1,1))
        return xold*backtoshape


class Resdblock(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3,padding='same')
        self.relu1  = nn.LeakyReLU()
        self.bn1=nn.BatchNorm2d(num_features=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,padding='same')
        self.bn2=nn.BatchNorm2d(num_features=out_ch)
        self.relu2  = nn.LeakyReLU()
        self.SEb= SEblock(in_ch=out_ch)

    def forward(self, x):
        xold=x
        coved1=self.bn1(self.relu1(self.conv1(x)))
        coved2=self.bn2(self.relu2(self.conv2(coved1)))
        seed=self.SEb(coved2)
        resed=seed+xold
        return resed
    
class Cblock(nn.Module): 
    def __init__(self, in_ch,out_ch,ksize=3,st=2,pad=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, ksize,padding=pad,stride=st)
        self.relu1  = nn.LeakyReLU()
        self.bn1=nn.BatchNorm2d(num_features=out_ch)

    def forward(self, x):
        coved1=self.relu1(self.bn1(self.conv1(x)))
        return coved1

class CTblock(nn.Module): 
    def __init__(self, in_ch,out_ch,ksize=3,st=2,pad=1,output_pad=0):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, ksize,padding=pad,stride=st,output_padding=output_pad)
        self.relu1  = nn.LeakyReLU()
        self.bn1=nn.BatchNorm2d(num_features=out_ch)

    def forward(self, x):
        coved1=self.bn1(self.relu1(self.conv1(x)))
        # coved1=self.relu1(self.conv1(x))
        return coved1


class CTblocklast(nn.Module): 
    def __init__(self, in_ch,out_ch,ksize=3,st=2,pad=1,output_pad=0):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, ksize,padding=pad,stride=st,output_padding=output_pad)
        self.relu1  = nn.LeakyReLU()

    def forward(self, x):
        coved1=self.relu1(self.conv1(x))
        return coved1

class StressNetori(nn.Module):
    def __init__(self,chlist=[8,32,64,128,64,32,9]):
        super().__init__()
        self.c1=Cblock(chlist[0],chlist[1],ksize=10,pad=4)
        self.c2=Cblock(chlist[1],chlist[2],ksize=3,pad=1,st=2)
        self.c3=Cblock(chlist[2],chlist[3],ksize=3,pad=1,st=2)
        self.RN_blocks = nn.ModuleList([Resdblock(128) for i in range(5)])
        self.c4=CTblock(chlist[3],chlist[4],ksize=3,pad=1,st=2,output_pad=1)
        self.c5=CTblock(chlist[4],chlist[5],ksize=3,pad=1,st=2,output_pad=1)
        self.c6=CTblocklast(chlist[5],chlist[6],ksize=10,pad=4)  

    def forward(self, x):
        ftrs=[]
        x1=self.c1(x)
        x2=self.c2(x1)
        x3=self.c3(x2)
        xin=x3
        for block in self.RN_blocks:

                xin = block(xin)
                # print(x.shape)
                ftrs.append(xin)
        x4=self.c4(xin)
        x5=self.c5(x4)
        x6=self.c6(x5)
        return x6

   
class StressNetsimp(nn.Module):
    def __init__(self,chlist=[8,32,64,128,64,32,9]):
        super().__init__()
        self.c1=Cblock(chlist[0],chlist[1],ksize=3,pad=1,st=1)
        self.c2=Cblock(chlist[1],chlist[2],ksize=3,pad=1,st=1)
        self.c3=Cblock(chlist[2],chlist[3],ksize=3,pad=1,st=1)
        self.RN_blocks = nn.ModuleList([Resdblock(128) for i in range(5)])
        self.c4=CTblock(chlist[3],chlist[4],ksize=3,pad=1,st=1)
        self.c5=CTblock(chlist[4],chlist[5],ksize=3,pad=1,st=1)
        self.c6=CTblocklast(chlist[5],chlist[6],ksize=3,pad=1,st=1)  

    def forward(self, x):
        # ftrs=[]
        x1=self.c1(x)
        x2=self.c2(x1)
        x3=self.c3(x2)
        xin=x3
        # for block in self.RN_blocks:

        #         xin = block(xin)
        #         # print(x.shape)
        #         # ftrs.append(xin)
        x4=self.c4(xin)
        x5=self.c5(x4)
        x6=self.c6(x5)
        return x6


class StressNetgit(nn.Module):
    def __init__(self,chlist=[8,32,64,128,64,32,9]):
        super().__init__()
        self.c1=Cblock(chlist[0],chlist[1],ksize=9,pad='same',st=1)
        self.c2=Cblock(chlist[1],chlist[2],ksize=3,pad=1,st=2)
        self.c3=Cblock(chlist[2],chlist[3],ksize=3,pad=1,st=2)
        self.RN_blocks = nn.ModuleList([Resdblock(128) for i in range(5)])
        self.c4=CTblock(chlist[3],chlist[4],ksize=3,pad=1,st=2,output_pad=1)
        self.c5=CTblock(chlist[4],chlist[5],ksize=3,pad=1,st=2,output_pad=1)
        self.c6=CTblocklast(chlist[5],chlist[6],ksize=5,pad=2,st=1)  

    def forward(self, x):
        # ftrs=[]
        x1=self.c1(x)
        x2=self.c2(x1)
        x3=self.c3(x2)
        xin=x3
        # print(x3.shape)
        for block in self.RN_blocks:

                xin = block(xin)
                # print(x.shape)
                # ftrs.append(xin)
        x4=self.c4(xin)
        # print(x4.shape)
        x5=self.c5(x4)
        # print(x5.shape)
        x6=self.c6(x5)
        # print(x6.shape)
        return x6
    
tim=1
class StressNetgit4shrink(nn.Module):
    def __init__(self,chlist=[6,32,64,128,256,256,128,128,64,32,1]):
        super().__init__()
        self.c1=Cblock(chlist[0],chlist[1],ksize=9,pad='same',st=1)
        self.c2=Cblock(chlist[1],chlist[2],ksize=3,pad=1,st=2)
        self.c3=Cblock(chlist[2],chlist[3],ksize=3,pad=1,st=2)
        self.cadd1=Cblock(chlist[3],chlist[4],ksize=3,pad=1,st=2)
        # self.cadd3=Cblock(chlist[4],chlist[5],ksize=3,pad=1,st=2)
        self.RN_blocks = nn.ModuleList([Resdblock(256) for i in range(5)])
        self.cadd2=CTblock(chlist[5],chlist[6],ksize=3,pad=1,st=2,output_pad=1)
        # self.cadd4=CTblock(chlist[6],chlist[7],ksize=3,pad=1,st=2,output_pad=1)
        self.c4=CTblock(chlist[7],chlist[8],ksize=3,pad=1,st=2,output_pad=1)
        self.c5=CTblock(chlist[8],chlist[9],ksize=3,pad=1,st=2,output_pad=1)
        self.c6=CTblocklast(chlist[9],chlist[10],ksize=9,pad=4,st=1)  

    def forward(self, x):
        # ftrs=[]
        x1=self.c1(x)
        x2=self.c2(x1)
        x3=self.c3(x2)
        xad=self.cadd1(x3)
        # xad=self.cadd3(xad)
        xin=xad
        # print(x3.shape)
        for block in self.RN_blocks:

                xin = block(xin)
                # print(x.shape)
                # ftrs.append(xin)
        xad=self.cadd2(xin)
        # xad=self.cadd4(xad)
        x4=self.c4(xad)
        # print(x4.shape)
        x5=self.c5(x4)
        # print(x5.shape)
        x6=self.c6(x5)
        # print(x6.shape)
        return x6
    

####### for ###https://doi.org/10.1016/j.mechmat.2021.104191
k=16


# class SEdenseblock(nn.Module):
#     def __init__(self, in_ch=1):
#         super().__init__()
#         self.avgpool=nn.AdaptiveAvgPool2d((1,1))
#         self.lin1=nn.Linear(in_ch,in_ch//16)
#         self.relu1  = nn.LeakyReLU()
#         self.lin2=nn.Linear(in_ch//16,in_ch)
#         self.sig2  = nn.LeakyReLU()

#     def forward(self, x):
#         xold=x
#         avged=self.avgpool(x)
#         viewed=avged.view(avged.shape[0:-2])
#         red=self.relu1(self.lin1(viewed))
#         sed=self.sig2(self.lin2(red))
#         # print(avged.shape[0:-2])
#         backtoshape=sed.view((xold.shape[0],xold.shape[1],1,1))
#         return xold*backtoshape



class SE_Resdblock(nn.Module):
    def __init__(self,in_ch=k, out_ch=k):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,padding=1)
        self.relu1  = nn.LeakyReLU()
        self.bn1=nn.BatchNorm2d(num_features=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,padding=1)
        # self.bn2=nn.BatchNorm2d(num_features=out_ch)
        # self.relu2  = nn.LeakyReLU()
        self.SEb= SEblock(in_ch=out_ch)

    def forward(self, x):
        xold=x
        coved1=self.relu1(self.bn1(self.conv1(x)))
        coved2=self.conv2(coved1)
        seed=self.SEb(coved2)
        resed=seed+xold
        return resed


class Block1(nn.Module):
    def __init__(self,chin=1,chout=k,ksize=3,pad=1,st=1):
        super().__init__()
        self.c1=Cblock(chin,chout,ksize=ksize,pad=pad,st=st)
        self.se1=SE_Resdblock(in_ch=chout, out_ch=chout)
    def forward(self, x):
        # ftrs=[]
        x1=self.c1(x)
        x2=self.se1(x1)
        # print(x2.shape)
        return x2

class Block2(nn.Module):
    def __init__(self,chin=k,chout=k,ksize=3,pad=1,st=2):
        super().__init__()
        self.c1=Cblock(chin,chout*2,ksize=ksize,pad=pad,st=st)
        self.se1=SE_Resdblock(in_ch=chout*2, out_ch=2*chout)
    def forward(self, x):
        # ftrs=[]
        x1=self.c1(x)
        x2=self.se1(x1)
        # print(x2.shape)
        return x2

class Block3(nn.Module):
    def __init__(self,chin=2*k,chout=2*k,ksize=3,pad=1,st=2):
        super().__init__()
        self.c1=Cblock(chin,chout*2,ksize=ksize,pad=pad,st=st)
        self.se1=SE_Resdblock(in_ch=chout*2, out_ch=2*chout)
        self.ct2=CTblock(2*chout,2*chout,ksize=3,st=2,pad=1,output_pad=1)
    def forward(self, x):
        # ftrs=[]
        x1=self.c1(x)
        x2=self.se1(x1)
        x3=self.ct2(x2)
        # print(x3.shape)
        return x3

class Block4(nn.Module):
    def __init__(self,chin=4*k,chout=2*k):
        super().__init__()
        self.se1=SE_Resdblock(in_ch=chin, out_ch=chin)
        self.ct1=CTblock(chin,chout,ksize=3,st=2,pad=1,output_pad=1)
    def forward(self, x):
        # ftrs=[]
        x1=self.se1(x)
        x2=self.ct1(x1)
        # print(x2.shape)
        return x2

class Block5(nn.Module):
    def __init__(self,chin=2*k,chout=1*k,finalout=8):
        super().__init__()
        self.se1=SE_Resdblock(in_ch=chin, out_ch=chin)
        self.ct1=nn.Conv2d(chin, finalout, 1,padding=0)
        self.lin1=nn.Linear(finalout,finalout)
    def forward(self, x):
        # ftrs=[]
        x1=self.se1(x)
        x2=self.ct1(x1)
        # print(x2.shape)
        x2=self.lin1(torch.transpose(x2,1,-1))
        # print(x2.shape)
        return torch.transpose(x2,1,-1)

class Modfiedunet3shrink(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[5,32,64,128,128,128,128,128,64,32,1]):
        super().__init__()
        self.c1=Block1(chin=5,chout=k)
        self.c2=Block2(chin=k,chout=k)
        self.c22=Block2(chin=2*k,chout=2*k)
        self.c3=Block3(chin=4*k,chout=4*k)
        self.c42=Block4(chin=12*k,chout=4*k)
        self.c4=Block4(chin=6*k,chout=2*k)
        self.c5=Block5(chin=3*k,chout=1*k,finalout=1)
        self.pad=PaddingBlock()
        self.cut=CuttingBlock()

    def forward(self, x):
        x=self.pad(x)
        x1=self.c1(x)
        xor=x1
        x2=self.c2(x1)
        x22=self.c22(x2)
        x3=self.c3(x22)
        x42=self.c42(torch.cat((x22,x3),1))
        x4=self.c4(torch.cat((x2,x42),1))
        x5=self.c5(torch.cat((x1,x4),1))
        x5=self.cut(x5)
        return x5
    

class Modfiedunet4shrink(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Block1(chin=6,chout=k)
        self.c2=Block2(chin=k,chout=k)
        self.c22=Block2(chin=2*k,chout=2*k)
        self.c23=Block2(chin=4*k,chout=4*k)
        self.c3=Block3(chin=8*k,chout=8*k)
        self.c43=Block4(chin=24*k,chout=8*k)
        self.c42=Block4(chin=12*k,chout=6*k)
        self.c4=Block4(chin=8*k,chout=4*k)
        self.c5=Block5(chin=5*k,chout=1*k,finalout=1)
        self.pad=PaddingBlock()
        self.cut=CuttingBlock()

    def forward(self, x):
        x=self.pad(x)
        x1=self.c1(x)
        x2=self.c2(x1)
        x22=self.c22(x2)
        x23=self.c23(x22)
        x3=self.c3(x23)
        x43=self.c43(torch.cat((x23,x3),1))
        x42=self.c42(torch.cat((x22,x43),1))
        x4=self.c4(torch.cat((x2,x42),1))
        x5=self.c5(torch.cat((x1,x4),1))
        x5=self.cut(x5)
        return x5

class Modfiedunet(nn.Module): ###https://doi.org/10.1016/j.mechmat.2021.104191
    def __init__(self,chlist=[8,32,64,128,128,128,128,128,64,32,9]):
        super().__init__()
        self.c1=Block1(chin=5,chout=k)
        self.c2=Block2(chin=k,chout=k)
        self.c3=Block3(chin=2*k,chout=2*k)
        self.c4=Block4(chin=6*k,chout=2*k)
        self.c5=Block5(chin=3*k,chout=1*k,finalout=1)
        self.pad=PaddingBlock()
        self.cut=CuttingBlock()
    def forward(self, x):
        x=self.pad(x)
        x1=self.c1(x)
        x2=self.c2(x1)
        x3=self.c3(x2)
        x4=self.c4(torch.cat((x2,x3),1))
        x5=self.c5(torch.cat((x1,x4),1))
        x5=self.cut(x5)
        return x5
    
class PaddingBlock(nn.Module):
    def __init__(self,chin=2*k,chout=1*k,finalout=8):
        super().__init__()

    def forward(self, x): ##xshape batch,chan,h,w
        xup=torch.flip(x[:,:,0:16,:], [2])
        xdown=torch.flip(x[:,:,-16:,:], [2])
        xupdown=torch.cat((xup,x,xdown),dim=2)
        xleft=torch.flip(xupdown[:,:,:,0:16], [3])
        xright=torch.flip(xupdown[:,:,:,-16:], [3])
        xall=torch.cat((xleft,xupdown,xright),dim=3)
        return xall
    
class CuttingBlock(nn.Module):
    def __init__(self,chin=2*k,chout=1*k,finalout=8):
        super().__init__()

    def forward(self, x): ##xshape batch,chan,h,w
        x=x[:,:,16:-16,16:-16]
        return x