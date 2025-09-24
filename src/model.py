import torch
import torch.nn as nn
import torch.nn.functional as F
class CompressedSensingNet(nn.Module):
    def __init__(self):
        super(CompressedSensingNet, self).__init__()
        
        self.convSR1 = nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1)
        self.BNSR1 = nn.BatchNorm2d(3)
        self.convSR2 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1) 
        self.BNSR2 = nn.BatchNorm2d(64)        
        self.convSR3 = nn.Conv2d(128,192,kernel_size=3,stride=2,padding=1)
        self.BNSR3 = nn.BatchNorm2d(128)
        
        self.convC = nn.Conv2d(192, 6144,kernel_size=8,stride=8,padding=0,bias=False)
        
        self.convRX = nn.Conv2d(6144,12288,kernel_size=1,stride=1,bias=False)
        
        self.convRR = nn.Conv2d(3, 64,kernel_size=7,stride=1,padding=3,bias=False)
        self.BNRR = nn.BatchNorm2d(64)
        
        self.convR1 = nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1)
        self.BNCR1 = nn.BatchNorm2d(64)
        self.convR2 = nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1)
        self.BNCR2 = nn.BatchNorm2d(64)
        self.convR3 = nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1) 
        self.BNCR3 = nn.BatchNorm2d(64)
        self.convR4 = nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1)
        self.BNCR4 = nn.BatchNorm2d(64)
        
        self.up1 = nn.Conv2d(64, 128,kernel_size=1,stride=1)
        
        self.convR5 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1)
        self.BNCR5 = nn.BatchNorm2d(128)
        self.convR6 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1)
        self.BNCR6 = nn.BatchNorm2d(128)
        self.convR7 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1)
        self.BNCR7 = nn.BatchNorm2d(128)
        self.convR8 = nn.Conv2d(128, 128,kernel_size=3,stride=1,padding=1)
        self.BNCR8 = nn.BatchNorm2d(128)
        
        self.up2 = nn.Conv2d(128, 256,kernel_size=1,stride=1)
        
        self.convR9 = nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1)
        self.BNCR9 = nn.BatchNorm2d(256)
        self.convR10 = nn.Conv2d(256, 256,kernel_size=3,stride=1,padding=1)
        self.BNCR10 = nn.BatchNorm2d(256)

        self.convx =  nn.Conv2d(256, 3,kernel_size=3,stride=1,padding=1)

    def forward(self, x):

        outSR = F.relu(self.BNSR1(x))
        outSR = self.convSR1(outSR)
        outSR = F.relu(self.BNSR2(outSR))
        outSR = self.convSR2(outSR)
        outSR = F.relu(self.BNSR3(outSR))
        outSR = self.convSR3(outSR)
        
        outSR = outSR.view(-1,3,64,64)+x
        outSR = outSR.view(-1,192,8,8)
        
        
        outC = self.convC(outSR)
        
        outR = self.convRX(outC)
        
        outProX_Z=outR.view(-1,192,8,8)
        outProX_Z=self.convC(outProX_Z)
        
        
        outProX = outR.view(-1,3,64,64) 
        

        
       
        outRR = self.convRR(outProX)
        outRR = F.relu(self.BNRR(outRR))       
        

        outNR = F.relu(self.BNCR1(outRR))
        outNR = self.convR1(outNR)
        outNR = F.relu(self.BNCR2(outNR))
        outNR = self.convR2(outNR)
        outNR = outNR+outRR
        out_1 = outNR
        
        outNR = F.relu(self.BNCR3(outNR))
        outNR = self.convR3(outNR)
        outNR = F.relu(self.BNCR4(outNR))
        outNR = self.convR4(outNR)
        outNR = outNR+out_1

        outNR = self.up1(outNR)
        
        out_2 = outNR  
        outNR = F.relu(self.BNCR5(outNR))
        outNR = self.convR5(outNR)
        outNR = F.relu(self.BNCR6(outNR))
        outNR = self.convR6(outNR)
        outNR = outNR+out_2
        
        out_3 = outNR
        outNR = F.relu(self.BNCR7(outNR))
        outNR = self.convR7(outNR)
        outNR = F.relu(self.BNCR8(outNR))
        outNR = self.convR8(outNR)
        outNR = outNR+out_3
        
        outNR = self.up2(outNR)
        
        out_4 = outNR
        outNR = F.relu(self.BNCR9(outNR))
        outNR = self.convR9(outNR)
        outNR = F.relu(self.BNCR10(outNR))
        outNR = self.convR10(outNR)
        outNR = outNR+out_4
        
        outX = self.convx(outNR)
        out = outX.view(-1,3,64,64) 
        out = out+outProX
        return out ,outProX_Z, outProX , outC,outSR