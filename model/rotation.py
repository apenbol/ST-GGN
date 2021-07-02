import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ms_tcn import MultiScale_TemporalConv
class Rotation_gcn(nn.Module):
    def __init__(self,numnodes,in_channel,out_channel,num_heads=13):
        super(Rotation_gcn, self).__init__()
        self.rotation_Convs = nn.ModuleList(
            [nn.Conv2d(numnodes,numnodes,(1,1)) for i in range(num_heads)]
        )
        self.conv = nn.Conv2d(in_channel,out_channel,(1,1))
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        """

        Args:
            x: N,C,T,V

        Returns:
            x: N,C,T,V
        """

        out = x.permute(0,3,1,2).contiguous()
        out = sum([i(out) for i in self.rotation_Convs])# [N,V,C,T;...]->N,V,C*num_heads,T
        out = out.permute(0,2,3,1).contiguous()
        out = x + out
        out = self.bn(self.conv(out))
        return F.relu(out)
def test_roat():
    x = torch.randn(64,3,300,25)
    model = Rotation_gcn(25,3,13)
    x = model(x)
    print(x.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn = nn.BatchNorm1d(2 * 3 * 25)
        self.spatios = nn.ModuleList( ( Rotation_gcn(25,3,96,13),
                        Rotation_gcn(25,192,192,13),
                        Rotation_gcn(25,384,384,13)))


        self.times =  nn.ModuleList((
                                nn.Sequential(
                                    MultiScale_TemporalConv(96,96),
                                    MultiScale_TemporalConv(96,96),
                                    MultiScale_TemporalConv(96,192,stride=2)
                                ),
                                nn.Sequential(
                                    MultiScale_TemporalConv(192,192),
                                    MultiScale_TemporalConv(192,192),
                                    MultiScale_TemporalConv(192,384,stride=2)
                                ),
                                nn.Sequential(
                                    MultiScale_TemporalConv(384,384),
                                    MultiScale_TemporalConv(384,384),
                                    MultiScale_TemporalConv(384,384)
                                )
        ))





        self.avg = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm2d(384)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(384,60)
        self.soft = nn.Softmax(-1)

    def forward(self, x):
        """

        Args:
            x: N, C, T, V, M

        Returns:
            x: N,384
        """
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        for i,j in zip(self.spatios,self.times):
            x = i(x)
            x = j(x)


        # x = self.avg(x)
        # x = self.norm(x)
        # x = self.flat(x)
        # x = self.soft(self.fc(x))
        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out      #todo 参数量不太对 1.2M 原文是0.8M



def test_roat():
    x = torch.randn(2,3,300,25,2)
    model = Model()
    x = model(x)
    print(x.shape)


    model = Model()
    x = torch.randn(1,3,48,25,2)
    from thop import profile
    model.eval()
    macs, params = profile(model, inputs=(x,))#警告如果在容器上 并不影响，还是会包含底层操作的
    print(macs,params)#6.689G,1.208M acc 86.75(origin data) 81.1(after process)


