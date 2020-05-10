# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

import torch
import numpy
from timeit import timeit
from time import time
class Model(torch.nn.Module):
    def __init__(self,cuda=True):
        super(Model,self).__init__()  
        input_dim=4
        output_dim=2
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.layer1=torch.nn.Linear(input_dim,input_dim*output_dim).to(self.device)
        self.layer2=torch.nn.Linear(input_dim*output_dim,output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss=100
        self.softplus=torch.nn.Softplus()
    
    def __put_on_device(self,x):
        if isinstance(x,torch.Tensor) and x.device is not self.device:
            x=x.to(self.device)
        else:
            x=torch.tensor(x,dtype=torch.float,device=self.device) 
        return x
        
    def forward(self,x):
        x=self.__put_on_device(x)
        y=self.layer1(x)
        y=self.layer2(torch.relu(y))
        return y
    
    def on_step(self,x):
        y=self(x)
        m=y[0]
        std=self.softplus(y[1])
        return [torch.normal(m,std).tolist()],m.item(),std.item()

    def train(self,inputs,targets,target_loss=0.1,timeout=60):
        t0=time()
        to=t0+timeout
        inputs=self.__put_on_device(inputs)
        targets=self.__put_on_device(targets)
        self.loss=100
        i=0
        while time()<to and self.loss>target_loss:
            self.optimizer.zero_grad()
            y = self(inputs)
            self.loss=((targets-y[:,0]).pow(2)+self.softplus(y[:,1]).pow(2)).mean()
            self.loss.backward()
            self.optimizer.step()
            i+=1
        return "train "+str(i)+" times, lasts:"+str(time()-t0)+"s, loss:"+str(self.loss.item())
    
class PID:
    def __init__(self,k0,k1=0,k2=0):
        self.Kp=k0
        self.Ki=k1
        self.Kd=k2
        self.err=0
        self.I=0
        
    def step(self,error,dt):
        self.I+=error
        D=(error-self.err)/dt
        return self.Kp*error+self.Ki*self.I+self.Kd*D