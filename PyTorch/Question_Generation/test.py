from models import attentionRNN as m 
import numpy as np 
from torch.autograd import Variable
import torch

e=m.EncoderRNN(10,50,50)
d=m.BahdanauAttnDecoderRNN(50,50,10)

x=Variable(torch.from_numpy(np.random.randint(9,size=(10,15))))
lens=[9]*15

y=Variable(torch.from_numpy(np.random.randint(9,size=(15))))

outputs,hn=e(x,lens)
print(outputs.size())
print(hn.size())

h0=Variable(torch.zeros(1,1,15,50))

output,hidden=d(y,h0,outputs)
print(output.size())