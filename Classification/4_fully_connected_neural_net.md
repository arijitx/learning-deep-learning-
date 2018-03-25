

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math,sys

data=pd.read_csv('dataset/data.csv')
def one_zero(x):
    if x==-1:
        return 0
    return x
data['label']=data['label'].apply(one_zero)
```


```python
#https://gist.github.com/rougier/c0d31f5cbdaac27b876c
def progress(value,  length=40, title = " ", vmin=0.0, vmax=1.0):
    # Block progression is 1/8
    blocks = ["", "▏","▎","▍","▌","▋","▊","▉","█"]
    vmin = vmin or 0.0
    vmax = vmax or 1.0
    lsep, rsep = "▏", "▕"
    value = min(max(value, vmin), vmax)
    value = (value-vmin)/float(vmax-vmin)
    v = value*length
    x = math.floor(v) # integer part
    y = v - x         # fractional part
    base = 0.125      # 0.125 = 1/8
    prec = 3
    i = int(round(base*math.floor(float(y)/base),prec)/base)
    bar = "█"*x + blocks[i]
    n = length-len(bar)
    bar = lsep + bar + " "*n + rsep
    sys.stdout.write("\r" + title + bar + " %.1f%%" % (value*100))
    sys.stdout.flush()
    
class neural_net():
    def __init__(self,ni,nh,no,lr=0.01,max_steps=10):
        self._lr=lr
        self._max_steps=max_steps
        self._n_hidden=nh
        self._n_output=no
        self._n_input=ni
        self._layers=[]
        self._labels=[]
        last_len=self._n_input+1
        idx=1
        for e in self._n_hidden:
            hidden_layer = [{'weights':np.random.rand(last_len),'name':'hidden_layer_'+str(idx)+'_unit_'+str(i+1)} for i in range(e)]
            self._layers.append(hidden_layer)
            last_len=e
            idx+=1
        output_layer= [{'weights':np.random.rand(last_len ),'name':'output_layer_unit_'+str(i+1)} for i in range(self._n_output)]    
        self._layers.append(output_layer)
    
    def preprocessing(self,X,Y):
        #add bias to input
        X=np.hstack((X,np.ones(X.shape[0]).reshape((X.shape[0],1))))
        #one hot encode label
        if Y is not None:
            self._labels=list(set(Y.flatten()))
            labels=dict(enumerate(self._labels))
            labels={v: k for k, v in labels.items()}
            new_y=np.zeros((Y.shape[0],len(labels)))
            for i in range(Y.shape[0]):
                new_y[i][labels[Y[i][0]]]=1
            Y=new_y
        return X,Y
    
    def activation(self,x,name='sigmoid'):
        if name=='sigmoid':
            return 1./(1.+np.exp(-x))
        else:
            return 1.
    
    def output(self,inputs,weights):
        return np.dot(weights.T,inputs)
    
    def forward_prop(self,row):
        inputs=row
        for layer in self._layers:
            new_inputs=[]
            for unit in layer:
                x=self.output(inputs,unit['weights'])
                unit['output']=self.activation(x)
                new_inputs.append(unit['output'])
            inputs=np.array(new_inputs)
        return inputs
    
    def derivate_b(self,x):
        return x*(1-x)
    
    def back_prop_error(self,y):
        for i in reversed(range(len(self._layers))):
            layer=self._layers[i]
            if i==len(self._layers)-1:
                for j in range(len(layer)):
                    layer[j]['delta']=y[j]-layer[j]['output']
            else:   
                for j in range(len(layer)):
                    error=0.
                    for unit in self._layers[i+1]:
                        error+=unit['weights'][j]*unit['delta']
                    layer[j]['delta']=error*self.derivate_b(layer[j]['output'])
    
    def update_weights(self,row):
        for i in range(len(self._layers)):
            layer=self._layers[i]
            inputs=row
            #for other layer except the first input is output of previous node
            if(i!=0):
                inputs=[unit['output'] for unit in self._layers[i-1]]
            for unit in self._layers[i]:
                unit['weights']+=unit['delta']*self._lr*np.array(inputs)
            
    def loss(self,X,Y):
        #cross entropy loss
        m=X.shape[0]
        error=0.
        for i in range(X.shape[0]):
            output=self.forward_prop(X[i])
            for j in range(Y.shape[1]):
                error+=Y[i][j]*np.log(output[j])+(1-Y[i][j])*np.log(1-output[j])
        return -error/m
    
    def train(self,X,Y):
        X,Y=self.preprocessing(X,Y)
        
        for step in range(self._max_steps):
            percentage=1
            for i in range(X.shape[0]):
                self.forward_prop(X[i])
                self.back_prop_error(Y[i])
                self.update_weights(X[i])
                if((i/X.shape[0])*100>percentage-1):
                    progress(i/X.shape[0]+.01,title='Epoch : '+str(step+1))
                    percentage+=1
            print('\nEpoch:',step+1,'Loss:',self.loss(X,Y))
                
                    
    def predict(self,X):
        X,_=self.preprocessing(X,None)
        res_full=[]
        for i in range(X.shape[0]):
            res=[]
            self.forward_prop(X[i])
            for unit in self._layers[-1]:
                res.append(unit['output'])
            res_full.append(self._labels[np.argmax(np.array(res))])
            #res_full.append(res)
        return np.array(res_full)
```


```python
X=data[['x','y']].as_matrix()
Y=data[['label']].as_matrix()

nn=neural_net(2,[3,2],2,lr=0.1,max_steps=50)
nn.train(X,Y)

```

    Epoch : 1▏████████████████████████████████████████▕ 100.0%
    Epoch: 1 Loss: 1.85685129062
    Epoch : 2▏████████████████████████████████████████▕ 100.0%
    Epoch: 2 Loss: 1.88902351545
    Epoch : 3▏████████████████████████████████████████▕ 100.0%
    Epoch: 3 Loss: 1.872657279
    Epoch : 4▏████████████████████████████████████████▕ 100.0%
    Epoch: 4 Loss: 1.8530375063
    Epoch : 5▏████████████████████████████████████████▕ 100.0%
    Epoch: 5 Loss: 1.83335052907
    Epoch : 6▏████████████████████████████████████████▕ 100.0%
    Epoch: 6 Loss: 1.81320306288
    Epoch : 7▏████████████████████████████████████████▕ 100.0%
    Epoch: 7 Loss: 1.79117586455
    Epoch : 8▏████████████████████████████████████████▕ 100.0%
    Epoch: 8 Loss: 1.7651999918
    Epoch : 9▏████████████████████████████████████████▕ 100.0%
    Epoch: 9 Loss: 1.73242523995
    Epoch : 10▏████████████████████████████████████████▕ 100.0%
    Epoch: 10 Loss: 1.69428898819
    Epoch : 11▏████████████████████████████████████████▕ 100.0%
    Epoch: 11 Loss: 1.65908035408
    Epoch : 12▏████████████████████████████████████████▕ 100.0%
    Epoch: 12 Loss: 1.62876947878
    Epoch : 13▏████████████████████████████████████████▕ 100.0%
    Epoch: 13 Loss: 1.60904371783
    Epoch : 14▏████████████████████████████████████████▕ 100.0%
    Epoch: 14 Loss: 1.65230481203
    Epoch : 15▏████████████████████████████████████████▕ 100.0%
    Epoch: 15 Loss: 1.70693783378
    Epoch : 16▏████████████████████████████████████████▕ 100.0%
    Epoch: 16 Loss: 1.77809617875
    Epoch : 17▏████████████████████████████████████████▕ 100.0%
    Epoch: 17 Loss: 1.75808548617
    Epoch : 18▏████████████████████████████████████████▕ 100.0%
    Epoch: 18 Loss: 1.53901056116
    Epoch : 19▏████████████████████████████████████████▕ 100.0%
    Epoch: 19 Loss: 1.1531068313
    Epoch : 20▏████████████████████████████████████████▕ 100.0%
    Epoch: 20 Loss: 2.33022241059
    Epoch : 21▏████████████████████████████████████████▕ 100.0%
    Epoch: 21 Loss: 2.1742940648
    Epoch : 22▏████████████████████████████████████████▕ 100.0%
    Epoch: 22 Loss: 1.75447800166
    Epoch : 23▏████████████████████████████████████████▕ 100.0%
    Epoch: 23 Loss: 0.28155747023
    Epoch : 24▏████████████████████████████████████████▕ 100.0%
    Epoch: 24 Loss: 0.618078972651
    Epoch : 25▏████████████████████████████████████████▕ 100.0%
    Epoch: 25 Loss: 0.192366363865
    Epoch : 26▏████████████████████████████████████████▕ 100.0%
    Epoch: 26 Loss: 0.587879471366
    Epoch : 27▏████████████████████████████████████████▕ 100.0%
    Epoch: 27 Loss: 0.862986565704
    Epoch : 28▏████████████████████████████████████████▕ 100.0%
    Epoch: 28 Loss: 2.85736274116
    Epoch : 29▏████████████████████████████████████████▕ 100.0%
    Epoch: 29 Loss: 0.59444430716
    Epoch : 30▏████████████████████████████████████████▕ 100.0%
    Epoch: 30 Loss: 1.46594207374
    Epoch : 31▏████████████████████████████████████████▕ 100.0%
    Epoch: 31 Loss: 2.89677069314
    Epoch : 32▏████████████████████████████████████████▕ 100.0%
    Epoch: 32 Loss: 2.71936491444
    Epoch : 33▏████████████████████████████████████████▕ 100.0%
    Epoch: 33 Loss: 1.532905264
    Epoch : 34▏████████████████████████████████████████▕ 100.0%
    Epoch: 34 Loss: 0.301287613243
    Epoch : 35▏████████████████████████████████████████▕ 100.0%
    Epoch: 35 Loss: 0.155600872742
    Epoch : 36▏████████████████████████████████████████▕ 100.0%
    Epoch: 36 Loss: 0.109986416027
    Epoch : 37▏████████████████████████████████████████▕ 100.0%
    Epoch: 37 Loss: 2.19071283101
    Epoch : 38▏████████████████████████████████████████▕ 100.0%
    Epoch: 38 Loss: 1.87423690916
    Epoch : 39▏████████████████████████████████████████▕ 100.0%
    Epoch: 39 Loss: 2.35773006873
    Epoch : 40▏████████████████████████████████████████▕ 100.0%
    Epoch: 40 Loss: 0.604164677258
    Epoch : 41▏████████████████████████████████████████▕ 100.0%
    Epoch: 41 Loss: 0.253646102076
    Epoch : 42▏████████████████████████████████████████▕ 100.0%
    Epoch: 42 Loss: 0.0773051909411
    Epoch : 43▏████████████████████████████████████████▕ 100.0%
    Epoch: 43 Loss: 0.0894037889627
    Epoch : 44▏████████████████████████████████████████▕ 100.0%
    Epoch: 44 Loss: 0.040485002233
    Epoch : 45▏████████████████████████████████████████▕ 100.0%
    Epoch: 45 Loss: 0.0363999076203
    Epoch : 46▏████████████████████████████████████████▕ 100.0%
    Epoch: 46 Loss: 0.0329192171073
    Epoch : 47▏████████████████████████████████████████▕ 100.0%
    Epoch: 47 Loss: 0.0301419774542
    Epoch : 48▏████████████████████████████████████████▕ 100.0%
    Epoch: 48 Loss: 0.0279016546811
    Epoch : 49▏████████████████████████████████████████▕ 100.0%
    Epoch: 49 Loss: 0.0260193921313
    Epoch : 50▏████████████████████████████████████████▕ 100.0%
    Epoch: 50 Loss: 0.0244009013483

