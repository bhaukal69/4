import numpy as np

X=np.array(([2,9],[1,5],[3,6]),dtype=float)
Y=np.array(([92],[85],[89]),dtype=float)
X=X/np.amax(X)
Y=Y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))
def der_sigmoid(x):
    return x*(1-x)

epoch=5000
lr=0.1

wh=np.random.uniform(size=(2,3))
bh=np.random.uniform(size=(1,3))
wout=np.random.uniform(size=(3,1))
bout=np.random.uniform(size=(1,1))

for i in range(epoch):
    hinp=np.dot(X,wh)+bh
    h_layer=sigmoid(hinp)
    outinp=np.dot(h_layer,wout)+bout
    output=sigmoid(outinp)
    
    h_grad=der_sigmoid(h_layer)
    out_grad=der_sigmoid(output)
    
    EO=Y-output
    d_output=EO*out_grad
    
    EH=d_output.dot(wout.T)
    d_hid=EH*h_grad
    
    wout+=h_layer.T.dot(d_output)*lr
    wh+=X.T.dot(d_hid)*lr
    
print(X)
print(Y)
print(output)
