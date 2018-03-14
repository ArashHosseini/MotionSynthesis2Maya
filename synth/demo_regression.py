import sys
import time
import pickle
import numpy as np
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from network import create_core, create_regressor, create_footstepper
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_valid
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
preprocess_footstepper = np.load('preprocess_footstepper.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

batchsize = 1

def create_network(window, input):
    network_first = create_regressor(batchsize=batchsize, window=window, input=input, dropout=0.0)
    network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2)
    network_second.load(np.load('network_core.npz'))
    network = Network(network_first, network_second[1], params=network_first.params)
    network.load(np.load('network_regression.npz'))
    return network_first, network_second, network

from AnimationPlot import animation_plot

print(X.shape)

indices = [(30, 15*480), (60, 15*480), (90, 15*480)]

for index, length in indices:

    input = theano.tensor.ftensor3()

    Torig = np.load('../data/curves.npz')['C'][:,:,index:index+length]
    Torig = (Torig - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]
    
    network_footstepper = create_footstepper(batchsize=batchsize, window=Torig.shape[2], dropout=0.0)
    network_footstepper.load(np.load('network_footstepper.npz'))
    network_footstepper_func = theano.function([input], network_footstepper(input), allow_input_downcast=True)

    start = time.clock()
    W = network_footstepper_func(Torig[:,:3])
    W = (W * preprocess_footstepper['Wstd']) + preprocess_footstepper['Wmean']
    
    
    # alpha - user parameter scaling the frequency of stepping.
    #         Higher causes more stepping so that 1.25 adds a 
    #         quarter more steps. 1 is the default (output of
    #         footstep generator)
    #
    # beta - Factor controlling step duration. Increasing reduces 
    #        the step duration. Small increases such as 0.1 or 0.2 can
    #        cause the character to run or jog at low speeds. Small 
    #        decreases such as -0.1 or -0.2 can cause the character 
    #        to walk at high speeds. Too high values (such as 0.5) 
    #        may cause the character to skip steps completely which 
    #        can look bad. Default is 0.
    #
    #alpha, beta = 1.25, 0.1
    alpha, beta = 1.0, 0.0
    
    # controls minimum/maximum duration of steps
    minstep, maxstep = 0.9, -0.5
    
    off_lh, off_lt, off_rh, off_rt = 0.0, -0.1, np.pi+0.0, np.pi-0.1
    Torig = (np.concatenate([Torig,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lh)>np.clip(np.cos(W[:,1:2])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lt)>np.clip(np.cos(W[:,2:3])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rh)>np.clip(np.cos(W[:,3:4])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,
        (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rt)>np.clip(np.cos(W[:,4:5])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1], axis=1))
    
    print('Footsteps: %0.4f' % (time.clock() - start))

    #############
    
    network_first, network_second, network = create_network(Torig.shape[2], Torig.shape[1])
    network_func = theano.function([input], network(input), allow_input_downcast=True)
    
    start = time.clock()
    Xrecn = network_func(Torig)
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xtraj = ((Torig * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]).copy()
    print('Synthesis: %0.4f' % (time.clock() - start))
    
    Xnonc = Xrecn.copy()
    Xrecn = constrain(Xrecn, network_second[0], network_second[1], preprocess, multiconstraint(
        foot_sliding(Xtraj[:,-4:]),
        trajectory(Xtraj[:,:3]),
        joint_lengths()), alpha=0.01, iterations=250)
    Xrecn[:,-7:] = Xtraj
    
    animation_plot([Xnonc, Xrecn], interval=15.15)

