import numpy as np
from theano import *
import theano.tensor as T
from theano.tensor.nnet import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

import optimizers
import utilities
import nn_utilities

"""Parameters"""
def unpack_params(tparams, li):
    return [tparams[name] for name in li]

"""Basic NN's"""
def nn_layer1(x, W, b):
    return x * W + b

#R^m -> Parameters -> R^p, where W::R^{m x p} and b::R^p.
def nn_layer(x, tparams):
    W, b = unpack_params(tparams, ["W", "b"])
    return nn_layer1(x, W, b)

#R^k -> R^k -> R
def logloss(pred, actual):
    #sum on innermost axis.
    return -(actual * corrected_log(pred)).sum(axis=-1)
#CHECK THAT THIS MAPS

#Warning: this doesn't map.
#R^k -> Nat -> R
def logloss_i(pred, actual_i):
    return -corrected_log(pred[actual_i])

#R -> R
def corrected_log(x):
    return T.log(T.max(1e-6,x))

#return dictionary of parameters (with default random initialization)
def init_params_nn(n, m, init='zeros'):
    rand_f = lambda l1, l2: np.asarray(np.random.normal(0, 1/np.sqrt(n), (l1,l2)))
    (f, g) = case(init,
                [('zeros', (np.zeros, lambda c, r: np.zeros((c, r)))),
                 ('rand', (np.zeros, rand_f))])
    return init_params_with_f_nn(n,m,f,g)

#returns a dictionary of parameters, initialized using the functions f and g.
def init_params_f_nn(n,m,f,g):
    pairs = [("W",np.asarray(g(n, m))),
             ("b",np.asarray(f(m)))]
    return OrderedDict(pairs)


"""LSTM functions"""
#x, C, h are the inputs, and C1, h1 as the outputs.
#the rest are weight vectors.
def step_lstm1(x, C, h, Wf, bf, Wi, bi, WC, bC, Wo, bo):
    hx = T.concatenate([h,x]) #dimension m+n
    f = sigmoid(T.dot(hx, Wf) + bf) #dimension m
    i = sigmoid(T.dot(hx, Wi) + bi) #dimension m
    C_add = T.tanh(T.dot(hx, WC) + bC) #dimension m
    C1 = f * C + i * C_add #dimension m
    o = sigmoid(T.dot(hx, Wo) + bo) #dimension m
    h1 = o * T.tanh(C1) #dimension m
    return [C1, h1] #dimension 2m (as 2 separate lists)

#the same function, but with the parameters grouped together.
#R^n -> R^m -> R^m -> Parameters -> (R^m, R^m)
def step_lstm(x, C, h, tparams): 
    Wf, bf, Wi, bi, WC, bC, Wo, bo = unpack_params(tparams, ["Wf", "bf", "Wi", "bi", "WC", "bC", "Wo", "bo"])
    return step_lstm1(x, C, h, Wf, bf, Wi, bi, WC, bC, Wo, bo)

#Now for scanning and mapping
#1. unfold step_lstm into something that accepts and gives a sequence
#2. make it something that will operate on a whole batch of sequences
#R^m -> R^m -> R^{s x n} -> Parameters -> (R^{s x m}, R^{s x m})
def sequence_lstm(C0, h0, xs, tparams):
    #we need tparams because we need a link to the shared variables.
    #CHECK: please check that this gives the weights in the right order.
    Wf, bf, Wi, bi, WC, bC, Wo, bo = unpack_params(tparams, ["Wf", "bf", "Wi", "bi", "WC", "bC", "Wo", "bo"])
    #the function fn should have arguments in the following order:
    #sequences, outputs_info (accumulators), non_sequences
    #(x, C, h, Wf, bf, Wi, bi, WC, bC, Wo, bo)
    ([C_vals, h_vals], updates) = theano.scan(fn=step_lstm1,
                                          sequences = xs, 
                                          outputs_info=[C0, h0], #initial values of the memory/accumulator
                                          non_sequences=[Wf, bf, Wi, bi, WC, bC, Wo, bo], #fixed parameters
                                          strict=True)
    return [C_vals, h_vals]

#play around with numpy to see how things map, to define step_multiple_lstm.

def sequence_multiple_lstm1(Cs0, hs0, xss, tparams):
    return tmap(f, [Cs0, hs0, xss], tparams)

def step_multiple_lstm(xs, C, h, tparams):
    #Everything maps automatically. (We've only used matrix multiplication and scalar functions like sigmoid.)
    return step_lstm(xs, C, h, tparams)

def sequence_multiple_lstm(Cs0, hs0, xss, tparams):
    #Everything maps. Note xss, Cs0, hs0 must be Theano matrices, not a list of Theano lists.
    #However, the input will be $((R^n)^k)^s$ so we need to switch axes.
    #Dimensions count inwards        2  1  0
    Cs0.dimshuffle([0,2,1])
    return sequence_lstm(Cs0, hs0, xss, tparams)

"""Functions to evaluate the NN's and calculate loss"""
#unmapped version. taking indices
def fns_lstm(C0, h0, xis, yi, tparams1, tparams2)
    #, last_only = True):
    #evaluate the LSTM on this sequence
    [C_vals, h_vals] = sequence_lstm(C0, h0, xs, tparams)
    #it's simpler to get both the function for the last and the function for all
    """ 
    if last_only:
        h_vals = h_vals[-1]
        C_vals = C_vals[-1]
    """
    #feed into the neural net and get vector of activations
    acts = nn_layer(h_vals)
    #prediction is the argmax value. Take argmax along innermost (-1) axis
    pred = T.argmax(acts, axis=-1)
    #loss function
    loss = logloss(acts, yi)
    acts_last = acts[-1]
    pred_last = pred[-1]
    loss_last = loss[-1]
    #1 if predicted next one correctly, 0 otherwise
    acc_last = yi[pred_last]
    return acts_last, pred_last, loss_last, acc_last, acts, pred, loss

#is ALMOST THE SAME as above...
def fns_multiple_lstm(m, xis, yi, (tparams1, tparams2))
    C0 = np.zeros(m)
    h0 = np.zeros(m)
    #evaluate the LSTM on this sequence
    [C_vals, h_vals] = sequence_lstm(C0, h0, xs, tparams)
    #feed into the neural net and get vector of activations
    acts = nn_layer(h_vals)
    #prediction is the argmax value. Take argmax along innermost (-1) axis
    pred = T.argmax(acts, axis=-1)
    #loss function
    loss = logloss(acts, yi)
    acts_last = acts[...,-1] #add ellipses
    pred_last = pred[...,-1]
    loss_last = loss[...,-1]
    #1 if predicted next one correctly, 0 otherwise
    #http://stackoverflow.com/questions/33929368/how-to-perform-a-range-on-a-theanos-tensorvariable
    acc_last = yi[T.arange(xis.size[0]),pred_last]
    #http://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
    return acts_last, pred_last, loss_last, acc_last, acts, pred, loss

#return dictionary of parameters (with default random initialization)
def init_params_lstm(n, m, init='zeros'):
    #normalize this!
    rand_f = lambda l1, l2: np.asarray(np.random.normal(0, 1/np.sqrt(n), (l1,l2)))
    (f, g) = case(init,
                [('zeros', (np.zeros, lambda c, r: np.zeros((c, r)))),
                 ('rand', (np.zeros, rand_f))])
    return init_params_with_f_lstm(n,m,f,g)

#returns a dictionary of parameters, initialized using the functions f and g.
def init_params_with_f_lstm(n,m,f,g):
    pairs = [("Wf",np.asarray(g(m+n, m))),
             ("bf",np.asarray(f(m))),
             ("Wi",np.asarray(g(m+n, m))),
             ("bi",np.asarray(f(m))),
             ("WC",np.asarray(g(m+n, m))),
             ("bC",np.asarray(f(m))),
             ("Wo",np.asarray(g(m+n, n))),
             ("bo",np.asarray(f(o)))]
    return OrderedDict(pairs)

#Int^b -> R^{l * n} -> (R^{b * s * n}, R^{b * n})
def get_data_f(indices, data):
    #given indices, get the sequences in data starting at those indices.
    #(seqs, ys)\in R^{b*s*n} * R^{b*n}
    return ([data[i:i+s] for i in indices], [data[i+s-1] for i in indices])
    #does s include last? 

#li's are sequences, ex. [0,3,2,1,1,3,1]
#the elements of the sequence are in [0..(n-1)], ex. n=4 above
#m is the memory size
#s is the sequence length, ex. 3 divides the above into [0,3,2],..,[1,3,1].
def train_lstm(li_train, li_valid, li_test, n, m, s, batch_size):
    #m:Int -> Int -> R^m
    hot_li_train, hot_li_valid, hot_li_test = [map(lambda x: oneHot(n, x), li) for li in [li_train, li_valid, li_test]]
    #n_seqs_train, n_seqs_valid, n_seqs_test = [len(li) - s + 1 for li in [li_train, li_valid, li_test]]
    #note alternatively we can keep it as a single tensor...

    def batch_maker(data):
        return get_minibatches_idx(len(data)-s, batch_size, shuffle=True)

    xis = T.dtensor3('xis')
    yi = T.dmatrix('yi')
    tparams1 = init_params_lstm(m,n,'rand')
    tparams2 = init_params_lstm(m,n,'rand')
    _,_,loss,acc,_,_,_ = fns_multiple_lstm(m, xis, yi, tparams1, tparams2)
    #loss_f = function([xis,yi],loss)
    #acc_f = function([xis,yi],acc)

    train(patience=10,  # Number of epoch to wait before early stop if no progress
          max_epochs=5000,  # The maximum number of epoch to run,
          optimizer=rmsprop,
          saveto='model.npz',
          validFreq=500,  # Compute the validation error after this number of update.
          saveFreq=1000,  # Save the parameters after every saveFreq updates
          batch_size=16,  # The batch size during training.
          valid_batch_size=64,  # The batch size used for validation/test set.
          init_params=init_params_lstm(n, m), # initial parameters
          data_train=li_train , # : a (should be list of some sort)
          data_valid=li_valid, # : a
          data_test=li_test, # : a
          batch_maker, # : a -> [b] 
          #function that given the data, returns a list batch identifiers (ex. [Int])
          get_data_f, # : b -> a -> train
          #function that given a list of batch identifiers, gives a function that takes the data and gives training
          #CHECK the numpy initialization
          loss, # : (train -> Theano Float)
          acc
          args=[xis,yi]
)
    
