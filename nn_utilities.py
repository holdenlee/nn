import numpy as np
import theano
from collections import OrderedDict

#tparams is dictionary to theano variables
#pars is strings of parameters to include
#R -> Parameters -> [String] -> R
def weight_decay(decay_c, tparams, pars):
    tdecay_c = theano.shared(np_floatX(decay_c), name='decay_c')
    total = 0.
    #do this in a for loop because the parameters may be different dimensions - so it's awkward to concatenate.
    for name in pars:
        total += (tparams[name] ** 2).sum()
    return decay_c * total

# Int -> Int -> Bool -> [(Int, [Int])], enumerated list of minibatch indices.
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    #idx_list = [0..(n-1)]
    idx_list = np.arange(n, dtype="int32")
    
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

# Dict String a -> Dict String (Theano a)
def wrap_theano_dict(params, tparams=None):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    if tparams==None: 
        #if no pointer to a dictionary given, create one
        #TODO: initialize theano variables!
        tparams = OrderedDict()
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)
    return tparams

# Dict String (Theano a) -> Dict String a
def unwrap_theano_dict(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

#Int -> c:Int -> R^c
def oneHot(choices, n):
    #return [T.eq(n,x) for x in range(choices)]
    return T.as_tensor_variable([T.eq(n,x) for x in range(choices)])

def mapped_oneHot(choices, ns):
    return tmap(lambda x: oneHot(choices,x), ns)

def mapped_mapped_oneHot(choices, nss):
    return tmap2(lambda x: oneHot(choices,x), nss)

def tmap(f, n, fixed=[]):
    x, _ = theano.map(f, n, non_sequences=fixed)
    return x

def tmap2(f,n, fixed=[]):
    return tmap(lambda x: tmap(f, x, non_sequences=fixed), n)
