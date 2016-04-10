from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy as np
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import utilities
import nn_utilities

"""
Single steps for various optimizers
Each optimizer returns two (compiled) theano functions (with updates)
# f_grad_shared: calculates the cost, and updates its own parameters
# f_updates: update the neural net weights
Note cost is a Theano variable, not a compiled function.
"""

def sgd(lr, tparams, grads, cost, args):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    #zip(gshared,grads)

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(args, cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, cost, args):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * np_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(args, cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, cost, args):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(args, cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * np_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def train(
    init_params, # initial parameters (not in Theano)
    data_train, # : a (should be list of some sort)
    data_valid, # : a
    data_test, # : a
    batch_maker, # : Int -> a -> [[b]] 
        #function that given the batch size and data, returns a list of list of batch identifiers (ex. Int)
    get_data_f, # : [[b]] -> (a -> train)
        #function that given a list of list of batch identifiers, gives a function that takes the data and gives training
    cost, # : (train -> Theano Float)
        #cost function
    pred_error,
    args,
    tparamss,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run,
    dispFreq=10,  # Display to stdout the training progress every N updates
    optimizer=rmsprop,
    saveto='model.npz',
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
):
    print 'Building model'

    ## initialize a theano variable dictionary with parameter values from init_params
    ## tparamss = [wrap_theano_dict(init_param) for init_param in init_params]
    ## careful of overlapping...

    # tparamss is a list of dictionaries (CHECK THIS)
    # ? Are these theano dicts?
    tparams = union(tparamss)

    # ! use_noise is for dropout
    """(use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)"""
    
    # Get values of tparams (dictionary of all parameters)
    all_params=tparams.values()
    ## concat([tparams.values() for tparams in tparamss])

    # Compile the theano functions. 
    ## I don't think this commend is true - Note "args" contains information about the length of the vectors, so this effectively locks in the sequence length. IS THIS TRUE?
    # Compile the cost function.
    # cost : train -> Theano Float
    f_cost = theano.function(args, cost, name='f_cost')

    # Take the gradient, and compile that too.
    grads = tensor.grad(cost, wrt=all_params)
    f_grad = theano.function(args, grads, name='f_grad')

    #learning rate
    lr = tensor.scalar(name='lr')
    # The optimizer takes as arguments the learning rate, parameter dictionary (of theano vars), gradient and cost function, and arguments to those functions. All inputs are theano variables.
    # optimizer returns the f_grad_shared and the update function
    # ? what does f_grad_shared do?
    f_grad_shared, f_update = optimizer(lr, tparams, grads, cost, args)

    # Now starting optimization
    print 'Optimization'

    # The data can be in two forms ([a], [b]) or [a].
    # The length of ([a],[b]) or [a]
    def _len(li_or_pair):
        if type(li_or_pair)=="tuple":
            return len(li_or_pair[0])
        else:
            return len(li_or_pair)

    # If input is ([a],[b]), gives [a]; if input is [a] just gives [a]
    def get_first_if_tuple(maybe_tuple):
        if type(maybe_tuple)=="tuple":
            return maybe_tuple[0]
        else:
            return maybe_tuple

    # length of training data
    l_train = _len(data_train)
    l_valid = _len(data_valid)
    ## Ignore test right now.
    ## length of validation data
    ## l_test = _len(data_test)

    # batch_maker : a -> [[b]] 
    # function that given the data, returns a list of list of batch identifiers (ex. Int)
    # this is for the validation data. We run batch_maker on the training data inside the epoch loop.
    valid_batch_ids = batch_maker(valid_batch_size, data_valid)
    #test_batch_ids = batch_maker(data_test)

    # print the length of training data
    print "%d train examples" % l_train
    print "%d valid examples" % l_valid
    #print "%d test examples" % l_test
    
    # initialize history_errs, which will contain the validation errors from each time it checks the validation error.
    # best_p ?
    # bad count to 0.
    history_errs = []
    best_p = None
    bad_count = 0

    # if no validation frequency is give, validate once an epoch
    # (the length of an epoch is l_train / batch_size because each iteration ? takes batch_size samples.)
    if validFreq == -1:
        validFreq = l_train / batch_size
    #if no save frequency is give, validate once an epoch
    if saveFreq == -1:
        saveFreq = l_train / batch_size

    uidx = 0  # the number of updates done (increment by 1 every time we look at a batch and make an update)
    estop = False  # early stop
    start_time = time.time()
    try:
        #EPOCH LOOP
        #epoch index. (An epoch means going through the data once.)
        for eidx in range(max_epochs):
            # ?
            n_samples = 0
            
            # initialize epoch:
            # Call batch_maker to partition the training data into batches.
            # (ex. get the list of shuffled indices for the training set)
            ## kf = get_minibatches_idx(l_train, batch_size, shuffle=True)
            batch_ids = batch_maker(batch_size, data_train)
            
            #BATCH LOOP
            for batch_id in batch_maker:
                # batch_id is a batch (note batch_maker has the indices, not the actual data). Call get_data_f to get the actual data.
                # increase number of updates done by 1
                uidx += 1
                ## use_noise.set_value(1.)
                
                # Select the random examples for this minibatch
                """
                if type(train)=="tuple":
                    inputs = map(lambda li: [li[t] for t in train_index], list(train))
                else:
                    #only 1 argument. also wrap up in single-element list for consistency.
                    inputs = [[train[t] for t in train_index]]
                n_samples += args[0].shape[0]
                """
                
                # get the batch 
                # [[b]] -> (a -> train)
                batch = get_data_f(data_train, batch_id)
                ## n_samples += batch[0].shape[0]
                # Expect batch to be a list or a tuple representing multiple arguments. If it's a single argument, wrap it in a list so we can use *batch to unpack the arguments.
                if not isinstance(batch, (list, tuple)):
                    batch = [batch]
                # Compute the cost 
                cost = f_grad_shared(*batch)
                # f_update updates the ? given the learning rate.
                f_update(lrate)

                # if the cost is infinite or undefined, stop.
                if np.isnan(cost) or np.isinf(cost):
                    print 'bad cost detected: ', cost
                    return 1., 1., 1.

                # Display if it's time to do so.
                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                # Save if it's time to do so.
                if saveto and np.mod(uidx, saveFreq) == 0:
                    print 'Saving...',
                    
                    # save the best parameters---not the current ones.
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unwrap_theano_dict(tparams)
                    
                    # Save the arrays into "saveto" (which should be a .npz file).
                    # Save the history of errors.
                    # ? params are saved with the labels given by the dictionary.
                    np.savez(saveto, history_errs=history_errs, **params)
                    
                    ## Warning: python2 notation
                    # Dump the model options into [saveto].pkl.
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                # If it's time to validate
                if np.mod(uidx, validFreq) == 0:
                    ## use_noise.set_value(0.)
                    # For each batch identifier in batch,
                    # get that batch from data_train
                    # and calculate prediction error.
                    # Sum all these errors
                    train_err = sum([pred_error(get_data_f(data_train, batch_id)) for batch_id in batch])/data_train.size[0]
                    # Do the same for the validation error.
                    valid_err = sum([pred_error(get_data_f(data_valid, batch_id)) for batch_id in batch_valid])/data_valid.size[0]
                    ## test_err = sum([pred_error(get_data_f(data_test, batch_id)) for batch_id in batch])/data_test.size[0]
                    
                    # record the validtion error in the history.
                    history_errs.append(valid_err) #[valid_err, test_err])

                    # if the validation error is smaller than any seen so far
                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:, 0].min()):
                        # then save the parameters to best_p
                        best_p = unzip(tparams)
                        # set bad_counter to 0.
                        bad_counter = 0
                    
                    # Show the training and validation error.
                    print ('Train ', train_err, 'Valid ', valid_err) #,
                           #'Test ', test_err)
                    
                    # If the current validation error is greater than the minimum validation error up to <patience> trials ago (why aren't we looking at the past few?), add 1 to bad_counter 
                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        #If this happens more times than patience allows, then signal that we stopped early and stop.
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break
            
            # Show number of samples seen. 
            # ! This is currently 0.
            print 'Seen %d samples' % n_samples

            # If early stop activated, then stop.
            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        # not sure what this does
        # Puts the best_p (best parameters) into a dictionary?
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

##    use_noise.set_value(0.)
    

    ## kf_train_sorted = get_minibatches_idx(_len(train), batch_size)
    # At the very end, calculate the training and validation error again.
    kf_train_sorted = batch_maker(batch_size, data_train)
    # Calculate training and validation error (note we also did this with [validFreq] frequency)
    train_err = sum([pred_error(get_data_f(data_train, batch_id)) for batch_id in batch])/data_train.size[0]
    valid_err = sum([pred_error(get_data_f(data_valid, batch_id)) for batch_id in batch_valid])/data_valid.size[0]

    print 'Train ', train_err, 'Valid ', valid_err #, 'Test ', test_err

    # Final save. Save the training error, validation error, history of errors, and the best parameters (unpacked)
    if saveto:
        np.savez(saveto, train_err=train_err,
                    valid_err=valid_err, #test_err=test_err,
                    history_errs=history_errs, **best_p)
    # How long the code took to run
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err #, test_err
