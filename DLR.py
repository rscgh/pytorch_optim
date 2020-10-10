import torch
from torch.optim import Optimizer


class DLR(Optimizer):
    r"""Implements .
    
    Faster Biological Gradient Descent Learning
    Ho Ling Li (2020)
    https://arxiv.org/abs/2009.12745

    based on the pytorch SDG implementation

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        alpha: 

    Example:
        >>> optimizer = torch.optim.DLR(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    
    Maths:
        SDG: 
          grad_w_ij = -n* (dC  / d_w_ij)

        DLR:
          grad_w_ij = -n0* ( [ |w_ij| + a ]  /  [ || wi || + a] ) *  (dC  / d_w_ij)

    """

    def __init__(self, params, lr=0.01, alpha = 0.1):

        defaults = dict(lr=lr, alpha=alpha)

        super(DLR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DLR, self).__setstate__(state)
        
        #for group in self.param_groups:
        #    group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                '''
                Structure of P:
                P ~ weights ~ model[0].weight
                model[0] ~ Linear(in_features=784, out_features=128, bias=True)     # has 128 nodes; and thus 784 weights per node
                model[0].weight.shape      torch.Size([128, 784]) ->                # model[0].weight[0, :] -> all the weights for the first node/neuron
                '''


                # difference in weights for use as update taken from the calculated gradient
                d_p = p.grad


                if len(p.shape) != 2: 
                  # this is to catch the relu / softmax case where we have only one to one connections
                  # and also potential other cases?

                  '''Standard SGD: grad_w_ij = -n* (dC  / d_w_ij) ~ update_graident = -learning_rate * gradient'''
                  p.add_(d_p * -group['lr']) # ~ #p.add_(d_p, alpha=-group['lr'])
                  continue

                
                '''
                New DLR in General: 
                grad_w_ij = -n0* ( [ |w_ij| + a ]  /  [ || wj || + a] ) *  (dC  / d_w_ij)

                update_graident = -learning_rate *  (synapse_weight + alpha / sum_of_all_arriving_syn + alpha)     * gradient
                                                     low weight     + alpha / low_sum + alpha       => 1
                                                     hight weight   + alpha / avg_sum + alpha       => <1 ~ 0.4
                                                     low weight     + alpha / avg_sum + alpha       => ~0

                                                     alpha should be chosen so that in the beginning we still have a stable learning rate, as weights are still
                                                     marginal


                Specific implementation of the DLR: 

                # prepare the weight updates to be weighted by individual synapse strength compared to sum of strength across all synapses arrving at the unit 
                            
                weight_spec_multpl_factors = individual absolute weight / sum of abs weights at unit;
                corrected by some alpha to have sufficient learning rates in the beginning
                '''

                # detach weights as we dont want to backprop the further calculations
                weights = p.detach()  # shape for model[0] (128,784) ~ number of nodes/neurons x number of arriving synapses
                weight_spec_multpl_factors = (weights.abs() + alpha) * (1 / (weights.abs().sum(axis=1) + alpha ) ).reshape(weights.shape[0], 1)   

                # update the weights (= parameter) in place
                p.add_(d_p * weight_spec_multpl_factors * -group['lr'])



        return loss
