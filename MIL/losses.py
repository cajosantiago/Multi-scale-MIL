import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geomloss import SamplesLoss
import ot

import torch
from cvxopt import matrix, spdiag, solvers
import numpy as np

class DistAndClassWeightedCE(torch.nn.Module):
    def __init__(self, num_classes, class_weights, use_LOW=False, f=lambda d: d):
        super(DistAndClassWeightedCE, self).__init__()
        
        # Matrice de poids de distance (sans normalisation)
        self.register_buffer('W', torch.zeros((num_classes, num_classes)))
        for i in range(num_classes):
            for j in range(num_classes):
                self.W[i, j] = f(abs(i - j))/num_classes
        
        # Poids de classe
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        
        #self.distance_weight = distance_weight

    def forward(self, inputs, targets, use_LOW=False):
        log_probs = F.log_softmax(inputs, dim=1)
        W = self.W.to(inputs.device)

        ce_loss = F.nll_loss(log_probs, targets, reduction='none')

        predicted_classes = torch.argmax(inputs, dim=1)
        distance_penalty = W[targets, predicted_classes]

        weighted_loss = ce_loss *(1+distance_penalty) * self.class_weights[targets]
        if use_LOW:
            return weighted_loss
        else:
            return weighted_loss.mean()     
        
# inspired from Ines's paper loss :

class DistanceLoss(torch.nn.Module):
    def __init__(self, args, class_weights):
        super(DistanceLoss, self).__init__()
        self.num_classes = args.num_classes
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """Wasserstein distance between predicted and true CDFs
        
        Args:
            inputs : (batch_size, num_classes) - raw logits
            targets : (batch_size,) - true class indices
        """

        num_classes = self.num_classes
        class_weights = self.class_weights.to(inputs.device)

        probs = F.softmax(inputs, dim=1)
        cdf_preds = torch.cumsum(probs, dim=1)

        target_onehot = F.one_hot(targets, num_classes=num_classes).float()
        cdf_true = torch.cumsum(target_onehot, dim=1)

        wasserstein = (cdf_preds - cdf_true).abs().sum(dim=1)
        loss = (wasserstein * class_weights[targets]).mean()
    
        return loss #/ (batch_size * batch_size)



class TotalLoss(torch.nn.Module):
    def __init__(self, num_classes, class_weights, args, alpha=0.2):
        super(TotalLoss, self).__init__()
        if args is None:
            raise ValueError("args must be provided to TotalLoss")
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        self.dist_loss = DistanceLoss(args, class_weights=class_weights)
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dist = self.dist_loss(inputs, targets)
        return self.alpha * ce + (1 - self.alpha) * dist
    


def compute_weights(lossgrad, lamb):

    device = lossgrad.get_device()
    lossgrad = lossgrad.data.cpu().numpy()

    # Compute Optimal sample Weights
    aux = -(lossgrad**2+lamb)
    sz = len(lossgrad)
    P = 2*matrix(lamb*np.identity(sz))
    q = matrix(aux.astype(np.double))
    A = spdiag(matrix(-1.0, (1,sz)))
    b = matrix(0.0, (sz,1))
    Aeq = matrix(1.0, (1,sz))
    beq = matrix(1.0*sz)
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 20
    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    sol = solvers.qp(P, q, A, b, Aeq, beq)
    w = np.array(sol['x'])
    
    return torch.squeeze(torch.tensor(w, dtype=torch.float, device=device))



class LOWLoss(torch.nn.Module):
    def __init__(self, lamb=0.1):
        super(LOWLoss, self).__init__()
        self.lamb = lamb # higher lamb means more smoothness -> weights closer to 1
        self.loss = DistAndClassWeightedCE()  # replace this with any loss with "reduction='none'"

    def forward(self, logits, target):
        # Compute loss gradient norm
        output_d = logits.detach()
        loss_d = torch.mean(self.loss(output_d.requires_grad_(True), target), dim=0)
        loss_d.backward(torch.ones_like(loss_d))
        lossgrad = torch.norm(output_d.grad, 2, 1)

        # Computed weighted loss
        weights = compute_weights(lossgrad, self.lamb)
        loss = self.loss(logits, target)
        loss = torch.mean(torch.mul(loss, weights), dim=0)

        return loss