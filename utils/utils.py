import numpy as np
import torch, torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

def warmup_scheduler(lr, gamma_up, gamma_down, n_epochs_warmup, step):
    """
    Learning rate scheduler with warm-up
    
    Args:
        lr: base learning rate (float)
        gamma_up: multiplicative factor during warm-up epochs (float)
        gamma_down: multiplicative factor after warm-up epochs (float)
        n_epochs_warmup: number of warm-up epochs (int)

    Return:
        lambda_ : lambda function to calculate new lr (callable)
    """
    def lambda_(epoch):
        if epoch < n_epochs_warmup:
            # gradually augment learning rate from 0 to lr with a sine scheme
            # return lr * math.sin((math.pi / 2) * (epoch / n_epochs_warmup))
            return (gamma_up ** (n_epochs_warmup - epoch)) * lr
        else:
            k = (epoch - n_epochs_warmup) // step 
            return (gamma_down ** k) * lr

    return lambda_


def model_parameters_count(model):
    """
    Count the model total number of parameters

    Args:
        model : (nn.Module)

    Return a dict with name of module and number of params
    """
    trainable, not_trainable = 0, 0
    for p in model.parameters():
        count = p.flatten().size()[0]
        if p.requires_grad:
            trainable += count
        else:
            not_trainable += count
    
    return dict(trainable=trainable, fixed=not_trainable)


def smooth_cross_entropy(pred, targets, p=0.0, n_classes=None):
    """
        Cross-entropy with label smoothing for slide
        
        Args:
            pred : model output (torch.Tensor)
            targets : ground-truth as index (torch.Tensor)
            p : probability for label smoothing (float)
            n_classes : number of classes (int)

        Return tensor of loss (torch.Tensor)
    """

    def smooth_labels(targets, p, n_classes):
        # produce on-hot encoded vector of targets
        # fill with:  p / (n_classes - 1)
        # fill true classes value with : 1 - p
        
        res = torch.empty(size=(targets.size(0), n_classes), device=targets.device)
        res.fill_(p /(n_classes - 1))
        res.scatter_(1, targets.data.unsqueeze(1), 1. - p)
        return res
    
    def smooth_labels_randomly(targets, p, n_classes):
        # produce on-hot encoded vector of targets
        # fill true classes value with random value in : [1 - p, 1]
        # and completes the other to sum up to 1
        
        res = torch.zeros((targets.size(0), n_classes), device=targets.device)
        rand = 1 - torch.rand(targets.data.unsqueeze(1).shape, device=targets.device)*p
        res.scatter_(1, targets.data.unsqueeze(1), rand)
        fill_ = (1 - res.sum(-1))/(n_classes - 1)
        return res.maximum(fill_.unsqueeze(dim=-1).repeat(1, n_classes))

    assert isinstance(pred, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    
    if p:
        if n_classes is None:
            n_classes = pred.size(-1)
            
        targets = smooth_labels_randomly(targets, p, n_classes)
        pred = pred.log_softmax(dim=-1)
        cce_loss = torch.sum(-targets * pred, dim=1)
        return torch.mean(cce_loss)
    else:
        return nn.functional.cross_entropy(pred, targets.to(dtype=torch.long))


def compute_classes_weights(y):
    """
    Compute class weights based on sklearn.utils.class_weight.compute_class_weight

    args:
        y (iterable): list of ground-truth for all samples of train data
    """

    return compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)

def compute_binary_metrics(y, y_pred):
    """
    Return dict of binary metrics

    args:
        y (list, numpy.ndarray): 1D ground-truth labels
        y_pred (list, numpy.ndarray): 1D prediction
    """

    metrics = dict()

    # precision
    precision = precision_score(y, y_pred)
    metrics.update({"precision": precision})

    # recall
    recall = recall_score(y, y_pred)
    metrics.update({"recall": recall})

    # f1 score
    fscore = f1_score(y, y_pred)
    metrics.update({"f1_score": fscore})

    # get confusion metrix for next metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred, normalize=None).ravel()

    # specificity (true negative rate)
    specificity = tn / (tn+fp)
    metrics.update({"specificity": specificity})

    # negative predictive value
    npv = tn / (tn+fn)
    metrics.update({"NPV": npv})
    
    return metrics