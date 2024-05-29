import os, argparse, logging, time, json, random
import tqdm
import numpy as np
import pandas
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from utils import data_loaders, utils
from models import FusionMIL

global TRAINING_CFG
TRAINING_CFG = dict(
    abmil={
        "optimizer": torch.optim.Adam,
        "optimizer_opts": dict(lr=1e-4, weight_decay=5e-4),
        "scheduler": None,
        "epochs": 100,
    },
    
    transmil={
        "optimizer": torch.optim.Adam,
        "optimizer_opts": dict(lr=2e-4, weight_decay=1e-5),
        "scheduler": None,
        "epochs": 200,
    },
    
    dtfdmil={
        "optimizer": torch.optim.Adam,
        "optimizer_opts": dict(lr=1e-4, weight_decay=1e-4),
        "scheduler": torch.optim.lr_scheduler.MultiStepLR,
        "scheduler_opts": dict(milestones=[100], gamma=0.2),
        "epochs": 200,
    },
)

def train(args, dataloader, model, optimizer, device, class_weights=None):
    """
    Training steps on whole train data

    args:
        args (Namespace): args of main script
        dataloader (iterable): data loader
        model (nn.Module): model to train
        optimizer (torch.Optimizer): optimizer for model training
        device (torch.device): device for model inference
        class_weights (list or numpy.ndarray): list of class weights

    return total loss and accuracy
    """

    # set data loader in training mode
    dataloader.train()
    
    loss, acc = [], []
    for batch in tqdm.tqdm(dataloader, ncols=50):

        if batch is StopIteration:
            break

        x, label = batch
        
        # select features and unsqeeze to get shape (1,N,C)
        x = [x[k].unsqueeze(dim=0).to(device=device) for k in model.feature_extractor]
        if len(x) == 1:
            # if only one feature extractor used
            x = x[0]
        
        # convert ground-truth
        y = torch.tensor([label], dtype=torch.float32, device=device)

        # forward model
        pred = model(x)

        # computes batch loss
        if args.num_cls == 1:
            if isinstance(pred, tuple):
                pred = [torch.sigmoid(i).squeeze(dim=0) for i in pred]
            else:
                pred = torch.sigmoid(pred).squeeze(dim=0)
            loss_fn = nn.BCELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        batch_loss = model.calculate_loss(pred, y, loss_fn)

        if not class_weights is None:
            batch_loss = batch_loss * class_weights[label]

        # optimizer reset gradients
        optimizer.zero_grad()
        
        # backpropagation
        batch_loss.backward()

        # clip gradients
        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # optimizer update
        optimizer.step()

        # save for loss and accuracy
        loss.append(batch_loss.to(device='cpu').item())
        if isinstance(pred, (list, tuple)):
            # TODO make this handling safer !
            # DTFD-MIL return tuple
            pred = pred[1]
        acc.append((y.to(device='cpu').numpy(), pred.detach().to(device='cpu').numpy()))
    
    # calculate total accuracy
    y, pred = zip(*acc)
    y = np.array(y).squeeze(axis=-1)
    pred = np.array(pred)
    if args.num_cls > 1:
        pred = np.argmax(pred, axis=-1).squeeze(axis=-1)
    else:
        pred = pred.squeeze(axis=-1).round()
    acc = accuracy_score(y.astype(dtype=np.int16), pred.astype(dtype=np.int16))

    return float(np.mean(loss)), float(acc)

def validate(args, dataloader, model, device):
    """
    Validation step

    args:
        args (Namespace): args of main script
        dataloader (iterable): data loader
        model (nn.Module): model to validate
        device (torch.device): device for model inference

    return total loss and accuracy
    """

    # set data loader in validation mode
    dataloader.valid()

    loss, acc = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, ncols=50):

            if batch is StopIteration:
                break

            x, y = batch

            # select features
            x = [x[k].unsqueeze(dim=0).to(device=device) for k in model.feature_extractor]
            # if only one feature extractor used
            if len(x) == 1:
                x = x[0]
            
            # convert ground-truth
            y = torch.tensor([y], dtype=torch.float32, device=device)

            # forward model
            pred = model(x)

            # computes batch loss
            if args.num_cls == 1:
                if isinstance(pred, tuple):
                    pred = [torch.sigmoid(i).squeeze(dim=0) for i in pred]
                else:
                    pred = torch.sigmoid(pred).squeeze(dim=0)
                loss_fn = nn.BCELoss()
            else:
                pred = pred.squeeze(dim=0)
                loss_fn = nn.CrossEntropyLoss()
            batch_loss = model.calculate_loss(pred, y, loss_fn)

            if isinstance(pred, (list, tuple)):
                # TODO make this handling safer !
                # DTFD-MIL return tuple
                pred = pred[1]

            # save for loss and accuracy
            loss.append(batch_loss.to(device='cpu').item())
            acc.append((y.to(device='cpu').numpy(), pred.to(device='cpu').numpy()))
    
    # calculate total accuracy
    y, pred = zip(*acc)
    y = np.array(y, dtype=np.int16).squeeze(axis=-1)
    pred = np.array(pred, dtype=np.int16)
    if args.num_cls > 1:
        pred = np.argmax(pred, axis=-1).squeeze(axis=-1)
    else:
        pred = pred.squeeze(axis=-1).round()
    acc = accuracy_score(y, pred)

    metrics = utils.compute_binary_metrics(y, pred)

    return float(np.mean(loss)), float(acc), metrics

def test(args, dataloader, model, device):
    """
    Test model on test data

    args:
        args (Namespace): args of main script
        dataloader (iterable): data loader
        model (nn.Module): model to test
        device (torch.device): device for model inference

    return test metrics
    """

    # set data loader in test mode
    dataloader.test()

    Y_hat, Y = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, ncols=50):

            if batch is StopIteration:
                break
            
            x, y = batch

            # select features
            x = [x[k].unsqueeze(dim=0).to(device=device) for k in model.feature_extractor]
            # if only one feature extractor used
            if len(x) == 1:
                x = x[0]
            
            # convert ground-truth
            y = torch.tensor([y], dtype=torch.float32, device=device)

            # forward model
            pred = model(x)

            # check for DTFD-MIL
            if isinstance(pred, (list, tuple)):
                pred = pred[1]

            # computes batch loss
            if args.num_cls == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = nn.functional.softmax(pred, dim=-1)

            # save for loss and accuracy
            Y_hat.append(pred.to(device='cpu').numpy())
            Y.append(y.to(device='cpu').numpy())
    
    # stack predictions and labels
    Y_hat = np.concatenate(Y_hat, axis=0)
    Y = np.stack(Y, axis=0)

    metrics = dict()

    # ROC AUC
    roc_auc = roc_auc_score(Y, Y_hat)
    metrics.update({"auc": roc_auc})
    
    # convert to class index
    if args.num_cls == 1:
        Y_hat = Y_hat.round().squeeze(axis=-1)
    else:
        Y_hat = Y_hat.argmax(axis=-1).squeeze(axis=-1)
    
    Y_hat = Y_hat.astype(dtype=np.int16)
    Y = Y.astype(dtype=np.int16)

    # accuracy
    acc = accuracy_score(Y, Y_hat)
    metrics.update({"acc": acc})

    for k, v in utils.compute_binary_metrics(Y, Y_hat).items():
        metrics.update({k: v})
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="path to the dataset of features")
    parser.add_argument('--fold', type=str, required=True, help='path to CSV file that specify fold for each patient')
    parser.add_argument('--output', type=str, required=True, help='where to store the model')
    parser.add_argument('--name', required=True, type=str, help="name of model to save files")
    parser.add_argument('--num_cls', required=True, type=int)
    parser.add_argument('--labels', type=str, required=True, help='path to treatment response file')
    parser.add_argument('--gpu', required=True, type=str)
    parser.add_argument('--extractor', type=str, required=True, help='feature extractor to use \
                        choices are [conch, gigapath, hipt, resnet], it can be one or a combination of these')
    parser.add_argument('--mil_aggregator', type=str, required=True, choices=["abmil", "dtfdmil"], help='MIL aggregator to use')
    parser.add_argument('--fusion', type=str, default=None, choices=[None, "concat", "attn_pool"], help='features fusion to use \
                        concat: simple concatenation of features \
                        attn_pool: MLP (project to fixed shared dimension) + attention pooling')
    parser.add_argument('--attn_out_dim', default=None, type=int, help="dimension of attention pooling output")
    parser.add_argument('--attn_shared_dim', default=None, type=int, help="dimension inside attention pooling")
    parser.add_argument('--class_weighting', action='store_true', help="mitigate class imbalance with class weighting")
    parser.add_argument('--limit_per_slide', type=int, default=None, help='limit of tiles per slide, if None use every tiles available')
    parser.add_argument('--clip_grad', type=float, default=0, help='gradient clipping norm')
    parser.add_argument('--save_freq', type=int, default=-1, help='frequency, in epochs, to save the model')
    args = parser.parse_args()

    args.name = "{}_{}_{}_{}".format(args.name, 
                                     "_".join(args.extractor.split(",")),
                                     args.fusion,
                                     args.mil_aggregator)

    # set torch device
    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    print('device: ', device)

    # set random seeds for reproductibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # load folds csv
    fold_split = pandas.read_csv(args.fold)

    for fold_id in fold_split['fold'].unique():

        print("\n training model on fold {}".format(fold_id))

        # logger
        print('\t log arguments..')
        handler = logging.FileHandler(filename=os.path.join(model_dir, 'log'))
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger = logging.getLogger(name=str(fold_id))
        logger.setLevel(level=logging.INFO)
        logger.addHandler(handler)

        logger.info(time.strftime("%x %X"))
        for k, v in vars(args).items():
            logger.info('{} : {}'.format(k,v))

        # load data
        print("\t loading data..")
        logger.info("loading data..")
        dataloader = data_loaders.Dataset(args, fold_id)

        # class weights
        if args.class_weighting:
            print("\t calculate class weights..")
            dataloader.train()
            class_weights = dataloader.class_weights()
        else:
            class_weights = None

        # build model
        print("\t initialize model..")
        logger.info("initialize model..")
        model = FusionMIL.FusionMIL(args)

        # directory to save the model
        model_dir = os.path.join(args.output, args.name, "fold_{}".format(fold_id))
        os.makedirs(model_dir, exist_ok=True)

        # save model config
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            json.dump(vars(args), f)
       
        # optimizer and lr scheduler
        optimizer = TRAINING_CFG[args.mil_aggregator]["optimizer"](model.parameters(), **TRAINING_CFG[args.mil_aggregator]["optimizer_opts"])
        scheduler = TRAINING_CFG[args.mil_aggregator]["scheduler"]
        if not scheduler is None:
            scheduler = scheduler(optimizer, **TRAINING_CFG[args.mil_aggregator]["scheduler_opts"])
        
        # send model to GPU
        print("\t send model to {}..".format(device))
        logger.info("send model to {}..".format(device))
        model = model.to(device=device)
        
        train_metrics = dict(loss=[], acc=[])
        val_metrics = dict(loss=[], acc=[])
        for epoch in range(TRAINING_CFG[args.mil_aggregator]["epochs"]):
            logger.info("epoch {}".format(epoch+1))
            
            # train steps
            print("\t train")
            model.train()
            train_loss, train_acc = train(args, dataloader, model, optimizer, device, class_weights=class_weights)
            train_metrics['loss'].append(train_loss)
            train_metrics['acc'].append(train_acc)

            # validation steps
            print("\t validation")
            model.eval()
            val_loss, val_acc, metrics = validate(args, dataloader, model, device)
            val_metrics['loss'].append(val_loss)
            val_metrics['acc'].append(val_acc)
            for k, v in metrics.items():
                if not k in list(val_metrics.keys()):
                    val_metrics.update({k: []})

                val_metrics[k].append(v)

            print('end of epoch {}'.format(epoch+1))
            logger.info('end of epoch {} : {}'.format(epoch+1, time.strftime("%x %X")))
            
            # lr scheduler step
            if not scheduler is None:
                scheduler.step()

            # save training metrics
            with open(os.path.join(model_dir, "train.json"), "w") as outfile:
                json.dump(train_metrics, outfile)

            # save validation metrics
            with open(os.path.join(model_dir, "valid.json"), "w") as outfile:
                json.dump(val_metrics, outfile)

            # save the weights of the model if current epoch proportionnal to save_freq
            if args.save_freq > 0 and ((epoch+1) % args.save_freq == 0):
                logger.info("saving checkpoint")
                torch.save(model.state_dict(), os.path.join(model_dir, 'checkpoint_{}.pth'.format(epoch+1)))

            # save the weights of the model if current validation error is lowest
            if val_metrics['loss'][-1] == min(val_metrics['loss']):
                logger.info("saving best model")
                torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth'))
            
        # test model
        print("test")
        logger.info("testing model..")
        model.eval()
        fold_metrics = test(args, dataloader, model, device)

        # save metrics
        with open(os.path.join(model_dir, "test.json"), "w") as outfile:
            json.dump(fold_metrics, outfile)
    
    print("\n done")
