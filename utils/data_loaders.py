from itertools import chain
import os, glob, random
import numpy as np
import pandas
import torch
from sklearn.utils.class_weight import compute_class_weight

class Dataset:
    def __init__(self, args, fold_id, **kwargs):
        """
        Dataset for treatment response that contains train and validation splits

        args:
            args (Namespace): main script arguments
            fold_id (int): fold to consider as test
        """

        self.args = args

        # mode
        self.mode = "train"
        
        # load labels and folds
        self.fold_split = pandas.read_csv(args.fold).set_index('slide')['fold'].to_dict()
        self.labels = pandas.read_csv(args.labels).set_index('slide_id')['treatment_response'].to_dict()

        # load data
        self.data = {"train": {}, "test": {}}
        for slide in self.fold_split.keys():

            # get slide label
            label = int(self.labels[slide])

            # gather WSI features
            features = self.load_features(args, slide)

            # add to appropriate split
            if self.fold_split[slide] == fold_id:
                self.data["test"].update({slide: (features, label)})
            else:
                self.data["train"].update({slide: (features, label)})
            
    def load_features(self, args, slide):
        """
        Load a slide features, considering feature extractors in arguments

        args:
            args (Namespace): args from main script
            slide (str): slide to load features

        return a dict of features (each feature extractor in a dict)
        {
            feature extractor 1: [path/to/tile, ...],
            feature extractor 2: [path/to/tile, ...],
        }
        """

        data = dict()
        for ftext in os.listdir(args.dataset):
            if ftext in args.extractor:
                fts = glob.glob(os.path.join(args.dataset, ftext, slide, '*.pt'))

                data.update({ftext: fts})
        
        return data
    
    def __len__(self):
        if self.mode == "train":
            return len(self.data["train"])
        else:
            return len(self.data["test"])
    
    def __iter__(self):
        if self.mode in ["train", "valid"]:
            splitkey = "train" if self.mode == "train" else "test"
            self.iterator_data = list(self.data[splitkey].values())
        else:
            self.iterator_data = list(self.data["test"].values())

        # shuffle list of data
        random.shuffle(self.iterator_data)

        # we start at -1 because the counter is updated  at beginning of each iteration
        self.iterator_count = -1
        
        return self
    
    def __next__(self):

        self.iterator_count += 1

        if self.iterator_count < len(self.iterator_data) - 1:
            x, y = self.iterator_data[self.iterator_count]
            if self.mode in ["train", "valid"]:
                if isinstance(self.args.limit_per_slide, int) and self.args.limit_per_slide > 0:
                    # number of tiles in the slide
                    N = len(list(x.values())[0])
                    
                    # randomly select tiles
                    if N > self.args.limit_per_slide:
                        # be careful to select the same tiles for all feature extractor
                        idx = random.sample(range(N), k=self.args.limit_per_slide)
                        x = {k: [v[i] for i in idx] for k, v in x.items()}
            
            # load features
            x = {k: torch.stack(list(map(torch.load, v)), dim=0) for k, v in x.items()}

            return x, y
        
        # reinitialise counter and return StopIteration when run out of data
        self.iterator_count = -1
        return StopIteration

    def train(self):
        self.change_mode("train")

    def valid(self):
        self.change_mode("valid")

    def test(self):
        self.change_mode("test")
    
    def change_mode(self, mode):
        self.mode = mode
        self.iterator_data = None
        self.iterator_count = -1
    
    def class_weights(self):
        """
        Compute class weights based on sklearn.utils.class_weight.compute_class_weight
        """

        # initialize data
        _ = iter(self)

        # get labels
        _, y = zip(*self.iterator_data)

        # return class weights
        return compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
