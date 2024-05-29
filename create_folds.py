import os, glob, argparse
import random
import pandas
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="path to the dataset with PNG tiles")
    parser.add_argument('--output', type=str, required=True, help="location to save CSV file")
    parser.add_argument('--k', type=int, required=True, help="number of folds")
    parser.add_argument('--labels', type=str, required=True, help='path to treatment response file, for class')
    args = parser.parse_args()

    treatment_response = pandas.read_csv(args.labels).set_index('slide_id')['treatment_response'].to_dict()

    data = glob.glob(os.path.join(args.dataset, "*"))
    
    confirmed = False
    while not confirmed:
        random.shuffle(data)

        folds = np.array_split(data, args.k)

        for i, f in enumerate(folds):
            classes = []
            for slide in f:
                label = int(treatment_response[slide])
                label = "pos" if label == 1 else "neg"
                n = len(glob.glob(os.path.join(slide, "*")))
                classes.extend([label for _ in range(n)])
            
            print("fold ", i)
            
            classes = {k: classes.count(k) for k in set(classes)}
            for k, v in classes.items():
                print("\t {}: {} ({}%)".format(k, v, round(100*v / sum(classes.values()))))
        
        confirmed = input("confirm these folds ? (yes/no)")
        confirmed = True if confirmed == "yes" else False
    
    df = []
    for i, f in enumerate(folds):
        for slide in f:
            df.append({'slide': os.path.split(slide)[1], 'fold': i})

    df = pandas.DataFrame(df)
    df.to_csv(args.output)
    print("done")
