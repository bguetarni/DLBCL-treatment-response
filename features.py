import os, argparse, glob, math
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
import huggingface_hub
from torchvision import transforms, models
from conch.open_clip_custom import create_model_from_pretrained

import HIPT


def conch(**kwargs):
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=kwargs["hf_token"])
    return model, preprocess

def gigapath(**kwargs):
    gigapath = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return gigapath, preprocess

def hipt(**kwargs):
    model = HIPT.get_vit256(kwargs["weights"])

    # for normalization values see https://github.com/mahmoodlab/HIPT/issues/6
    # or also https://github.com/mahmoodlab/HIPT/blob/780fafaed2e5b112bc1ed6e78852af1fe6714342/HIPT_4K/hipt_model_utils.py#L111
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    return model, preprocess

def resnet(**kwargs):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return model, preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', type=str, required=True, help='token for HuggingFace authorization')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset with patches CSV')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save features')
    parser.add_argument('--model', type=str, required=True, choices=["conch", "gigapath", "hipt", "resnet"], help='which model to extract features')
    parser.add_argument('--weights', type=str, default=None, help='path to weights if necessary')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for model inference')
    parser.add_argument('--gpu', type=str, default='', help='GPU to use (e.g. 0)')
    args = parser.parse_args()

    if args.model == 'hipt':
        assert not args.weights is None, 'To use HIPT model, provide path to ViT-256 model weights'

    # set torch device
    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    print('device: ', device)

    print("patches are taken from {}".format(args.dataset))
    print("features will be saved in {}".format(os.path.join(args.output, args.model)))

    # Hugging Face API token
    huggingface_hub.login(token=args.hf_token)

    # load model and preprocess routine
    model, preprocess = eval(args.model)(**vars(args))
    model = model.eval().to(device=device)

    for slide in tqdm(glob.glob(os.path.join(args.dataset, "*")), ncols=50):
        base, name = os.path.split(slide)

        output_dir = os.path.join(args.output, args.model, name)
        os.makedirs(output_dir, exist_ok=True)

        # read patches
        patches = glob.glob(os.path.join(slide, "*.png"))

        # get features
        features = []
        with torch.no_grad():
            for x in np.array_split(patches, math.ceil(len(patches)/args.batch_size)):

                # preprocess
                x = map(Image.open, x)
                x = map(preprocess, x)
                x = torch.stack(list(x), dim=0)

                # straigthforward pass
                x = x.to(device=device)
                if args.model == "conch":
                    fts = model.encode_image(x)
                else:
                    fts = model(x)
                
                features.append(fts.to(device='cpu'))
        
        features = torch.cat(features, dim=0)

        # save features
        for path, fts in zip(patches, features):
            file_name = os.path.split(path)[1]
            file_name = os.path.splitext(file_name)[0]
            # IMPORTANT: clone has to be called to not save the whole tensor but the current slice
            torch.save(fts.clone(), os.path.join(output_dir, "{}.pt".format(file_name)))
