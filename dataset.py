import os, argparse, math
import numpy as np
import openslide
import pandas
from tqdm import tqdm

import utils.dataset as dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi', type=str, required=True, help='directory with WSI files')
    parser.add_argument('--output', type=str, required=True, help='path to directory save results')
    parser.add_argument('--csvfile', type=str, required=True, help='path to treatment response file')
    parser.add_argument('--level', type=int, required=True, help='WSI level to use')
    parser.add_argument('--psize', type=int, required=True, help='patch size')
    parser.add_argument('--overlap', type=int, default=0, help='overlapping between two patches (pixels)')
    parser.add_argument('--mask_level', type=int, default=5, help='level for tissue mask creation')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for tissue')
    args = parser.parse_args()

    assert args.overlap < args.psize, 'ERROR: overlap between two patches must be smaller than patch size'

    treatment_response = pandas.read_csv(args.csvfile).set_index('slide_id')['treatment_response'].to_dict()

    for file_ in tqdm(os.listdir(args.wsi), ncols=50):

        slide = os.path.splitext(file_)[0]

        if (not int(slide) in list(treatment_response.keys())) or math.isnan(treatment_response[int(slide)]):
            continue

        # load slide file
        wsi = openslide.OpenSlide(os.path.join(args.wsi, file_))

        # handle case where mask_level is not among whole slide available levels
        mask_level = min(args.mask_level, len(wsi.level_dimensions)-1)
        
        # create tissue and annotation mask
        mask = dataset.get_tissue_mask_hsv(wsi, mask_level)

        # create output directory
        output_dir = os.path.join(args.output, slide)
        os.makedirs(output_dir, exist_ok=True)

        w, h = wsi.level_dimensions[args.level]
        for x in range(0, (w - args.psize) + 1, args.psize - args.overlap):
            for y in range(0, (h - args.psize) + 1, args.psize - args.overlap):
                
                # retrieve part of mask belonging to tile
                tile_mask = dataset.get_tile_mask(wsi, args.level, mask, mask_level, x, y, args.psize)
                
                # sometimes we step outside the mask due to coordinates scaling
                if tile_mask.size == 0:
                    continue
                
                # check mask cover enough patch
                ratio = np.count_nonzero(tile_mask) / tile_mask.size
                if ratio < args.threshold:
                    continue
                
                # scale coordinates to 0 level and add information to csv data
                x_patch_scaled, y_patch_scaled = dataset.scale_coordinates(wsi, (x, y), args.level, 0)
                
                # read patch
                img = wsi.read_region(location=(x_patch_scaled,y_patch_scaled), level=args.level, size=(args.psize,args.psize)).convert('RGB')

                # save patch
                img.save(os.path.join(output_dir, "{}_{}.png".format(x, y)))
