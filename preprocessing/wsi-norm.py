import argparse
import io
from pathlib import Path
import sys
import shutil
from typing import Dict, Tuple
from concurrent import futures
# from urllib.parse import urlparse
import warnings
import glob
import logging
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-o', '--output-path', type=Path, required=True,
                        help='Path to save results to.')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--wsi-dir', metavar='DIR', type=Path,
                        help='Path of where the whole-slide images are.')
    source_group.add_argument('--cached-slide-dir', metavar='DIR', type=Path,
                        help='Existing cache with <slide>/slide.jpg before stain normalization; preserve its pixel resolution.')
    parser.add_argument('-m', '--model', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
    parser.add_argument('--cache-dir', type=Path, default=None,
        help='Directory to cache extracted features etc. in.')
    parser.add_argument(
        '--normalization-template', type=Path,
        default=Path(__file__).resolve().parent/'normalization_template.jpg',
        help='Macenko reference image (defaults to the repository template).')
    parser.add_argument('--cached-black-threshold', type=int, default=8)
    parser.add_argument('--cached-min-occupancy', type=float, default=0.05)
    parser.add_argument('--min-tissue-fraction', type=float, default=0.0)
    parser.add_argument('--min-edge-density', type=float, default=0.0)
    parser.add_argument('--min-focus', type=float, default=0.0)
    parser.add_argument('--augmented-repetitions', type=int, default=0,
                        help='Additional random image views per tile; default 0.')
    parser.add_argument('--normalization-scope', choices=('tile', 'slide'), default='tile',
                        help='Legacy per-tile scaling or one common stain scale per slide; use separate output caches.')
    parser.add_argument('--concentration-brightness', choices=('original', 'standardized'), default=None,
                        help='RGB brightness used for concentration fitting; defaults to original for tile scope and standardized for slide scope.')
    parser.add_argument(
        '--max-tiles', type=int, default=0,
        help='Keep at most this many highest-QC tiles per slide; 0 keeps all.')
    parser.add_argument(
        '--delete-source-after-processing', action='store_true',
        help='Delete source SVS files only after an explicit opt-in.')
    # parser.add_argument('--cache-dir', type=Path, default=None,
    #                     help='Directory to cache extracted features etc. in.')

    args = parser.parse_args()
    if args.cached_slide_dir and args.delete_source_after_processing:
        parser.error('--delete-source-after-processing cannot be used with --cached-slide-dir')




# if (p := './RetCCL') not in sys.path:
#     sys.path = [p] + sys.path
# import ResNet
# import torch.nn as nn
# import torch
# from torchvision import transforms
import os
from matplotlib import pyplot as plt
import openslide
from tqdm import tqdm
import numpy as np
# from fastai.vision.all import load_learner
# from pyzstd import ZstdFile
import PIL
import stainNorm_Macenko
import cv2
from numba import jit
from wsi_cache_tiles import extract_cached_normalized_tiles, filter_tiles_by_qc

# supress DecompressionBombWarning: yes, our files are really that big (‘-’*)
PIL.Image.MAX_IMAGE_PIXELS = None


def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    pos = tuple(int(v) for v in pos)
    stride = tuple(int(v) for v in stride)
    target_size = tuple(int(v) for v in target_size)
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in tiles to
    #  1. parallelize the loading process
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    steps = 8
    stride = np.ceil(np.array(slide.dimensions)/steps).astype(int)
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Read slide MPP of {slide_mpp} from meta-data")
    except:
        print(f"Error: couldn't load MPP from slide!")
        return None
    tile_target_size = np.round(stride*slide_mpp/target_mpp).astype(int)
    #changed max amount of threads used
    read_workers = int(os.environ.get("WSI_READ_THREADS", "122"))
    with futures.ThreadPoolExecutor(read_workers) as executor:
        # map from future to its (row, col) index
        future_coords: Dict[futures.Future, Tuple[int, int]] = {}
        for i in range(steps):  # row
            for j in range(steps):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_target_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an image as soon as they are loaded
        im = np.zeros((*(tile_target_size*steps)[::-1], 3), dtype=np.uint8)
        for tile_future in tqdm(futures.as_completed(future_coords), total=steps*steps, desc='Reading WSI tiles', leave=False):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_target_size * (j, i)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile

    return im

import time
from datetime import timedelta

#xiyue wang fex
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from marugoto.marugoto.extract.extract import extract_features_
from marugoto.marugoto.extract.xiyue_wang.RetCLL import ResNet
from concurrent_canny_rejection import reject_background
from PIL import Image

# %%

def extract_xiyuewang_features_(norm_wsi_img: PIL.Image, wsi_name: str, coords: list, checkpoint_path: str, outdir: Path, **kwargs):
    """Extracts features from slide tiles.
    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL>.
    """
   # calculate checksum of model
    sha256 = hashlib.sha256()
    with open(checkpoint_path, 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert sha256.hexdigest() == '931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff'

    model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    #put the model on the CPU for HPC
    pretext_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)

    if torch.cuda.is_available():
        model=model.cuda()
    
    #TODO: replace slide_tile_paths with the actual tiles which are in memory
    return extract_features_(norm_wsi_img=norm_wsi_img, wsi_name=wsi_name, coords=coords, model=model, outdir=outdir, model_name='xiyuewang-retcll-931956f3', **kwargs) #removed model.cuda()



def get_raw_tile_list(I_shape: tuple, bg_reject_array: np.array, rejected_tile_array: np.array, patch_shapes: np.array):
    canny_output_array=[]
    for i in range(len(bg_reject_array)):
        if not rejected_tile_array[i]:
            canny_output_array.append(np.array(bg_reject_array[i]))

    canny_img = Image.new("RGB", (I_shape[1], I_shape[0]))
    coords_list=[]
    i_range = range(I_shape[0]//patch_shapes[0][0])
    j_range = range(I_shape[1]//patch_shapes[0][1])

    for i in i_range:
        for j in j_range:
            idx = i*len(j_range) + j
            canny_img.paste(Image.fromarray(np.array(bg_reject_array[idx])), (j*patch_shapes[idx][1], 
            i*patch_shapes[idx][0],j*patch_shapes[idx][1]+patch_shapes[idx][1],i*patch_shapes[idx][0]+patch_shapes[idx][0]))
            
            if not rejected_tile_array[idx]:
                coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))

    return canny_img, canny_output_array, coords_list

if __name__ == "__main__":
    if args.cache_dir is None:
        raise ValueError('--cache-dir is required for WSI normalization')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logdir = args.cache_dir/'logfile'
    logging.basicConfig(filename=logdir, force=True)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Stored logfile in {logdir}')
    #init the Macenko normaliser
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    has_gpu=torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}")
    norm=True
    if norm:
        print("\nInitialising Macenko normaliser...")
        target = cv2.imread(str(args.normalization_template))
        if target is None:
            raise FileNotFoundError(
                f'Could not read normalization template: {args.normalization_template}'
            )
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        normalizer = stainNorm_Macenko.Normalizer(concentration_scope=args.normalization_scope,
                                                 concentration_brightness=args.concentration_brightness)
        normalizer.fit(target)
        logging.info('Running WSI to normalised feature extraction...')
    else:
        logging.info('Running WSI to raw feature extraction...')
    # norm = MacenkoNormalizer()
    # norm.fit(target)
    total_start_time = time.time()
    svs_dir = (sorted(args.cached_slide_dir.glob('*/slide.jpg')) if args.cached_slide_dir
               else glob.glob(f"{args.wsi_dir}/*.svs"))
    for slide_url in (progress := tqdm(svs_dir, leave=False)):
        # breakpoint()
        slide_name = Path(slide_url).parent.name if args.cached_slide_dir else Path(slide_url).stem
        progress.set_description(slide_name)
        slide_cache_dir = args.cache_dir/slide_name
        slide_cache_dir.mkdir(parents=True, exist_ok=True)
        normalization_record = slide_cache_dir/'normalization.json'
        cached_settings = json.loads(normalization_record.read_text()) if normalization_record.exists() else {}
        cached_scope = cached_settings.get('concentration_scope', 'tile')
        cached_brightness = cached_settings.get('concentration_brightness',
                                                'standardized' if cached_scope == 'slide' else 'original')
        if ((slide_cache_dir/'norm_slide.jpg').exists() or (args.cache_dir/f'{slide_name}.h5').exists()) and (
                cached_scope != args.normalization_scope or cached_brightness != normalizer.concentration_brightness):
            raise ValueError(f'{slide_name}: cached normalization is {cached_scope}, requested {args.normalization_scope}; cached brightness is {cached_brightness}, requested {normalizer.concentration_brightness}; use a separate --cache-dir')
        # print('\n')
        # print((f'{args.cache_dir}/{slide_name}.h5'))
        # print((os.path.exists((f'{args.cache_dir}/{slide_name}.h5'))))
        if not (os.path.exists((f'{args.cache_dir}/{slide_name}.h5'))):
            # Load WSI as one image
            if (slide_jpg := slide_cache_dir/'norm_slide.jpg').exists():
                img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
                image_array = np.array(img_norm_wsi_jpg)
                canny_patch_list, coords_list = extract_cached_normalized_tiles(
                    image_array,
                    tile_size=224,
                    black_threshold=args.cached_black_threshold,
                    min_occupancy_fraction=args.cached_min_occupancy,
                )
                total = (image_array.shape[0] // 224) * (image_array.shape[1] // 224)
                print(
                    "Loaded normalised canny image, "
                    f"{len(canny_patch_list)}/{total} tiles remain"
                )
            else:
                logging.info(f"\nLoading {slide_name}")
                #measure time performance
                start_time = time.time()
                if args.cached_slide_dir:
                    with Image.open(slide_url) as source_image:
                        slide_array = np.asarray(source_image.convert('RGB'))
                else:
                    try:
                        slide = openslide.OpenSlide(str(slide_url))
                    except Exception as e:
                        logging.error(f"Failed loading {slide_name}, error: {e}")
                        continue
                    try:
                        slide_array = load_slide(slide)
                    finally:
                        slide.close()
                if slide_array is None:
                    print(f"Skipping {slide_url} due to missing MPP...")
                    continue
                #save raw .svs jpg
                if args.cached_slide_dir:
                    cached_raw = slide_cache_dir/'slide.jpg'
                    if not cached_raw.exists():
                        cached_raw.symlink_to(Path(slide_url).resolve())
                else:
                    (Image.fromarray(slide_array)).save(f'{slide_cache_dir}/slide.jpg')

                #remove .SVS from memory (couple GB)
                
                print("\n--- Loaded slide: %s seconds ---" % (time.time() - start_time))
                #########################

                #########################
                #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=(224,224), step=224, outdir=args.cache_dir, save_tiles=False)

                #measure time performance
                start_time = time.time()
                #pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
                if norm:
                    logging.info(f"Normalising {slide_name}...")
                    canny_img, img_norm_wsi_jpg, canny_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, rejected_tile_array, patch_shapes)
                    print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                    img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg
                    normalization_record.write_text(json.dumps({
                        'concentration_scope': args.normalization_scope,
                        'concentration_brightness': normalizer.concentration_brightness,
                        'source': str(slide_url),
                        'source_kind': 'cached_slide_jpeg' if args.cached_slide_dir else 'svs',
                    }, indent=2)+'\n')

                else:
                    canny_img, canny_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array, rejected_tile_array, patch_shapes)

                print("Saving Canny background rejected image...")
                canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')
                # norm_wsi_jpg = norm.transform(np.array(slide_array))
                
                #remove original slide jpg from memory
                del slide_array
                #print(f"Deleting slide {slide_name} from local folder...")
                #os.remove(str(slide_url))


                # img_norm_wsi_jpg = PIL.Image.fromarray(norm_wsi_jpg)
                img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

            canny_patch_list, coords_list, tile_qc = filter_tiles_by_qc(
                np.asarray(canny_patch_list),
                coords_list,
                min_tissue_fraction=args.min_tissue_fraction,
                min_edge_density=args.min_edge_density,
                min_focus=args.min_focus,
                max_tiles=args.max_tiles or None,
                black_threshold=args.cached_black_threshold,
            )
            logging.info(
                'QC retained %d tiles for %s (mean score %.4f)',
                len(canny_patch_list),
                slide_name,
                float(tile_qc['tile_qc_score'].mean()) if len(canny_patch_list) else float('nan'),
            )
            if len(canny_patch_list) == 0:
                logging.warning('No tiles passed QC for %s; skipping', slide_name)
                continue

            print(f"Extracting xiyue-wang macenko features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            extract_xiyuewang_features_(
                norm_wsi_img=np.asarray(canny_patch_list),
                wsi_name=slide_name,
                coords=coords_list,
                checkpoint_path=args.model,
                outdir=slide_cache_dir,
                tile_qc=tile_qc,
                augmented_repetitions=args.augmented_repetitions,
            )
            print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            #########################
            feature_output = args.cache_dir/f'{slide_name}.h5'
            if args.delete_source_after_processing and feature_output.exists():
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))
            elif args.delete_source_after_processing:
                logging.warning(
                    'Retaining %s because the feature HDF5 was not created', slide_url
                )

        else:
            print(f"{slide_name}.h5 already exists. Skipping...")
            if args.delete_source_after_processing:
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))

    print(f"--- End-to-end processing time of {len(svs_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")
