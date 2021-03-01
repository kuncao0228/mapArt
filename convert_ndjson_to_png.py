import argparse
import logging
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import multiprocessing


matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO)

OBJECT_LIST = ['airplane', 'apple', 'angel', 'ant', 'anvil', 'axe', 'backpack',
                            'banana', 'bandage', 'baseball', 'basketball', 'bear',
                            'bed', 'bear', 'bee', 'bicycle', 'binoculars', 'bird',
                            'book', 'boomerang', 'bowtie', 'brain', 'bus', 'butterfly',
                            'cactus', 'camel', 'candle', 'cannon', 'car', 'castle', 'cat', 'cello', 'chair',
                            'clarinet', 'computer', 'cow', 'crab', 'crocodile', 'crown', 
                            'diamond', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'clock', 
                            'drums', 'duck', 'elephant', 'eye', 'fish', 'flamingo', 'flower', 'frog',
                            'giraffe', 'guitar', 'fish']
IMAGE_BASE_DIR = ''
OBJECT_LIMIT = 1000
N_PROCESSES = 1
DRAWING_LIST_SHARED = []


def save_image(stroke_list,
               image_name,
               directory,
               width=100,
               my_dpi=50):
    """Draw with a list of strokes and save the image as png"""
    fig = plt.figure(figsize=(width / my_dpi, width / my_dpi),
                     dpi=my_dpi,
                     frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    min_x = 255
    max_x = 0
    min_y = 255
    max_y = 0
    for stroke in stroke_list:
        x = stroke[0]
        y = stroke[1]
        min_x = min(min_x, min(x))
        max_x = max(max_x, max(x))
        min_y = min(min_y, min(y))
        max_y = max(max_y, max(y))
        plt.plot(x, y, c='k')

    plt.xlim((min_x, max_x))
    plt.ylim((min_y, max_y))
    plt.gca().invert_yaxis()
    

    plt.savefig(os.path.join(directory, image_name))
    plt.close(fig)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_limit', type=int, default=OBJECT_LIMIT)
    parser.add_argument('--n_processes', type=str, default=N_PROCESSES)
    parser.add_argument('--image_base_dir', type=str, default=IMAGE_BASE_DIR)
    return parser.parse_args()


def save_image_single_process(arg):
    arg = arg.split('|')
    image_id, object_name = int(arg[0]), arg[1]
    
    save_image(DRAWING_LIST_SHARED[image_id],
               "{}.png".format(image_id),
               os.path.join(IMAGE_BASE_DIR, object_name))


if __name__ == '__main__':
    args = parse_args()

    
    IMAGE_BASE_DIR = args.image_base_dir
    # Create subfolders if not exist
    for object_name in OBJECT_LIST:
        subfolder_name = os.path.join(args.image_base_dir, object_name)
        
        if not os.path.exists(subfolder_name):
            print("Making Directory " + subfolder_name)
            os.makedirs(subfolder_name)

    for object_name in OBJECT_LIST:

        logging.info('Working on object class {}.'.format(object_name))

        f_name = os.path.join(args.image_base_dir, object_name) + '.ndjson'

        # Read ndjson file
        with open(f_name, 'r') as f:
            raw_file = f.read()
        drawing_json = raw_file.split('\n')

        # Convert ndjson into a list
        DRAWING_LIST_SHARED = []
        for d in drawing_json[:args.object_limit]:
            try:
                DRAWING_LIST_SHARED.append(json.loads(d)['drawing'])
            except:
                pass

        pool = multiprocessing.Pool(processes=int(args.n_processes))
        arg_list = ['{}|{}'.format(i, object_name) for i in range(len(DRAWING_LIST_SHARED))]
        result_list = pool.map(save_image_single_process, arg_list)
        pool.close()
        pool.join()