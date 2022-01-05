import numpy as np
import pandas as pd

def extract_bb(annot):
    output = {}
    # Extract BB if it's not a negative example
    if not pd.isnull(annot):
        output['xmin'] = annot['x']
        output['ymin'] = annot['y']
        output['bb_width'] = annot['width']
        output['bb_height'] = annot['height']
        
    # otherwise set default values
    else:
        output['xmin'] = -1
        output['ymin'] = -1
        output['bb_width'] = -1
        output['bb_height'] = -1
    
    return output

def extract_annotations(df):
    """Take the bounding box format provided by the competition, transform
    to individual columns. Null boxes (images without boxes) set all valuaes
    to -1 for xmin, ymin, bounding box width, bounding box height.

    df: Dataframe with a single annotation dictionary per line (or null value
    when there is no annotation for an image). If an image has multiple bounding
    boxes, each appears on a unique line.
    """
    
    df['annotations_processed'] = df['annotations'].map(extract_bb)

    df['xmin'] = df['annotations_processed'].map(lambda x: x['xmin'])
    df['ymin'] = df['annotations_processed'].map(lambda x: x['ymin'])
    df['bb_width'] = df['annotations_processed'].map(lambda x: x['bb_width'])
    df['bb_height'] = df['annotations_processed'].map(lambda x: x['bb_height'])

    return df