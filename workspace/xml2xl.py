import os
import re
import xml.etree.ElementTree as et
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

# load and parse the file
src_dir = "Y:/ARod/4F/20200324_No5_CamA/preview_CamA_downsample/layer_0"
src_xml = glob(os.path.join(src_dir, "*.xml"))[0]
xtree = et.parse(src_xml)
xroot = xtree.getroot()

# /Extends/Image
xroot = xroot.findall('Extends')[0]

# pull out attributes
info = []
for node in xroot:
    info.append(dict(node.attrib))

# form DataFrame dict format
records = defaultdict(list)
for record in info:
    for key, value in record.items():
        records[key].append(value)

# convert to actual DataFrame
df = pd.DataFrame.from_dict(records)
 
# assign proper dtype
dtype_table = {k: np.float32 for k in df.columns}
# .. except filename
dtype_table['Filename'] = str
df = df.astype(dtype_table)

# calculate actual size
for ax in ('X', 'Y', 'Z'):
    df[ax] = (df[f'Max{ax}'] - df[f'Min{ax}']).round().astype(int)  # we know it is discrete

# calculate tile index
index = defaultdict(list)
pattern = r'_(\d+)_x(\d{3})_y(\d{3})'
for tile_name in df['Filename']:
    matches = re.search(pattern, tile_name)
    assert matches is not None, "unable to interpret filename"

    for ax, value in zip(('iZ', 'iX', 'iY'), matches.groups()):
        value = int(value)
        index[ax].append(value)
# write back
for key, value in index.items():
   df[key] = pd.Series(value)

print('>> BEFORE')
print(df)
print()

# convert to full resolution coordinate
for ax in ('X', 'Y'):
    df[f'Max{ax}'] = (df[f'Max{ax}'] * 2048 / df['X'])
    df[f'Min{ax}'] = (df[f'Min{ax}'] * 2048 / df['X'])

# replace 'Filename'
for 




print('>> AFTER')
print(df)
print()

# update attributes
for node in xroot:
    filename = node.get('Filename')
    
    # lookup updated attributes
    attributes = df[df['Filename'] == filename]
    assert len(attributes) == 1, 'duplicated filename'
    new_attrs = attributes.iloc[0].to_dict()

    # replace
    for key in node.attrib.keys():
        value = str(new_attrs[key])
        node.set(key, value)

# generate result
fname, fext = os.path.splitext(src_xml)
dst_xml = f'{fname}_modify{fext}'
xtree.write(dst_xml)
