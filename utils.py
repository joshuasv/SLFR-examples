import os
import json
from shutil import rmtree

import yaml
import zipfile

def read_json(json_fpath):
    with open(json_fpath, 'r') as fid:
        return json.load(fid)
    
def read_yaml(yaml_fpath):
    config_name = os.path.basename(yaml_fpath).split('.')[0]
    with open(yaml_fpath, 'r') as fid:
        toret = yaml.load(fid, yaml.SafeLoader)
        toret['config_name'] = config_name
        return toret
    
def ensure_clean_dir(dpath, assume_yes=False):
    if os.path.isdir(dpath):
        print(f'dir {dpath} already exists')
        answer = 'y' if assume_yes else input('delete it? [y]/n: ')
        if answer.lower() in ('y', ''):
            rmtree(dpath)
            print(f'dir {dpath} removed')
            os.makedirs(dpath)
            return True
        else:
            print(f'dir {dpath} not removed')
            return False
    else:
        os.makedirs(dpath)
        return True

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def zip_file(file_path, zip_path, name):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(file_path, arcname=name)