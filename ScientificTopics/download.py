import json
import os

import requests
import yaml

default_model_url = 'https://github.com/hhaoyan/ScientificTopics/' \
                    'raw/master/data_20190830/default_model.yaml'


def _make_dir():
    data_dir = os.path.expanduser('~/.scientific_topics')
    return data_dir


def _download_file(url, path):
    with requests.get(url, stream=True) as req:
        size = 0
        req.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
    return size


def download_model(metadata_url=default_model_url):
    data_dir = _make_dir()

    print("Fetching metadata...")
    metadata_req = requests.get(metadata_url)
    metadata = yaml.safe_load(metadata_req.content)

    model_home = os.path.join(data_dir, metadata['model_name'])
    if not os.path.exists(model_home):
        os.mkdir(model_home)
    if os.path.exists(os.path.join(model_home, 'downloaded')):
        return

    with open(os.path.join(model_home, 'meta.json'), 'w') as f:
        json.dump(metadata, f)

    for download in metadata['data_files']:
        print("Fetching file %s from %s..." % (download['file_fn'], download['file_url']), end='')
        print("%d bytes" % _download_file(download['file_url'], download['file_fn']))

    with open(os.path.join(model_home, 'downloaded'), 'w') as f:
        f.write('downloaded')

