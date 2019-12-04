import json
import os
import sys

import requests
import yaml

from ScientificTopics import LDAInfer

default_model_url = 'https://github.com/hhaoyan/ScientificTopics/' \
                    'raw/master/data_20190830/default_model.yaml'
default_model_name = 'data_20190830'


def _data_dir():
    data_dir = os.path.expanduser('~/.scientific_topics')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir


def _download_file(url, path):
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
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
    data_dir = _data_dir()

    print("Fetching metadata...")
    metadata_req = requests.get(metadata_url)
    metadata = yaml.safe_load(metadata_req.content)

    model_home = os.path.join(data_dir, metadata['model_name'])
    if os.path.exists(os.path.join(model_home, 'downloaded')):
        return
    try:
        os.mkdir(model_home)
    except FileExistsError:
        pass

    with open(os.path.join(model_home, 'meta.json'), 'w') as f:
        json.dump(metadata, f)

    for download in metadata['data_files']:
        print("Fetching file %s from %s..." % (
            download['file_fn'], os.path.join(model_home, download['file_fn'])), end='')
        sys.stdout.flush()
        print("%d bytes" % _download_file(
            download['file_url'],
            os.path.join(model_home, download['file_fn'])))

    with open(os.path.join(model_home, 'downloaded'), 'w') as f:
        f.write('downloaded')


def load_model(model_name=default_model_name, topic_model=0):
    data_dir = _data_dir()
    model_home = os.path.join(data_dir, model_name)
    if not os.path.exists(os.path.join(model_home, 'downloaded')):
        raise FileNotFoundError('No model found at %s, did you use '
                                '"python -m ScientificTopics.download"?' % model_home)

    with open(os.path.join(model_home, 'meta.json')) as f:
        metadata = json.load(f)

    if isinstance(topic_model, int):
        topic_model = metadata['topic_models'][topic_model]['name']
    try:
        topic_model = next(x for x in metadata['topic_models'] if x['name'] == topic_model)
    except StopIteration:
        raise NameError('No such topic model: ' + topic_model)

    punkt_file = os.path.join(model_home, metadata['punkt_data_fn'])
    spm_file = os.path.join(model_home, metadata['sentencepiece_data_fn'])
    stopwords = os.path.join(model_home, metadata['stopwords_data_fn'])
    topic_model_root = os.path.join(model_home, topic_model['root'])
    return LDAInfer(
        topic_model_root, punkt_model=punkt_file, spm_model=spm_file, stopwords=stopwords,
        beta=topic_model['beta'], alpha=topic_model['alpha'], num_vocab=metadata['vocab_size'])

if __name__ == '__main__':
    download_model()
