import argparse
import wget
from wget import bar_thermometer
import os
import configparser
import logging
import tarfile
from shutil import move, rmtree
import zipfile

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("DatasetPrepare")


def create_path(path: str):
    now_path = ''
    if path.startswith('/'):
        now_path = '/'
    for path_i in path.split('/'):
        now_path += path_i + '/'
        if not os.path.exists(now_path):
            os.mkdir(now_path)


def load_config(config_path: str):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser


def download(url: str):
    target_filename = url.split('/')[-1]
    target_path = 'tmp/' + target_filename
    if not os.path.exists(target_path):
        log.info(f'Download {url} to {target_path}')
        wget.download(url, target_path, bar=bar_thermometer)
    else:
        log.warning(f'File {url} is already exists on {target_path}')

    return target_path


def extract_only_path(tar: tarfile, path: str, destination: str):
    members = [tarinfo for tarinfo in tar.getmembers()
                      if tarinfo.name.startswith(path)]
    log.info(f"Extracting {len(members)} files from {path}")
    tar.extractall(destination, members=members)


def unpack_tar_dataset(filename: str, dest_dir: str, voc_version: str = None):
    with tarfile.open(filename) as tar:
        if voc_version is not None:
            dest_path = os.path.join(dest_dir, voc_version)
            create_path(dest_path)
            extract_only_path(tar, f'VOCdevkit/{voc_version}/Annotations/', dest_path)
            extract_only_path(tar, f'VOCdevkit/{voc_version}/JPEGImages/', dest_path)
            voc_to_root_folder(f'{dest_dir}/{voc_version}/VOCdevkit/{voc_version}', dest_dir)
            rmtree(f'{dest_dir}/{voc_version}')


def unpack_coco_dataset(filename: str, dest_dir: str):
    with zipfile.ZipFile(filename, 'r') as archive:
        dest_path = os.path.join(os.getcwd(), dest_dir)
        create_path(dest_path)
        archive.extractall(dest_path)


def voc_to_root_folder(voc_path: str, voc_root_folder: str):
    print(voc_path)
    directories = [os.path.join(voc_path, path) for path in os.listdir(voc_path)]
    for directory in directories:
        move(directory, os.path.join(voc_root_folder, directory.split('/')[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tool for load datasets (VOC, COCO)")
    parser.add_argument('-v', '--voc', help='Load VOC dataset', dest='voc_dn', action='store_true', default=False)
    parser.add_argument('-c', '--coco', help='Load COCO dataset', dest='coco_dn', action='store_true', default=False)
    parser.add_argument('-t', '--transform', help='Transform dataset to one type', dest='tranform', default='None')
    args = parser.parse_args()

    config = load_config('url_config.ini')
    create_path('tmp')

    if args.voc_dn:
        create_path('voc')
        files = []
        for url in config['VOC']['data'][1:-1].split(' '):
            url = url.strip('\'')
            files.append(download(url))
        print(files)
        unpack_tar_dataset(files[0], 'voc', 'VOC2012')

    if args.coco_dn:
        create_path('coco')
        files = []
        for url in config['COCO']['data'][1:-1].split(' '):
            url = url.strip('\'')
            files.append(download(url))
        for url in config['COCO']['image'][1:-1].split(' '):
            url = url.strip('\'')
            files.append(download(url))
        print(files)
        for file in files:
            unpack_coco_dataset(file, 'coco')