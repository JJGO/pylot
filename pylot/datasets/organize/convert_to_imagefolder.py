#!/usr/bin/env python

# Script for converting from original TinyImagenet format to
# Pytorch ImageFolder format.

import pathlib
import shutil


def main():
    val = pathlib.Path('val')
    valorig = pathlib.Path('val_orig')
    valimages = valorig / 'images'
    annotations = valorig / 'val_annotations.txt'

    with open(annotations) as infile:
        filename_category_map = {}
        for line in infile:
            filename, category = line.split()[:2]
            filename_category_map[filename] = category

    for category in set(filename_category_map.values()):
        (val / category).mkdir(parents=True, exist_ok=True)

    for p in valimages.iterdir():
        filename = p.name
        category = filename_category_map[filename]
        shutil.copy(p, val / category / filename)

    print('Done')


if __name__ == '__main__':
    main()
