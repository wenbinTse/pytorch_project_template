import os
import glob
import xml.etree.ElementTree as ET

from dataset import voc_name_to_class

data_root = '/data/xiewenbin/VOC2012/'

def parse_one_file(xml_file):
    id = os.path.basename(xml_file)[:-4]
    node = ET.ElementTree(file=xml_file)
    labels = [x.text for x in node.iterfind('object/name')]
    return id, set(labels)

files = glob.glob(data_root + 'Annotations/*.xml')
records = []
for file in files:
    id, labels = parse_one_file(file)
    for label in labels:
        records.append((id, voc_name_to_class[label] - 1))

with open(data_root + 'class.txt', 'w+') as f:
    for id, label in records:
        f.writelines(f'{id} {label}\n')
