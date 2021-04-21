import xml.dom.minidom
from os import listdir

path_to_dataset = '../datasets/RDD2020_filtered'
output = '../datasets/RDD2020_cropped'
annotations_path = path_to_dataset + '/Annotations'

for filename in listdir(annotations_path):
    doc = xml.dom.minidom.parse(annotations_path + '/' + filename)
    print(doc)