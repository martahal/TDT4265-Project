import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

"""Reads the annotation xml-files and plots each bounding box in a height vs. width plot"""

#loop over annotation files
#loop over objects

def find_bounding_boxes(dir):
    """

    Args:
        dir: directory of annotation files

    Returns:
        bounding_boxes: a pandas dataframe of all gorund truth bounding boxes in the dataset.

    """
    bounding_boxes_table = {'type':[], 'height':[], 'width':[]}
    for filename in os.listdir(dir):
        if filename.endswith(".xml"):
            file_path = os.path.join(dir,filename)
            #print(file_path)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for object in root.iter('object'):
                name = object.find('name').text
                xmin = object.find('bndbox').find('xmin').text
                ymin = object.find('bndbox').find('ymin').text
                xmax = object.find('bndbox').find('xmax').text
                ymax = object.find('bndbox').find('ymax').text

                height, width = calculate_width_and_height(int(xmin), int(ymin), int(xmax), int(ymax))
                bounding_boxes_table['type'].append(name)
                bounding_boxes_table['height'].append(height)
                bounding_boxes_table['width'].append(width)

        else:
            continue
    bounding_boxes = pd.DataFrame(data=bounding_boxes_table)
    return bounding_boxes


def balance_bounding_boxes(bounding_boxes, n_samples = 1000):
    """

    Args:
        bounding_boxes: pandas dataframe of all bounding boxes.
    Returns:
        balanced_bounding boxes: a pandas dataframe with equally many random samples of each type

    """

    d00 = bounding_boxes.query('type == "D00"')
    d10 = bounding_boxes.query('type == "D10"')
    d20 = bounding_boxes.query('type == "D20"')
    d40 = bounding_boxes.query('type == "D40"')


    try:
        min_count = n_samples
        balanced_bounding_boxes = pd.concat(
            [d00.sample(min_count, random_state=0),
             d10.sample(min_count, random_state=0),
             d20.sample(min_count, random_state=0),
             d40.sample(min_count, random_state=0)])
    except ValueError:
        min_count = min(d00.size, d10.size, d20.size, d40.size)
        print(f'n_samples is {n_samples} and is larger than the number of bounding boxes for the smallest type\n'
              f'which is {min_count} ')

        balanced_bounding_boxes = pd.concat(
            [d00.sample(min_count, random_state=0),
             d10.sample(min_count, random_state=0),
             d20.sample(min_count, random_state=0),
             d40.sample(min_count, random_state=0)])

    return balanced_bounding_boxes



def calculate_width_and_height(xmin, ymin, xmax, ymax):
    height = ymax - ymin
    width = xmax - xmin
    return height, width

def plot_width_vs_heights(bounding_boxes):
    sns.scatterplot(data=bounding_boxes, x='width', y='height', hue='type', palette='deep')
    plt.title('Ground truth bounding box width vs height')
    plt.xlim(0, 800)
    plt.ylim(0, 600)

    plt.show()


def main():
    bounding_boxes = find_bounding_boxes('datasets/RDD2020_filtered/Annotations')

    balanced = balance_bounding_boxes(bounding_boxes,n_samples=1000)
    plot_width_vs_heights(balanced)

if __name__ == '__main__':
    main()