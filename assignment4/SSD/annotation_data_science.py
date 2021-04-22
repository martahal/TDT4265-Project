import os
import xml.etree.ElementTree as ET
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random



def find_bounding_boxes_tdt(file_path, standard_image_size):
    """
    Args:
        file_path: directory of annotation files for tdt dataset
        standard_image_size: a tuple (height, width) always(1080,1920)

    Returns:
        bounding_boxes: a pandas dataframe of all gorund truth bounding boxes in the dataset.

    """
    bounding_boxes_table = {'type':[], 'height':[], 'width':[], 'cx':[], 'cy': [], 'size': []}
    img_width = 1920
    img_height =1080
    f = open(file_path, )
    data = json.load(f)
    for annotation in data['annotations']:
        cat_id = annotation['category_id']
        if cat_id == 0:
            type = 'D00'
        elif cat_id == 1:
            type = 'D10'
        elif cat_id == 2:
            type = 'D20'
        elif cat_id == 3:
            type = 'D40'
        else:
            type = 'Undefined'

        [xmin, ymin, width, height] = annotation['bbox']
        size = annotation['area']

        cx, cy = calculate_bb_centers_tdt(xmin, ymin, width, height)
        bounding_boxes_table['type'].append(type)
        bounding_boxes_table['height'].append(int(round(height)))
        bounding_boxes_table['width'].append(int(round(width)))
        bounding_boxes_table['cx'].append(int(round(cx)))
        bounding_boxes_table['cy'].append(int(round(cy)))
        bounding_boxes_table['size'].append(int(round(size)))

    bounding_boxes = pd.DataFrame(data=bounding_boxes_table)
    return bounding_boxes


def find_bounding_boxes_rdd(dir, standard_image_size):
    """

    Args:
        dir: directory of annotation files for rdd dataset
        standard_image_size: a tuple (height, width) always(600,600)

    Returns:
        bounding_boxes: a pandas dataframe of all gorund truth bounding boxes in the dataset.

    """
    bounding_boxes_table = {'type':[], 'height':[], 'width':[], 'cx':[], 'cy': []}
    for filename in os.listdir(dir):
        if filename.endswith(".xml"):
            file_path = os.path.join(dir,filename)
            #print(file_path)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for size in root.iter('size'):
                img_width = size.find('width').text
                img_height = size.find('height').text

            for object in root.iter('object'):
                name = object.find('name').text
                xmin = object.find('bndbox').find('xmin').text
                ymin = object.find('bndbox').find('ymin').text
                xmax = object.find('bndbox').find('xmax').text
                ymax = object.find('bndbox').find('ymax').text

                height, width = calculate_width_and_height(int(xmin), int(ymin), int(xmax), int(ymax), int(img_width), int(img_height), standard_image_size)
                cx, cy = calculate_bb_centers_rdd(int(xmin), int(ymin), int(xmax), int(ymax), int(img_width), int(img_width), standard_image_size)
                bounding_boxes_table['type'].append(name)
                bounding_boxes_table['height'].append(height)
                bounding_boxes_table['width'].append(width)
                bounding_boxes_table['cx'].append(cx)
                bounding_boxes_table['cy'].append(cy)

        else:
            continue

    bounding_boxes = pd.DataFrame(data=bounding_boxes_table)
    #add a new row with bounding box size
    bounding_boxes['size'] = bounding_boxes.apply(lambda row: row.width * row.height, axis=1)
    return bounding_boxes


def calculate_width_and_height(xmin, ymin, xmax, ymax, img_width, img_height, std_image_size):
    # needs to normalize box sizes relative to input image sizes.
    scale_x = std_image_size[1]/img_width
    scale_y = std_image_size[0]/img_height
    height = ymax - ymin
    width = xmax - xmin
    return int(round(height*scale_y)), int(round(width*scale_x))


def calculate_bb_centers_tdt(xmin, ymin, width, height):
    cx = xmin + width / 2
    cy = ymin + height / 2
    return cx, cy

def calculate_bb_centers_rdd(xmin, ymin, xmax, ymax, img_width, img_height, std_image_size):
    # needs to normalize box positions relative to input image sizes. Assuming square images

    # Note that pixel 0,0 is at the top left of the image, whereas 0,0 is in the bottom left of the plot
    scale_x = std_image_size[1] / img_width
    scale_y = std_image_size[0]/ img_height
    height = int(round((ymax - ymin) * scale_y))
    width = int(round((xmax - xmin) * scale_x))

    cy_prime = ymin + (height//2)
    cx_prime = xmin + (width//2)

    cy = std_image_size[1] - cy_prime
    cx = std_image_size[0] - cx_prime
    return cx, cy

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


def plot_size_distribution(bounding_boxes, dataset_name, standard_image_size):

    bounding_boxes['height'].plot.hist(by="type",bins=100, alpha=0.5)
    bounding_boxes['width'].plot.hist(by="width", bins = 100, alpha=0.5)
    #bounding_boxes['size'].plot.hist(by="type", bins=1000, alpha=0.5)
    plt.xlabel('Number of pixels')

    plt.xlim(0, standard_image_size[1])
    plt.title(f'Distribution of bounding box height and width in {dataset_name}')
    plt.savefig(f'figures/Bounding_box_h_w_distribution{dataset_name}.png')
    plt.show()


def plot_width_vs_heights(bounding_boxes, dataset_name,  centroids=None, standard_image_size=(600, 600), plot_centroids = False ):
    sns.scatterplot(data=bounding_boxes, x='width', y='height', hue='type', palette='pastel')
    if plot_centroids:
        sns.scatterplot(data=centroids, x='width', y='height', hue='type', palette='bright')

    plt.title(f'Ground truth bounding box width vs height on {dataset_name} dataset')

    plt.xlim(0, standard_image_size[1])
    plt.ylim(0, standard_image_size[0])
    if dataset_name == 'TDT4265':
        plt.savefig('figures/Bounding_box_clusters_TDT4265.png')
    elif dataset_name == 'RDD2020':
        plt.savefig('figures/Bounding_box_clusters_RDD2020.png')
    plt.show()

def plot_bb_positions(bounding_boxes, dataset_name, standard_image_size):
    sns.scatterplot(data=bounding_boxes, x='cx', y='cy', hue='type', palette='bright')
    plt.title(f'Ground truth bounding box positions in {dataset_name} dataset')
    plt.xlim(-50, standard_image_size[1])
    plt.ylim(-50, standard_image_size[0])
    if dataset_name == 'TDT4265':
        plt.savefig('figures/Bounding_box_positions_TDT4265.png')
    elif dataset_name == 'RDD2020':
        plt.savefig('figures/Bounding_box_positions_RDD2020.png')
    plt.show()

def find_min_and_max_bboxes(bounding_boxes):

    d00 = bounding_boxes.query('type == "D00"')
    d10 = bounding_boxes.query('type == "D10"')
    d20 = bounding_boxes.query('type == "D20"')
    d40 = bounding_boxes.query('type == "D40"')

    min_d00 = d00['size'].min()
    min_d10 = d10['size'].min()
    min_d20 = d20['size'].min()
    min_d40 = d40['size'].min()

    max_d00 = d00['size'].max()
    max_d10 = d10['size'].max()
    max_d20 = d20['size'].max()
    max_d40 = d40['size'].max()

    d00_min_bb = d00.loc[d00['size']== min_d00]
    d10_min_bb = d10.loc[d10['size']== min_d10]
    d20_min_bb = d20.loc[d20['size']== min_d20]
    d40_min_bb = d40.loc[d40['size']== min_d40]

    d00_max_bb = d00.loc[d00['size']== max_d00]
    d10_max_bb = d10.loc[d10['size']== max_d10]
    d20_max_bb = d20.loc[d20['size']== max_d20]
    d40_max_bb = d40.loc[d40['size']== max_d40]


    print(d00_min_bb)
    print(d10_min_bb)
    print(d20_min_bb)
    print(d40_min_bb)

    print(d00_max_bb)
    print(d10_max_bb)
    print(d20_max_bb)
    print(d40_max_bb)


def find_centroid_bboxes(bounding_boxes):
    d00 = bounding_boxes.query('type == "D00"')
    d10 = bounding_boxes.query('type == "D10"')
    d20 = bounding_boxes.query('type == "D20"')
    d40 = bounding_boxes.query('type == "D40"')

    height_mean_d00 = d00['height'].mean()
    height_mean_d10 = d10['height'].mean()
    height_mean_d20 = d20['height'].mean()
    height_mean_d40 = d40['height'].mean()

    width_mean_d00 = d00['width'].mean()
    width_mean_d10 = d10['width'].mean()
    width_mean_d20 = d20['width'].mean()
    width_mean_d40 = d40['width'].mean()

    d00_centroid = pd.DataFrame(
        [["D00 Centroid", height_mean_d00, width_mean_d00, height_mean_d00 * width_mean_d00]],
        columns=['type', 'height', 'width', 'size']
    )
    d10_centroid = pd.DataFrame(
        [["D10 Centroid", height_mean_d10, width_mean_d10, height_mean_d10 * width_mean_d10]],
        columns=['type', 'height', 'width', 'size']
    )
    d20_centroid = pd.DataFrame(
        [["D20 Centroid", height_mean_d20, width_mean_d20, height_mean_d20 * width_mean_d20]],
        columns=['type', 'height', 'width', 'size']
    )
    d40_centroid = pd.DataFrame(
        [["D40 Centroid", height_mean_d40, width_mean_d40, height_mean_d40 * width_mean_d40]],
        columns=['type', 'height', 'width', 'size']
    )

    centroids = pd.concat([
        d00_centroid,
        d10_centroid,
        d20_centroid,
        d40_centroid]
    )
    #print('CENTROIDS:')
    #print(d00_centroid)
    #print(d10_centroid)
    #print(d20_centroid)
    #print(d40_centroid)

    return centroids

def main():
    #dataset_name = 'RDD2020'
    #std_image_size = (600,600)
    #bounding_boxes = find_bounding_boxes_rdd('datasets/RDD2020_filtered/Annotations', standard_image_size = std_image_size)


    dataset_name =  'TDT4265'
    std_image_size = (1080,1920)
    bounding_boxes = find_bounding_boxes_tdt('datasets/tdt4265/labels.json', standard_image_size=std_image_size)
    print(bounding_boxes.head(100))
    find_min_and_max_bboxes(bounding_boxes)
    balanced = balance_bounding_boxes(bounding_boxes,n_samples=1000)
    centroids = find_centroid_bboxes(bounding_boxes) # Must come after balancing bounding boxes
    #print(centroids)
    plot_width_vs_heights(balanced,dataset_name,  centroids, standard_image_size =std_image_size, plot_centroids=True,)
    plot_bb_positions(balanced, dataset_name, standard_image_size =std_image_size )
    plot_size_distribution(bounding_boxes,dataset_name, standard_image_size =std_image_size )

if __name__ == '__main__':
    main()