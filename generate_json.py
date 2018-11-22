import os
import json
from collections import defaultdict
from PIL import Image

json_path = r'C:\Users\Abigail\Desktop\val.json'
output_dir = r'C:\Users\Abigail\Desktop'
source_txt = r'C:\Users\Abigail\Desktop\medical image processing\dataset1.2\class0\val\augmentated_txt'
source_img = r'C:\Users\Abigail\Desktop\medical image processing\dataset1.2\class0\val\augmentated_img'

def generate_coco_json():
    file_names = os.listdir(source_txt)
    ext_dict = defaultdict(dict)

    # generate dataset infomation
    info = defaultdict(dict)
    info['description'] = 'training dataset'
    info['version'] = '1.2'
    ext_dict['info'] = info

    # generate images
    images = []
    area = []
    for i, file_name in enumerate(file_names):
        img_path = os.path.join(source_img,  file_name.rstrip('.txt') + '.jpg')
        dict_img = {}
        img = Image.open(img_path, 'r')
        dict_img['file_name'] = file_name.rstrip('.txt') + '.jpg'
        dict_img['height'] = img.size[1]
        dict_img['width'] = img.size[0]
        dict_img['id'] = i
        dict_img['license'] = None
        dict_img['coco_url'] = None
        dict_img['date_captured'] = None
        dict_img['flickr_url'] = None
        images.append(dict_img)
        area.append(img.size[0] * img.size[1])
    print(images.__len__())
    print(images[0])
    ext_dict['images'] = images

    # generate annotations
    annotations = []
    index = 0
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(source_txt, file_name)
        dict_txt = {}
        dict_txt['image_id'] = i
        dict_txt['area'] = area[i]
        dict_txt['iscrowd'] = 0

        with open(file_path, 'r') as fp:
            data = fp.readlines()
        for line in data:
            line = line.split()
            dict_txt['id'] = index
            index += 1
            dict_txt['category_id'] = line[0]
            dict_txt['segmentation'] = None
            dict_txt['bbox'] = line[1:]

            annotations.append(dict_txt)
    print(annotations.__len__())
    print(annotations[0])
    ext_dict['annotations'] = annotations

    # generate categories
    categories = []
    categories.append({"supercategory": "specularity","id": 1,"name": "specularity"})
    categories.append({"supercategory": "saturation","id": 1,"name": "saturation"})
    categories.append({"supercategory": "artifact","id": 1,"name": "artifact"})
    categories.append({"supercategory": "blur","id": 1,"name": "blur"})
    categories.append({"supercategory": "contrast","id": 1,"name": "contrast"})
    categories.append({"supercategory": "bubbles","id": 1,"name": "bubbles"})
    categories.append({"supercategory": "instrument","id": 1,"name": "instrument"})
    ext_dict['catefories'] = categories

    return ext_dict

def main():
    data = generate_coco_json()
    with open(json_path, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()