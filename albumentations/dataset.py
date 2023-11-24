import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class SearchDataset(Dataset):

    def __init__(self, root_dir='../../fashionpedia', set='train2020', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        # TODO: change instatnce string
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_attributes_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)


    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        bboxes = np.zeros((0, 4))
        category_ids = np.zeros((0, 1))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return bboxes

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for a in coco_annotations:
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            bboxes = np.append(bboxes, np.array([a['bbox']]), axis=0)
            category_ids = np.append(category_ids, np.array([[a['category_id']]]), axis=0)

        return bboxes, category_ids


    def transform_bboxes(self, bboxes):
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        bboxes = np.array(bboxes)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        return bboxes


    def concat_annotations(self, bboxes, category_ids):
        # concat bboxes and category_ids
        return np.concatenate((bboxes, category_ids), axis=1, dtype=np.float32)


    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        img = self.load_image(index)
        bboxes, cat_ids = self.load_annotations(index)

        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                category_id=cat_ids
            )
            img = transformed['image']
            bboxes = transformed['bboxes']

        bboxes = self.transform_bboxes(bboxes)
        concatened = self.concat_annotations(bboxes, cat_ids)
        first_class_id = int(concatened[0][4])

        return img, first_class_id

if __name__ == "__main__":
    dataset = SearchDataset()
    print(dataset[0][0].shape)
    print(dataset[0][1])