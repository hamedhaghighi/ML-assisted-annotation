from datasets.datasets2D import Image2DAnnotationDataset
from datasets.datasets3D import Image3DAnnotationDataset


def get_dataset(root_dir, labels, labels_to_classes, data_format, img_size, resize_tuple, img_root_dir=None, labelled_filenames=None, parent=None, is_two_d=True):
    if is_two_d:
        return Image2DAnnotationDataset(root_dir, labels, labels_to_classes, data_format, img_size, resize_tuple, img_root_dir, labelled_filenames, parent)
    return Image3DAnnotationDataset(root_dir, labels, labels_to_classes, img_root_dir, labelled_filenames, parent)