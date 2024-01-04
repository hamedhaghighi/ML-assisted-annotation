from models.pointpillars import PointPillars
from models.yolo import Darknet


def get_model(*args, weights_init_normal=None, weights_path=None, is_two_d=True):
    if is_two_d:
        model = Darknet(*args)
        model.apply(weights_init_normal)
        model.load_weights(weights_path)
        return model

    return PointPillars()
