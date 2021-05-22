import typing

import numpy
import tensorflow
from sklearn.cluster import DBSCAN

from visualization_msgs.msg import Marker

from std_msgs.msg import ColorRGBA

from geometry_msgs.msg import (
    Vector3,
    Pose,
    Point,
    Quaternion,
)


class ColorGenerator:

    def __init__(
        self,
        n: int = 30,
    ):
        self.n = n
        self.colors = numpy.random.rand(n, 3)


    def get_color(
        self,
        track_id: int,
    ) -> np.ndarray:

        color = self.colors[track_id % self.n]

        return color


def rle_msg_to_mask(rle_msg):
    mask = mask_utils.decode(
        {
            'counts': rle_msg.data,
            'size':   [rle_msg.height, rle_msg.width],
        }
    )

    return mask


def config_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:

        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def get_rotation_along_y(
    angle: float,
) -> numpy.ndarray:

    c = numpy.cos(angle)
    s = numpy.sin(angle)

    matrix = [
        [ c, 0, s, ],
        [ 0, 1, 0, ],
        [ -s, 0, c ],
    ]

    return numpy.array(matrix)


def sample_one_obj(
    points: numpy.ndarray,
    num_points_pad: int,
) -> typing.NoReturn:

    num_points, dims = points.shape

    if num_points < num_points_pad:

        pad = numpy.zeros(
            shape = (num_points - num_points, 3),
            dtype = numpy.float32,
        )

        ret = numpy.concatenate(
            [points, pad],
            axis=0,
        )

    else:

        idx = numpy.arange(num_points)
        numpy.random.shuffle(idx)

        idx = idx[0: num_points_pad]

        ret = points[idx]

    return ret


def get_rotation_angle(
    model,
    points: numpy.ndarray,
    num_points_pad: int = 500,
    resample_num: int = 10,
) -> typing.List[float]:
        
    input_data = np.stack(
        [
            x for x in map(lambda x: sample_one_obj(points, num_points_pad), range(resample_num))
        ],
        axis=0,
    )
    pred_val = model.predict(input_data)
    pred_cls = numpy.argmax(pred_val, axis=-1)
    
    ret = (pred_cls[0]*3+1.5)*np.pi/180.

    return ret


def get_scales(
    points:  np.ndarray,
) -> tuple:
    
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    
    scales = maxs - mins
    
    return scales


def points_to_bbox(
    points:  np.ndarray,
    cluster: bool = True,
    model,
) -> numpy.ndarray:
    
    if cluster:
        points = clastering(points)
        
    center = points.mean(axis=0)
        
    rotation_angle = get_rotation_angle(
        model = model,
        points = points,
    )
    
    scales = get_scales(points)
    
    return center, scales, rotation_angle


def create_marker_from_bbox(
    bbox_center,
    bbox_scales,
    bbox_angle_along_y,
    header,
    unique_number,
    colormap,
) -> Marker:

    color = colormap.get_color(unique_number)
    matrix = get_rotation_along_y(bbox_angle_along_y)
    quat = tf.transformations.quaternion_from_matrix(matrix)

    marker = Marker(
        id       = unique_number,
        header   = header,
        type     = Marker.CUBE,
        action   = Marker.ADD,
        lifetime = rospy.Duration(1.),
        color    = ColorRGBA(
            r = color[0],
            g = color[1],
            b = color[2],
            a = 0.5,
        ),
        scale    = Vector3(
            x = bbox_scales[0],
            y = bbox_scales[1],
            z = bbox_scales[2],
        ),
        pose     = Pose(
            position    = Point(
                x = bbox_center[0],
                y = bbox_center[1],
                z = bbox_center[2],
            ),
            orientation = Quaternion(
                x = quat[0],
                y = quat[1],
                z = quat[2],
                w = quat[3],
            ),
        ),
    )

    return marker
