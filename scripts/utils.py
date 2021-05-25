import typing

import numpy
import tensorflow
from sklearn.cluster import DBSCAN
import pycocotools.mask as mask_utils

import cv2
import rospy
import tf
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
    ) -> numpy.ndarray:

        color = self.colors[track_id % self.n]

        return color

def clastering(pc):

    n_sampling = 500 if len(pc) > 500 else len(pc)

    indexes = numpy.linspace(0, len(pc), n_sampling, dtype=numpy.int32, endpoint=False)
    points_to_fit = pc[indexes]

    db = DBSCAN(eps=0.5, min_samples=2).fit(points_to_fit)

    values, counts = numpy.unique(db.labels_, return_counts=True)
    biggest_subcluster_id = values[numpy.argmax(counts)]

    mask_biggest_subcluster = db.labels_ == biggest_subcluster_id
    points_of_biggest_subcluster = points_to_fit[mask_biggest_subcluster]

    return points_of_biggest_subcluster


def rle_msg_to_mask(rle_msg):
    mask = mask_utils.decode(
        {
            'counts': rle_msg.data,
            'size':   [rle_msg.height, rle_msg.width],
        }
    )

    return mask


def config_gpu():
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')

    if gpus:

        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
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
        [ c, 0, s, 0,],
        [ 0, 1, 0, 0,],
        [-s, 0, c, 0,],
        [ 0, 0, 0, 1,],
    ]

    return numpy.array(matrix)


def sample_one_obj(
    points: numpy.ndarray,
    num_points_pad: int,
) -> typing.NoReturn:

    num_points, dims = points.shape

    if num_points < num_points_pad:

        pad = numpy.zeros(
            shape = (num_points_pad - num_points, 3),
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
    num_points_pad: int = 512,
    resample_num: int = 10,
) -> typing.List[float]:

    input_data = numpy.stack(
        [
            x for x in map(lambda x: sample_one_obj(points, num_points_pad), range(resample_num))
        ],
        axis=0,
    )

    pred_val = model.predict(input_data)
    pred_cls = numpy.argmax(pred_val, axis=-1)
    
    ret = (pred_cls[0]*3+1.5)*numpy.pi/180.
    angle = (pred_cls[0]*3+1.5)
    print(angle)

    return ret


def get_scales(
    points:  numpy.ndarray,
) -> tuple:
    
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    
    scales = maxs - mins
    
    return scales


def get_rotation_angle_via_cv(
    points: numpy.ndarray,
) -> float:
    
    _, (w, h), angle = cv2.minAreaRect(points)
    
    angle = angle*numpy.pi/180.

    return w, h, angle 


def points_to_bbox(
    points:  numpy.ndarray,
    model,
    cluster: bool = True,
) -> numpy.ndarray:
    
    if cluster:
        points = clastering(points)
        
    center = points.mean(axis=0)
        
    #rotation_angle = get_rotation_angle(
        #model = model,
        #points = points,
    #)

    w, h, rotation_angle = get_rotation_angle_via_cv(
        points[..., [2, 0]],
    )
    
    scales = get_scales(points)
   
    scales[0] = h
    scales[2] = w

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
