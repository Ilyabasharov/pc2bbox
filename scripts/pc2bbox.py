#!/usr/bin/python

from __future__ import annotations

import typing

import numpy
import tensorflow
from numpy.lib.recfunctions import structured_to_unstructured as s2u

import tf
import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import PointCloud2
from camera_objects_msgs.msg import ObjectArray
from visualization_msgs.msg import MarkerArray

import utils


class PC2BB:

    def __init__(
        self,
    ) -> PC2BB:

        rospy.init_node('pc2bbox')

        self.synchronizer = message_filters.TimeSynchronizer(
            [
                message_filters.Subscriber('point_cloud', PointCloud2),
                message_filters.Subscriber('objects',     ObjectArray),
            ],
            queue_size = 10,
        )

        self.synchronizer.registerCallback(self.process)

        self.publisher = rospy.Publisher(
            'visualisation',
            MarkerArray,
            queue_size = 10,
        )

        utils.config_gpu()

        model_path = rospy.get_param('~model_path', '/home/docker_solo/dataset')

        self.model = tensorflow.keras.models.load_model(model_path)
        self.colors = utils.ColorGenerator()

    def run(
        self,
    ) -> typing.NoReturn:

        rospy.spin()


    def process(
        self,
        pc_msg,
        objects_msg,
    ) -> MarkerArray:

        pc = ros_numpy.point_cloud2.pointcloud2_to_array(
            pc_msg
        )

        num_mask = numpy.isfinite(pc['x']) \
                 & numpy.isfinite(pc['y']) \
                 & numpy.isfinite(pc['z'])

        markers = []

        for object_msg in objects_msg.objects:

            if object_msg.track_id < 1:
                continue

            obj_mask = utils.rle_msg_to_mask(object_msg.rle).astype(np.bool)
            obj_pc = s2u(pc[num_mask & obj_mask][['x', 'y', 'z']])

            bbox_center, bbox_scales, bbox_angle = utils.points_to_bbox(
                model = self.model,
                points = obj_pc,
                cluster = True,
            )

            marker = utils.create_marker_from_bbox(
                bbox_center = bbox_center,
                bbox_scales = bbox_scales,
                bbox_angle_along_y = bbox_angle,
                header = pc_msg.header,
                unique_number = object_msg.track_id,
                colormap = self.colors,
            )

            markers.append(marker)

        out_msg = MarkerArray(markers)

        self.publisher.publish(out_msg)


def main():
    
    np.random.seed(100)

    pc2bb = PC2BB()
    pc2bb.run()


if __name__ == '__main__':
    main()
