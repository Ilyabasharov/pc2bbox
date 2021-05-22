# pc2bbox
PointCloud to Bounding Box convertor

## installation

```bash
cd catkin_ws/src
git clone https://github.com/Ilyabasharov/pc2bbox.git
git clone https://github.com/eric-wieser/ros_numpy.git
cd pc2bbox && pip install -r requirements.txt
cd ../..
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## model downloading

```bash
wget https://github.com/naurril/SUSTechPOINTS/releases/download/0.1/deep_annotation_inference.h5  \
	-O ../deep_annotation_inference.h5
```

## run

```bash
roslaunch pc2bbox main.launch \
	model_path:=../deep_annotation_inference.h5 \
	points:=/depth_registered/points \
	objects:=/stereo/objects \
	markers:=/pc2bbox/visualisation
```
