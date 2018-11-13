# YOLOv3-Caffe
The realization of yolov3 target detection based on Caffe and the optimization of target detection effect based on klaman multi-target tracking

# 依赖
1.caffe(add upsample layer https://github.com/maxuehao/caffe)  2.opencv2.4&python-opencv2.4  3.python2.7 

# 说明
1. 基于coco训练的yolov3 caffemodel https://pan.baidu.com/s/1u_xCALs-tGr-7GiocqvsNA 

# 效果图
![image](https://github.com/maxuehao/yolov3-caffe/blob/master/demo.png)

# kalman检测框滤波效果比对
Before Kalman filtering
![image](https://github.com/maxuehao/yolov3-caffe/blob/master/k2.png)

After Kalman filtering
![image](https://github.com/maxuehao/yolov3-caffe/blob/master/k1.png)
