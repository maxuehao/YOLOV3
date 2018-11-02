# -*- coding: UTF-8 -*-
from __future__  import division
import math
import caffe
import numpy as np
import cv2
from collections import Counter
import time,os

#nms算法
def nms(dets, thresh):
	#dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
	#thresh:0.3,0.5....
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)#求每个bbox的面积
	order = scores.argsort()[::-1]#对分数进行倒排序
	keep = []#用来保存最后留下来的bbox
	while order.size > 0:
		i = order[0]#无条件保留每次迭代中置信度最高的bbox
		keep.append(i)
		#计算置信度最高的bbox和其他剩下bbox之间的交叉区域
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		#计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		#求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		#保留ovr小于thresh的bbox，进入下一次迭代。
		inds = np.where(ovr <= thresh)[0]
		#因为ovr中的索引不包括order[0]所以要向后移动一位
		order = order[inds + 1]
	return keep


#定义sigmod函数
def sigmod(x):
  	return 1.0 / (1.0 + math.exp(-x))


#检测模型前向运算
def load_model(net,test_img,feature_conv_name):
	input_img = cv2.resize(test_img,(img_w,img_w),interpolation=cv2.INTER_AREA)
	#input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
	input_img = input_img.transpose(2,0,1)
	#input_img = input_img.reshape((1,3,img_w,img_w))
	out = net.forward_all(data=input_img/256.0)
	shape = []
	for i in range(3):
		shape.append(out[feature_conv_name[i]].transpose(0, 3, 2, 1)[0])
	return shape


#处理前向输出feature_map
def feature_map_handle(length, shape, test_img, box_list):

	confidence = confidence_dic[str(length)]

	for i in range(length):
		for j in range(length):
			anchors_boxs_shape = shape[i][j].reshape((3, cl_num + 5))
			#将每个预测框向量包含信息迭代出来
			for k in range(3):
				anchors_box = anchors_boxs_shape[k]
				#计算实际置信度,阀值处理,anchors_box[7]
				score = sigmod(anchors_box[4])
				if score > confidence:
					#tolist()数组转list
					cls_list = anchors_box[5:cl_num + 5].tolist()
					label = cls_list.index(max(cls_list))
					obj_score = score
					x = ((sigmod(anchors_box[0]) + i)/float(length))*len(test_img[0])
					y = ((sigmod(anchors_box[1]) + j)/float(length))*len(test_img)
					if length ==13:
						w = (((bias_w[k+6]) * math.exp(anchors_box[2]))/img_w)*len(test_img[0])
						h = (((bias_h[k+6]) * math.exp(anchors_box[3]))/img_w)*len(test_img)
					elif length ==26: 
						w = (((bias_w[k+3]) * math.exp(anchors_box[2]))/img_w)*len(test_img[0])
						h = (((bias_h[k+3]) * math.exp(anchors_box[3]))/img_w)*len(test_img)
					elif length ==52: 
						w = (((bias_w[k]) * math.exp(anchors_box[2]))/img_w)*len(test_img[0])
						h = (((bias_h[k]) * math.exp(anchors_box[3]))/img_w)*len(test_img)
					x1 = int(x - w * 0.5)
					x2 = int(x + w * 0.5)
					y1 = int(y - h * 0.5)
					y2 = int(y + h * 0.5)
					#后两个下标值是物体ID及物体运动速度dv的占位用
					box_list.append([x1,y1,x2,y2,round(obj_score,4),label, 0, 0])


#3个feature_map的预选框的合并及NMS处理
def dect_box_handle(out_shape, test_img):
	box_list = []
	output_box = []
	for i in range(3):
		length =  len(out_shape[i])
		feature_map_handle(length, out_shape[i], test_img, box_list)
	if box_list:
		retain_box_index = nms(np.array(box_list), 0.2)
		for i in retain_box_index:
			output_box.append(box_list[i])
	return output_box


#计算两个BOX的IOU重合程度
def calculateIoU(candidateBound, groundTruthBound):
	cx1 = candidateBound[0]
	cy1 = candidateBound[1]
	cx2 = candidateBound[2]
	cy2 = candidateBound[3]
 
	gx1 = groundTruthBound[0]
	gy1 = groundTruthBound[1]
	gx2 = groundTruthBound[2]
	gy2 = groundTruthBound[3]
 
	carea = (cx2 - cx1) * (cy2 - cy1) 
	garea = (gx2 - gx1) * (gy2 - gy1) 
 
	x1 = max(cx1, gx1)
	y1 = max(cy1, gy1)
	x2 = min(cx2, gx2)
	y2 = min(cy2, gy2)
	w = max(0, x2 - x1)
	h = max(0, y2 - y1)
	area = w * h 
	iou = area / (carea + garea - area)
	return iou


#kalman
def Kalman():
	kalman = cv2.KalmanFilter(8,4)
	#测量矩阵 measurement matrix (H)
	kalman.measurementMatrix =np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]],np.float32)
	#状态传递矩阵 state transition matrix (A) 
	kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],
					    [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
	#过程噪声协方差矩阵 process noise covariance matrix (Q)
	#kalman.processNoiseCov = np.array([ [1e-6,0,0,0,0,0,0,0],[0,1e-6,0,0,0,0,0,0],[0,0,1e-6,0,0,0,0,0],[0,0,0,1e-6,0,0,0,0],
	#				    [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1e-8,0],[0,0,0,0,0,0,0,1e-6]], np.float32)
	#先验误差计协方差矩阵 priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)
	kalman.errorCovPre = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
					    [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)*10000
	#测量噪声协方差矩阵 measurement noise covariance matrix (R)
	kalman.measurementNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)	
	return kalman


#比对当前检测物体与k-1帧的检测框的iou,大于某一阈值判断为同一物体,赋相同ID	
def box_id(output_box, previous_box ,iou_confidence):
		for k,k_value in enumerate(output_box):
			temp = False
			for i,i_value in enumerate(previous_box):	
				iou = calculateIoU(i_value, k_value)
				if iou > iou_confidence:
					temp = True
					temp_id = i_value[6]
					#求当前时刻与K-1时刻同一物体的中心点坐标的差值，衡量物体运动速度
					cpx1 = k_value[0]+0.5*(k_value[2]-k_value[0])
					cpy1 = k_value[1]+0.5*(k_value[3]-k_value[1])
					cpx2 = i_value[0]+0.5*(i_value[2]-i_value[0])
					cpy2 = i_value[1]+0.5*(i_value[3]-i_value[1])
					dv = abs(0.5*((cpx2-cpx1)+(cpy2-cpy1)))
					continue
			if temp:
				output_box[k][6] = temp_id
				output_box[k][7] = dv
			else:
				output_box[k][6] = ID
				global ID 
				ID += 1
#kalman跟踪主函数
def kalman_handle(kalman_dict, output_box, kalman_lose, iou_confidence, lose_cout):
	bbox = []
	if kalman_dict:
		for k, v in kalman_dict.items():
			current_pre =v.predict()
    			cpx, cpy, w, h = current_pre[0],current_pre[1],current_pre[2],current_pre[3]
			#kalman预测的box
			k_box = [cpx-0.5*w, cpy-0.5*h, cpx+0.5*w, cpy+0.5*h]
			#判断是否有和物体ID相同的滤波器，如果没有相对应的滤波器则判断滤波器预测的框与检测框的iou
			#大于某一阈值则重新匹配滤波器与物体ID
			s = False
			t = False
			#判断当前检测物体是否有相对应的滤波器
			for i, i_value in enumerate(output_box):
				if i_value[6] == k:
					s = True
					s_index = i
				k_iou = calculateIoU(i_value, k_box)
				if k_iou > iou_confidence:
					t = True
					t_index = i
			if s:
				w = output_box[s_index][2]-output_box[s_index][0]
				h = output_box[s_index][3]-output_box[s_index][1]	
				x = output_box[s_index][0]+(0.5*w)
				y = output_box[s_index][1]+(0.5*h)
				#更新滤波器参数
				current_mes = np.array([[np.float32(x)],[np.float32(y)],[np.float32(w)],[np.float32(h)]])
				v.correct(current_mes)
				#根据物体运动趋势调节kalman参数，防止物体运动过快跟踪不上
				if output_box[s_index][7] >= 1:
					temp = 1e-2
				else:
					temp = 1e-7
				v.processNoiseCov = np.array([ [1e-8,0,0,0,0,0,0,0],[0,1e-8,0,0,0,0,0,0],[0,0,1e-8,0,0,0,0,0],[0,0,0,1e-8,0,0,0,0],
				    [0,0,0,0,temp,0,0,0],[0,0,0,0,0,temp,0,0],[0,0,0,0,0,0,1e-5,0],[0,0,0,0,0,0,0,1e-5]], np.float32)
				bbox.append([int(cpx-0.5*w), int(cpy-0.5*h), int(cpx+0.5*w), int(cpy+0.5*h), output_box[s_index][5],k,output_box[s_index][7]])	

			#重新匹配滤波器id
			elif t:
				#交换id
				kalman_dict[output_box[t_index][6]] = v
				del kalman_dict[k]
				w = output_box[t_index][2]-output_box[t_index][0]
				h = output_box[t_index][3]-output_box[t_index][1]	
				x = output_box[t_index][0]+(0.5*w)
				y = output_box[t_index][1]+(0.5*h)
				current_mes = np.array([[np.float32(x)],[np.float32(y)],[np.float32(w)],[np.float32(h)]])
				kalman_dict[output_box[t_index][6]].correct(current_mes)
				if output_box[t_index][7] >= 3:
					temp = 1e-3
				else:
					temp = 1e-8
				kalman_dict[output_box[t_index][6]].processNoiseCov = np.array([ [1e-8,0,0,0,0,0,0,0],[0,1e-8,0,0,0,0,0,0],[0,0,1e-8,0,0,0,0,0],[0,0,0,1e-8,0,0,0,0],
				    [0,0,0,0,temp,0,0,0],[0,0,0,0,0,temp,0,0],[0,0,0,0,0,0,1e-4,0],[0,0,0,0,0,0,0,1e-4]], np.float32)
				bbox.append([int(cpx-0.5*w), int(cpy-0.5*h), int(cpx+0.5*w), int(cpy+0.5*h), output_box[t_index][5],k,output_box[t_index][7]])	
			else:
				if k in kalman_lose:
					if kalman_lose[k] > lose_cout:
						del kalman_dict[k]
					else:
						kalman_lose[k]=	kalman_lose[k]+1
				else:
					kalman_lose[k] = 0
				bbox.append([int(cpx-0.5*w), int(cpy-0.5*h), int(cpx+0.5*w), int(cpy+0.5*h), 10, k,"s"])		
	for i in output_box:
		w = i[2]-i[0]
		h = i[3]-i[1]	
		x = i[0]+(0.5*w)
		y = i[1]+(0.5*h)
		current_mes = np.array([[np.float32(x)],[np.float32(y)],[np.float32(w)],[np.float32(h)]])
		if i[6] not in kalman_dict:
			kalman_dict[i[6]]=Kalman()
			kalman_dict[i[6]].correct(current_mes)	
	return bbox	



if __name__ == "__main__":
	confidence_dic={"13":0.1, "26":0.1, "52":0.1}
	fps=0
	#类别数目
	cl_num = 9
	#输入图片尺寸	
	img_w = 416	
	#加载label文件
	label_name = []
	with open('v1.names', 'r') as f:
	   for line in f.readlines():
	      label_name.append(line.strip()) 

	#模型训练时设置的anchor_box比例
	bias_w = [10, 16, 33, 30, 62, 59, 116, 156, 372]
	bias_h = [13, 30, 23, 61, 45, 119, 90, 198, 362]

	#需要输出的３层feature_map的名称
	feature_conv_name = ["layer41-conv","layer53-conv","layer65-conv"]
	caffe.set_mode_gpu()
	#加载检测模型
	net = caffe.Net('YOLOV3.prototxt', 'YOLOV3.caffemodel', caffe.TEST)
	cap = cv2.VideoCapture('MOT16-03.mp4')

	#创建储存k-1时刻的buff数组
	previous_box = []
	#定义物体初始ID
	ID = 0
	#判断物体ID的IOU阈值
	iou_confidence = 0.45
	#滤波器丢失阈值
	lose_cout = 5
	#储存kalman滤波器的map
	kalman_dict = {}
	#kalman追踪丢失次数map
	kalman_lose = {}

	while True:
		ret, test_img = cap.read()
		test_img = cv2.resize(test_img,(1280,720),interpolation=cv2.INTER_AREA)
		start = time.clock()
		print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"
		print "FPS:%s"%fps
		fps += 1
		out_shape = load_model(net,test_img,feature_conv_name)
	   	end = time.clock()
	   	print "SPEND_TIME:"+str(end-start)
		output_box = dect_box_handle(out_shape, test_img)
		#为每个物体赋初始id
		box_id(output_box, previous_box ,iou_confidence)
		#kallman滤波处理
		bbox = kalman_handle(kalman_dict, output_box, kalman_lose, iou_confidence, lose_cout)
		for i in bbox:
			cv2.putText(test_img, "id:"+str(i[5]), (i[0], i[1]-40), 0, 0.7, (255, 0, 255), 2)
			if i[6] != "s":
				cv2.putText(test_img, "dv:"+str(i[6]), (i[0], i[1]-4), 0, 0.7, (0, 255, 255), 2)
			if i[4] == 10:
				cv2.putText(test_img, "loss", (i[0], i[1]-4), 0, 0.7, (0, 0, 255), 2)
			else:
				cv2.putText(test_img, label_name[i[4]], (i[0], i[1]-20), 0, 0.7, (0, 255, 0), 2)
			cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 245,0), 2)
	
		#更新buff数组
		previous_box = output_box
		cv2.imshow("capture", test_img)
		cv2.waitKey(1)
		fps += 1
	cap.release()
	cv2.destroyAllWindows()
	#cv2.imwrite("2.jpg", test_img)
