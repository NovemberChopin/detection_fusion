#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#include <std_msgs/String.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/tracking.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace cv;
using namespace dnn;
using namespace std;

struct DetectionInfo
{
	int index;							// 当前检测的帧的 index
	vector<Rect> track_boxes_pre;		// 记录上一次的边框位置
	vector<Rect> track_boxes;			// 当前帧检测到物体的边框位置
	vector<int> track_classIds;			// 当前帧检测物体的类别
	vector<float> track_confidences;	// 当前帧检测物体的置信度
	vector<float> track_speeds;
	vector<float> track_distances;
	vector<cv::Point3f> location;		// 物体真实世界坐标
	vector<int> leftOrRight;            // 开始检测异常变道事件时物体的位置在线左（0）或线右
	DetectionInfo(int index) : index(index) {}
};


class ObjectDetection {
private:
  Net net;
	vector<string> classes;

	int inpWidth = 416;
	int inpHeight = 416;
	float confThreshold = 0.5; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold

	std::string classesFile = "/home/js/catkin/jiujiang/src/detection_fusion/config/coco.names";

	// cv::String modelConfiguration = "./src/mul_t/resources/yolov4-tiny.cfg";
  // cv::String modelWeights = "./src/mul_t/resources/yolov4-tiny.weights";

	cv::String modelConfiguration = "/home/js/catkin/jiujiang/src/detection_fusion/config/yolo-fastest-xl.cfg";
  cv::String modelWeights = "/home/js/catkin/jiujiang/src/detection_fusion/config/yolo-fastest-xl.weights";
  vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"}; 

public:

  DetectionInfo* detecRes;

	Ptr<MultiTracker> multiTracker;

  ObjectDetection();
  ~ObjectDetection();

	// 物体检测相关函数
  void runODModel(cv::Mat& frame);
	vector<String> getOutputsNames(const Net& net);
	void postprocess(Mat& frame, const vector<Mat>& outs);
	void drawPred(int classId, float conf, float speed, float dist,
					      int left, int top, int right, int bottom, Mat& frame);
	// 物体跟踪相关函数
	void runTrackerModel(cv::Mat & frame);
	Ptr<Tracker> createTrackerByName(string trackerType);

  void hello();
};

#endif