
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "projector_lidar.hpp"
#include "object_detection.hpp"

#include "detection_fusion/detecInfo.h"
#include "detection_fusion/EventInfo.h"
#include "detection_fusion/ShowPCD.h"
#include "detection_fusion/SetDetecEvent.h"
#include "detection_fusion/GetConfig.h"
#include "detection_fusion/SetLineOrRect.h"
#include "geometry_msgs/Point.h"

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <vector>

#include <time.h>

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

typedef sync_policies::ApproximateTime<sensor_msgs::Image,
                        sensor_msgs::PointCloud2>syncPolicy;

class Run {

private:
  ros::NodeHandle nh_;
  message_filters::Subscriber<sensor_msgs::Image> *sub_img_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> *sub_lidar_;
  message_filters::Synchronizer<syncPolicy> *sync_;
  image_transport::Publisher pub_img_;
  ros::Publisher pub_detec_info_;           // 发布物体检测结果
  ros::Publisher pub_event_;                // 发布事件消息

  ros::ServiceServer srv_show_pcd;          // 设置点云是否可见的服务
  ros::ServiceServer srv_set_event;         // 设置交通事件检测是否开启
  ros::ServiceServer srv_get_config;        // 返回当前系统的设置信息
  ros::ServiceServer srv_set_line_roi;      // 设置事件检测的车道线和ROI区域

  Projector *projector;
  ObjectDetection *objectD;

  std::string intrinsic_path;
  std::string extrinsic_path;
  string camera_topic_name;
  string lidar_topic_name;
  string fusion_topic_name;
  string pub_detec_info_name;
  string pub_event_name;
  string srv_show_pcd_name;
  string srv_set_event_name;
  string srv_get_config_name;
  string srv_set_line_roi_name;

  bool show_pcd = false;            // 是否在视频中显示点云

  vector<vector<Rect2d>> cur_track_bboxs;       // 两次检测之间跟踪算法返回的的数据

  // 事件检测参数
  cv::Rect2d left_road_roi = Rect2d(320, 0, 320, 720);
  cv::Rect2d right_road_roi = Rect2d(640, 0, 320, 720);
  cv::Point2d point_1 = Point2d(0, 0);                              // 标识车道线的两点
  cv::Point2d point_2 = Point2d(0, 0);                              // 
  cv::Rect2d detec_ROI = Rect2d(0, 0, 0, 0);                        // 事件检测的ROI区域
  vector<Rect2d> vec_ROI;                                           // 存储每个事件的ROI区域
  int event_detec_interval = 10;                                    // 事件检测周期（10表示进行10次物体检测后进行一次事件检测）
  std::vector<bool> hasDetecEvent = std::vector<bool>(5, false);		// 对应五种事件是否开始检测
	std::vector<int> detec_event_index = std::vector<int>(5, -1);			// 每个事件检测的间隔

  // 相机参数
  int output_video_fps;                          // 视频帧率
  int object_detec_interval;                     // 物体检测间隔
  cv::Size image_size;
  cv::Mat I = cv::Mat::eye(3, 3, CV_32FC1);
  cv::Mat mapX, mapY;
  cv::Mat cameraMatrix, distCoeffs;
  cv::Mat R_matrix, T_matrix;               // 雷達到攝像頭的轉移矩陣和平移矩陣
  cv::Mat rotationMatrix, transVector;      // 相機到真實世界的轉移矩陣和平移矩陣
  cv::Point3f camera_coord;                 // 相機在真實世界的座標

public:
  Run();
  ~Run();
  std::string getCurTime();
  void getParams();
  bool isStatic(vector<vector<Rect2d>> &track_boxes, int index);
  void PubEventTopic(int type, std::string e_name, std::string level, 
                      std::string judge, cv::Mat &image);
  void Callback(const sensor_msgs::ImageConstPtr &msg_img,
                const sensor_msgs::PointCloud2ConstPtr &msg_lidar);
  void processOD(cv::Mat &image, int interval);           // 處理目標檢測函數
  void detecEvent(cv::Mat &image);                        // 交通事件检测事件
  cv::Point3f cameraToWorld(cv::Point2f point);           // 將圖像像素座標映射爲真實世界座標
  cv::Point2f getPixelPoint(Rect2d &rect, int type);
  float getDistBetweenTwoDetec(int index);
  bool setShowPCD(detection_fusion::ShowPCD::Request &req,        // ShowPCD 服务回调函数
                  detection_fusion::ShowPCD::Response &res);
  bool setDetecEvent(detection_fusion::SetDetecEvent::Request &req,
                      detection_fusion::SetDetecEvent::Response &res);
  bool getConfigCallback(detection_fusion::GetConfig::Request &req,
                          detection_fusion::GetConfig::Response &res);
  bool getLineOrROI(detection_fusion::SetLineOrRect::Request &req,
                    detection_fusion::SetLineOrRect::Response &res);
};

