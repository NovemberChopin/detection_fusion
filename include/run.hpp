
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

#include "detection_fusion/Location.h"
#include "detection_fusion/DetecInfo.h"
#include "detection_fusion/EventInfo.h"
#include "detection_fusion/ShowPCD.h"
#include "detection_fusion/ShowLine.h"
#include "detection_fusion/SetDetecEvent.h"
#include "detection_fusion/GetConfig.h"
#include "detection_fusion/SetLineOrRect.h"
#include "detection_fusion/SetDetecParams.h"
#include "detection_fusion/SetTopic.h"
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
  message_filters::Subscriber<sensor_msgs::Image> *sub_img_ = nullptr;
  message_filters::Subscriber<sensor_msgs::PointCloud2> *sub_lidar_ = nullptr;
  message_filters::Synchronizer<syncPolicy> *sync_;
  image_transport::Publisher pub_img_;
  ros::Publisher pub_detec_info_;           // 发布物体检测结果
  ros::Publisher pub_event_;                // 发布事件消息

  ros::ServiceServer srv_show_pcd;          // 设置点云是否可见的服务
  ros::ServiceServer srv_show_line;         // 设置车道线是否可见的服务
  ros::ServiceServer srv_set_event;         // 设置交通事件检测是否开启
  ros::ServiceServer srv_get_config;        // 返回当前系统的设置信息
  ros::ServiceServer srv_set_line_roi;      // 设置事件检测的车道线和ROI区域
  ros::ServiceServer srv_set_detec_params;
  ros::ServiceServer srv_set_topic;

  Projector *projector;
  ObjectDetection *objectD;

  /**** 配置信息 ****/
  std::string intrinsic_path;
  std::string extrinsic_path;
  string camera_topic_name;
  string lidar_topic_name;
  string fusion_topic_name;
  string pub_detec_info_name;
  string pub_event_name;
  string srv_show_pcd_name;
  string srv_show_line_name;
  string srv_set_detec_params_name;
  string srv_set_event_name;
  string srv_get_config_name;
  string srv_set_line_roi_name;
  string srv_set_topic_name;

  double event_jam_threshold;           // 判断交通拥堵事件的车辆密度阈值
  double event_jam_speed;               // 交通拥堵事件的速度阈值

  double event_park_variance;           // 异常停车事件速度阈值

  bool show_pcd = false;            // 是否在视频中显示点云
  bool show_line = false;           // 是否在视频中显示车道线

  int object_detec_interval;                      // 物体检测间隔
  int event_detec_interval;                       // 事件检测周期（10表示进行10次物体检测后进行一次事件检测）

  vector<vector<Rect2d>> cur_track_bboxs;         // 两次检测之间跟踪算法返回的的数据(ROI变化的时间序列)
  vector<vector<Point3d>> cur_track_location;     // 两次检测之间通过跟踪算法计算的物体实际位置坐标序列

  // 事件检测参数
  Point2d *line_params = nullptr;                                   // 车道线参数，x y 分别对应直线参数 k b
  Point2d p1 = Point2d(0, 0);
  Point2d p2 = Point2d(0, 0);
  vector<Rect2d> vec_ROI;                                           // 存储每个事件的ROI区域
  
  std::vector<bool> hasDetecEvent = std::vector<bool>(5, false);		// 对应五种事件是否开始检测
	std::vector<int> detec_event_index = std::vector<int>(5, -1);			// 每个事件检测的间隔

  // 相机参数
  int output_video_fps;                          // 视频帧率
  
  cv::Size image_size;
  // cv::Mat I = cv::Mat::eye(3, 3, CV_32FC1);
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
  cv::Point2d getLineParams(Point2d &p1, Point2d &p2);
  void SetTopic(string image, string lidar);
  bool isStatic(vector<vector<Rect2d>> &track_boxes, int index, double event_park_variance);
  bool isInLeft(double k, double b, Point2d point);
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
  bool setShowLine(detection_fusion::ShowLine::Request &req,
                  detection_fusion::ShowLine::Response &res);
  bool setDetecParams(detection_fusion::SetDetecParams::Request &req,
                      detection_fusion::SetDetecParams::Response &res);
  bool setDetecEvent(detection_fusion::SetDetecEvent::Request &req,
                      detection_fusion::SetDetecEvent::Response &res);
  bool getConfigCallback(detection_fusion::GetConfig::Request &req,
                          detection_fusion::GetConfig::Response &res);
  bool getLineOrROI(detection_fusion::SetLineOrRect::Request &req,
                    detection_fusion::SetLineOrRect::Response &res);
  bool setTopicCallback(detection_fusion::SetTopic::Request &req,
                    detection_fusion::SetTopic::Response &res);
};

