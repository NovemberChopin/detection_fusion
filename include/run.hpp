
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

#include "extrinsic_param.hpp"
#include "intrinsic_param.hpp"
#include "projector_lidar.hpp"
#include "object_detection.hpp"

#include "detection_fusion/detecInfo.h"
#include "detection_fusion/ShowPCD.h"
#include "geometry_msgs/Point.h"

#include <Eigen/Core>
#include <iostream>
#include <string>
#include <vector>

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
  ros::Publisher pub_detec_info_;

  ros::ServiceServer srv_show_pcd;          // 设置点云是否可见的服务

  Projector *projector;
  ObjectDetection *objectD;

  std::string intrinsic_path = "/home/js/catkin/jiujiang/src/detection_fusion/config/camera_info.yml";
  std::string extrinsic_path = "/home/js/catkin/jiujiang/src/detection_fusion/config/trans_matrix.yml";
  bool show_pcd = false;            // 是否在视频中显示点云
  int fps;                          // 视频帧率
  int interval;                     // 物体检测间隔
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
  void Callback(const sensor_msgs::ImageConstPtr &msg_img,
                const sensor_msgs::PointCloud2ConstPtr &msg_lidar);
  void processOD(cv::Mat &image, int interval);           // 處理目標檢測函數
  cv::Point3f cameraToWorld(cv::Point2f point);           // 將圖像像素座標映射爲真實世界座標
  cv::Point2f getPixelPoint(Rect &rect, int type);

  bool setShowPCD(detection_fusion::ShowPCD::Request &req,        // ShowPCD 服务回调函数
                  detection_fusion::ShowPCD::Response &res);
};

