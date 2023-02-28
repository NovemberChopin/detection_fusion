
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

#include <Eigen/Core>
#include <iostream>
#include <string>


using namespace sensor_msgs;
using namespace message_filters;

static image_transport::Publisher pub_img;

std::string intrinsic_json = "/home/js/catkin/jiujiang/src/detection_fusion/config/camera_intrinsic.json";
std::string extrinsic_json = "/home/js/catkin/jiujiang/src/detection_fusion/config/lidar_to_camera-extrinsic.json";

std::vector<double> dist;
static Eigen::Matrix3d intrinsic_matrix_;
static Eigen::Matrix4d calibration_matrix_;

Projector projector;

void callback(const sensor_msgs::ImageConstPtr &msg_img,
              const sensor_msgs::PointCloud2ConstPtr &msg_lidar) {
  cv::Mat img;
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg_img, sensor_msgs::image_encodings::RGB8);
    // cv::imshow("Demo", cv_ptr->image);
    // cv::waitKey(50);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg_img->encoding.c_str());
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr livox_pcl(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg_lidar, *livox_pcl);
  projector.loadPointCloud(*livox_pcl);
  cv::Mat current_frame = projector.ProjectToRawImage(cv_ptr->image , intrinsic_matrix_, dist, calibration_matrix_);

  sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", current_frame).toImageMsg();
  pub_img.publish(pub_msg);
}


int main(int argc, char** argv) {

  LoadIntrinsic(intrinsic_json, intrinsic_matrix_, dist);
  LoadExtrinsic(extrinsic_json, calibration_matrix_);

  ros::init(argc, argv, "detection_fusion_node");
  ros::NodeHandle nh;
  //①分别接收两个topic
  message_filters::Subscriber<sensor_msgs::Image> sub_img(
          nh, "/hik_cam_node/hik_camera", 1, ros::TransportHints().tcpNoDelay());
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar(
                nh, "/livox/lidar", 1, ros::TransportHints().tcpNoDelay());

  // ②将两个topic的数据进行同步
  typedef sync_policies::ApproximateTime<sensor_msgs::Image,
                                         sensor_msgs::PointCloud2>syncPolicy;
  Synchronizer<syncPolicy> sync(syncPolicy(10), sub_img, sub_lidar);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  image_transport::ImageTransport it(nh);
  pub_img = it.advertise("/hik_img", 1);

  ros::spin();
  return 0;
}