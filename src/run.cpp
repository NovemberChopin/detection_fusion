
#include "run.hpp"

Run::Run() {
  this->getParams();

  this->projector = new Projector();
  this->objectD = new ObjectDetection();

  // hasDetecEvent[1] = true;      // 测试，index 为 4 的事件开启

  image_size = cv::Size(1280, 720);
  this->camera_coord = cv::Point3f(0.0, 0.0, 0.0);
  cv::FileStorage cameraPameras(this->intrinsic_path, cv::FileStorage::READ);
  cv::FileStorage calibration_matrix(this->extrinsic_path, cv::FileStorage::READ);
  cameraPameras["camera_matrix"] >> this->cameraMatrix;
  cameraPameras["dist_coeffs"] >> this->distCoeffs;
  calibration_matrix["rotation_matrix"] >> this->R_matrix;
  calibration_matrix["trans_matrix"] >> this->T_matrix;
  // 計算相機姿態
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> world_points;
  cameraPameras["imagepoints"] >> image_points;
  cameraPameras["worldpoints"] >> world_points;
  cv::Mat rotationVector;
  cv::solvePnP(world_points, image_points, this->cameraMatrix, this->distCoeffs, rotationVector, this->transVector);
  cv::Rodrigues(rotationVector, this->rotationMatrix);
  // std::cerr << "Rotation Matrix:" << std::endl << this->rotationMatrix << std::endl;
  // std::cerr << "Trans Matrix:" << std::endl << this->transVector << std::endl;
  this->cameraMatrix.convertTo(this->cameraMatrix, CV_32FC1);
  this->distCoeffs.convertTo(this->distCoeffs, CV_32FC1);
  this->R_matrix.convertTo(this->R_matrix, CV_32FC1);
  this->T_matrix.convertTo(this->T_matrix, CV_32FC1);
  this->rotationMatrix.convertTo(this->rotationMatrix, CV_32FC1);
  this->transVector.convertTo(this->transVector, CV_32FC1);
  // 計算映射矩陣
  cv::initUndistortRectifyMap(this->cameraMatrix, this->distCoeffs, cv::Mat(), 
                      this->cameraMatrix, image_size, CV_32FC1, this->mapX, this->mapY);

  this->sub_img_ = new message_filters::Subscriber<sensor_msgs::Image>(
          this->nh_, this->camera_topic_name, 1, ros::TransportHints().tcpNoDelay());
  this->sub_lidar_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(
                this->nh_, this->lidar_topic_name, 1, ros::TransportHints().tcpNoDelay());
  this->sync_ = new message_filters::Synchronizer<syncPolicy>(
                syncPolicy(10), *sub_img_, *sub_lidar_);
  this->sync_->registerCallback(boost::bind(&Run::Callback, this, _1, _2));

  image_transport::ImageTransport it(this->nh_);
  this->pub_img_ = it.advertise(this->fusion_topic_name, 1);
  this->pub_detec_info_ = this->nh_.advertise<detection_fusion::detecInfo>(this->pub_detec_info_name, 5);
  this->pub_event_ = this->nh_.advertise<detection_fusion::EventInfo>(this->pub_event_name, 5);
  this->srv_show_pcd = this->nh_.advertiseService(this->srv_show_pcd_name, &Run::setShowPCD, this);
  this->srv_set_event = this->nh_.advertiseService(this->srv_set_event_name, &Run::setDetecEvent, this);
  this->srv_get_config = this->nh_.advertiseService(this->srv_get_config_name, &Run::getConfigCallback, this);
}


Run::~Run() {
  delete this->sub_img_;
  delete this->sub_lidar_;
  delete this->sync_;
}


void Run::getParams() {
  if (!ros::param::get("camera_info", this->intrinsic_path)) {
    cout << "Can not get the value of camera_info" << endl;
    exit(1);
  }

  if (!ros::param::get("trans_matrix", this->extrinsic_path)) {
    cout << "Can not get the value of trans_matrix" << endl;
    exit(1);
  }

  ros::param::get("camera_topic", this->camera_topic_name);
  ros::param::get("lidar_topic", this->lidar_topic_name);
  ros::param::get("fusion_topic", this->fusion_topic_name);
  ros::param::get("pub_detec_info", this->pub_detec_info_name);
  ros::param::get("pub_event", this->pub_event_name);
  ros::param::get("srv_show_pcd", this->srv_show_pcd_name);
  ros::param::get("srv_set_event", this->srv_set_event_name);
  ros::param::get("srv_get_config", this->srv_get_config_name);

  ros::param::get("output_video_fps", this->output_video_fps);
  ros::param::get("object_detec_interval", this->object_detec_interval);
  ros::param::get("event_detec_interval", this->event_detec_interval);
}


// 设置点云是否可见的服务函数
bool Run::setShowPCD(detection_fusion::ShowPCD::Request &req,        // ShowPCD 服务回调函数
                  detection_fusion::ShowPCD::Response &res) {
  std::cout << "req.flag: " << req.flag << std::endl;
  if (req.flag == "on") {
    this->show_pcd = true;      // 显示点云数据
  } else {
    this->show_pcd = false;
  }
  res.status = 0;
  return true;
}


bool Run::setDetecEvent(detection_fusion::SetDetecEvent::Request &req,
                      detection_fusion::SetDetecEvent::Response &res) {
  // std::cout << req.event_index << " " << req.flag << std::endl;
  if (req.event_index < 0 && req.event_index >= this->hasDetecEvent.size()) {
    res.status = 1;
    res.error = "event_index is not right";
    return false;
  } else {
    int index = req.event_index;
    if (req.flag == "on") {   // 开启检测
      this->hasDetecEvent[index] = true;
    } else {  // 关闭检测
      this->hasDetecEvent[index] = false;
      this->detec_event_index[index] = -1;
    }
    res.status = 0;
    return true;
  }
}


bool Run::getConfigCallback(detection_fusion::GetConfig::Request &req,
                          detection_fusion::GetConfig::Response &res) {
  res.showPCD = this->show_pcd;
  res.eventState.assign(this->hasDetecEvent.begin(), this->hasDetecEvent.end());
  return true;
}


void Run::Callback(const sensor_msgs::ImageConstPtr &msg_img,
                const sensor_msgs::PointCloud2ConstPtr &msg_lidar) {
  cv::Mat fixed_img;
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg_img, sensor_msgs::image_encodings::RGB8);
    cv::remap(cv_ptr->image, fixed_img, this->mapX, this->mapY, cv::INTER_LINEAR);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg_img->encoding.c_str());
  }
  // 執行物體檢測
  this->processOD(fixed_img, this->object_detec_interval);

  cv::Mat fusion_frame;
  if (this->show_pcd) {
    // 获取点云数据
    pcl::PointCloud<pcl::PointXYZI>::Ptr livox_pcl(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg_lidar, *livox_pcl);
    this->projector->loadPointCloud(*livox_pcl);
    fusion_frame = this->projector->ProjectToRawMat(fixed_img, 
                    this->cameraMatrix, this->distCoeffs, this->R_matrix, this->T_matrix);
  } else {
    fusion_frame = fixed_img;
  }
  
  // 將 Mat 轉化爲 sensor_msgs::Image 話題數據
  sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(
                          std_msgs::Header(), "rgb8", fusion_frame).toImageMsg();
  this->pub_img_.publish(pub_msg);

  // 发布检测结果话题
  detection_fusion::detecInfo detecMsg;
  DetectionInfo *info = this->objectD->detecRes;
  for (auto id: info->track_classIds)       // 物体ID（分类）
    detecMsg.id.push_back(id);
  for (auto conf: info->track_confidences)  // 识别置信度
    detecMsg.confid.push_back(conf);
  for (auto speed: info->track_speeds)      // 物体速度
    detecMsg.speeds.push_back(speed);
  for (auto dist: info->track_distances)    // 物体距离
    detecMsg.dist.push_back(dist);
  for (auto lo: info->location) {           // 坐标信息
    geometry_msgs::Point point;
    point.x = lo.x;
    point.y = lo.y;
    point.z = lo.z;
    detecMsg.location.push_back(point);
  }
  this->pub_detec_info_.publish(detecMsg);

}


/**
 * @brief 把像素坐标转化为世界坐标
 * 
 * @pax 像素 x 坐标
 * @param point 圖像像素座標
 * @return cv::Point3f 世界坐标
 */
cv::Point3f Run::cameraToWorld(cv::Point2f point) {
  cv::Mat invR_x_invM_x_uv1, invR_x_tvec, wcPoint;
  double Z = 0;   // Hypothesis ground:

	cv::Mat screenCoordinates = cv::Mat::ones(3, 1, cv::DataType<double>::type);
	screenCoordinates.at<double>(0, 0) = point.x;
	screenCoordinates.at<double>(1, 0) = point.y;
	screenCoordinates.at<double>(2, 0) = 1; // f=1

  invR_x_invM_x_uv1.convertTo(invR_x_invM_x_uv1, CV_32FC1);
  invR_x_tvec.convertTo(invR_x_tvec, CV_32FC1);
  wcPoint.convertTo(wcPoint, CV_32FC1);
  screenCoordinates.convertTo(screenCoordinates, CV_32FC1);

	// s and point calculation, described here:
	// https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
	// invR_x_invM_x_uv1 = rotationMatrix.inv() * cameraMatrix.inv() * screenCoordinates;
	// invR_x_tvec = rotationMatrix.inv() * transVector;
	// wcPoint = (Z + invR_x_tvec.at<double>(2, 0)) / invR_x_invM_x_uv1.at<double>(2, 0) * invR_x_invM_x_uv1 - invR_x_tvec;
	// //wcPoint = invR_x_invM_x_uv1 - invR_x_tvec;
	// cv::Point3f worldCoordinates(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));

	invR_x_invM_x_uv1 = this->rotationMatrix.inv() * this->cameraMatrix.inv() * screenCoordinates;
  invR_x_tvec = this->rotationMatrix.inv() * this->transVector;
  wcPoint = (Z + invR_x_tvec.at<double>(2, 0)) / invR_x_invM_x_uv1.at<double>(2, 0) * invR_x_invM_x_uv1 - invR_x_tvec;
  cv::Point3f worldCoordinates(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
  return worldCoordinates;
}


/**
 * @brief 返回物体像素坐标
 * 
 * @param rect 
 * @param type 返回质心:0或底部中心:1
 * @return cv::Point2f 
 */
cv::Point2f Run::getPixelPoint(Rect2d &rect, int type) {
    double x = rect.x;
    double y = rect.y;
    double width = rect.width;
    double height = rect.height;
    if (type == 0) {
        return cv::Point2f(x+width/2, y+height/2);   // 质心坐标
    }
    return cv::Point2f(x+width/2, y+height);        // 底部中心坐标
}


void Run::detecEvent(cv::Mat &image) {
  DetectionInfo *detec_info =  objectD->detecRes;
  // 交通逆行


  // 交通拥堵
  if (this->hasDetecEvent[1] && detec_event_index[1] == 0) {
    int left_car = 0;
    int right_car = 0;
    bool flag = false;
    for (int i=0; i < detec_info->track_classIds.size(); i++) {
      // 如果物体id大于0 : car. 且速度小于10
      if (detec_info->track_classIds[i] > 0 && abs(detec_info->track_speeds[i]) < 10) {
        // 计算检测框的质心
        double c_x = detec_info->track_boxes[i].x + detec_info->track_boxes[i].width / 2;
        double c_y = detec_info->track_boxes[i].y + detec_info->track_boxes[i].height / 2;
        cv::Point2d c_point = cv::Point2d(c_x, c_y);
        if (this->left_road_roi.contains(c_point)) left_car += 1;
        if (this->right_road_roi.contains(c_point)) right_car += 1;
      }
    }
    // TODO: 当左侧道路有超过三辆车，就认为交通拥挤。右侧同理
    if (left_car > 3) {
      cv::rectangle(image, this->left_road_roi, Scalar(255, 50, 50), 2);
      flag = true;
    }
    if (right_car > 3) {
      cv::rectangle(image, this->right_road_roi, Scalar(255, 50, 50), 2);
      flag = true;
    }
    if (flag) {
      this->PubEventTopic(1, "交通拥挤", "高", "正确", image);
    }
  }

  // 异常变道


  // 异常停车事件
  if (this->hasDetecEvent[3] && detec_event_index[3] == 0) {
    bool flag = false;
    for (int i=0; i < detec_info->track_classIds.size(); i++) {
      // TODO: 这里直接将速度小于 0.2 的车视为停车。
      // 后面可以优化判断条件，根据最近几次的位置来判断是否
      if (detec_info->track_classIds[i] > 0 && abs(detec_info->track_speeds[i]) < 0.2) {
        flag = true;
        cv::rectangle(image, detec_info->track_boxes[i], Scalar(255, 50, 50), 2);
      }
    }
    if (flag) {
      this->PubEventTopic(3, "异常停车", "高", "正确", image);
    }
  }


  // 弱势交通参与者闯入  默认检测整个画面
  if (hasDetecEvent[4] && detec_event_index[4] == 0) {
    bool flag = false;
    for (int i=0; i < detec_info->track_classIds.size(); i++) {
      if (detec_info->track_classIds[i] == 0) {   // 如果物体id为 0: person
        flag = true;
        cv::rectangle(image, detec_info->track_boxes[i], Scalar(255, 50, 50), 2);
      }
    }
    if (flag) {
      this->PubEventTopic(4, "弱势交通参与者闯入", "高", "正确", image);
    }
  }
}


void Run::PubEventTopic(int type, std::string e_name, std::string level, 
                        std::string judge, cv::Mat &image) {
  detection_fusion::EventInfo eventMsg;
  eventMsg.type = type;
  eventMsg.time = this->getCurTime();
  eventMsg.event_name = e_name;
  eventMsg.level = level;
  eventMsg.judge = judge;
  // cv::Mat tmp;
  // image.copyTo(tmp);
  // eventMsg.img = tmp;
  sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(
                        std_msgs::Header(), "rgb8", image).toImageMsg();
  eventMsg.img = *pub_msg;
  this->pub_event_.publish(eventMsg);
}


std::string Run::getCurTime() {
    time_t timep;
    time (&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&timep) );
    return tmp;
}


void Run::processOD(cv::Mat &image, int interval) {
  DetectionInfo *detec_info = this->objectD->detecRes;
  vector<Rect2d> tmp_bboxs;
  if (detec_info->index % interval == 0) {
    detec_info->track_boxes_pre.clear();
    // 將當前幀檢測結果賦值給 track_boxes_pre
    detec_info->track_boxes_pre.assign(          
      detec_info->track_boxes.begin(), detec_info->track_boxes.end());
    detec_info->track_boxes.clear();
    detec_info->track_classIds.clear();
    detec_info->track_confidences.clear();
    this->objectD->runODModel(image);

    // 計算各個目標的速度和距離
    detec_info->track_distances.clear();
    detec_info->track_speeds.clear();
    detec_info->location.clear();
    if (detec_info->track_boxes_pre.empty() || detec_info->track_boxes.empty() ||
          detec_info->track_boxes_pre.size() != detec_info->track_boxes.size()) {
      // 如果track_boxes_pre没有数据，则表示第一次检测，所有物体速度为0
      detec_info->track_speeds = vector<float>(detec_info->track_boxes.size(), 0.0);
      detec_info->track_distances = vector<float>(detec_info->track_boxes.size(), 0.0);
      // TODO: 这里应该使用上次非0的值
      for (int i=0; i < detec_info->track_boxes.size(); i++) {
        cv::Point3f point(0, 0, 0);
        detec_info->location.push_back(point);
      }
      
    } else {
      // 只有前后两帧 bboxes 长度一样时
      // for (int i=0; i<detec_info->track_boxes.size(); i++) {
      //   // 这里 type=1 表示返回底部中心坐标
      //   cv::Point2f pre_pixel = getPixelPoint(detec_info->track_boxes_pre[i], 1);
      //   cv::Point2f cur_pixel = getPixelPoint(detec_info->track_boxes[i], 1);
      //   // 计算真实世界坐标
      //   cv::Point3f wd_pre = cameraToWorld(pre_pixel);
      //   cv::Point3f wd_cur = cameraToWorld(cur_pixel);
      //   // 計算物體速度
      //   float delta = sqrt(pow(wd_cur.x-wd_pre.x, 2) + pow(wd_cur.y-wd_pre.y, 2));
      //   float speed = delta / float(interval / float(this->output_video_fps));
      //   // 計算物體離相機的距離
      //   float dist_pre = sqrt(pow(wd_pre.x-this->camera_coord.x, 2) + 
      //                           pow(wd_pre.y-this->camera_coord.y, 2));
      //   float dist = sqrt(pow(wd_cur.x-this->camera_coord.x, 2) + 
      //                           pow(wd_cur.y-this->camera_coord.y, 2));
      //   // 定义远离相机速度为正，靠近相机速度为负
      //   if (dist < dist_pre) {speed *= -1;}
      //     detec_info->track_speeds.push_back(speed);
      //     detec_info->track_distances.push_back(dist);
      //     detec_info->location.push_back(wd_cur);
      // }
    }
    
    
    /**
     * 开启某一个事件：
     *    hasDetecEvent[i] = true
     * 关闭某一个事件：
     *    hasDetecEvent[i] = false
     *    detec_event_index[i] == -1
    */
    // 图像帧计数器
    for (int i=0; i<hasDetecEvent.size(); i++) {
      if (hasDetecEvent[i] == true) {
        // 如果事件开启，则相应的计数器开始计数
        detec_event_index[i] = (detec_event_index[i] + 1) % event_detec_interval;
        std::cout << "---: i: " << i << " detec_event_index: " << detec_event_index[i] << std::endl;
      }
    }
    
    // 创建跟踪算法对象
    this->objectD->CreateTracker(image);
    this->cur_track_bboxs.clear();
    // 将当前检测到的Rect存入 this->cur_track_bboxs
    tmp_bboxs.clear();
    tmp_bboxs.assign(detec_info->track_boxes.begin(),
                        detec_info->track_boxes.end());
    this->cur_track_bboxs.push_back(tmp_bboxs);

  } else {  /// 跟踪算法模块
    
    this->objectD->multiTracker->update(image);
    // 如果当前跟踪的物体和检测的物体数量相同
    if (this->objectD->multiTracker->getObjects().size() == detec_info->track_boxes.size()) {
      tmp_bboxs.clear();
      tmp_bboxs.assign(objectD->multiTracker->getObjects().begin(),
                    objectD->multiTracker->getObjects().end());
      this->cur_track_bboxs.push_back(tmp_bboxs);
      // std::cout << "cur_track_bboxs.size(): " << this->cur_track_bboxs.size() << std::endl;
      // 如果当前次迭代为该周期内最后一次跟踪
      // 1. 更新速度、距离、坐标信息
      // 2. 检测事件
      if (this->cur_track_bboxs.size() == (this->object_detec_interval-1)) {
        detec_info->track_distances.clear();
        detec_info->track_speeds.clear();
        detec_info->location.clear();
        for (int i=0; i<detec_info->track_boxes.size(); i++) {
          // // 这里 type=1 表示返回底部中心坐标
          cv::Point2f pre_pixel = getPixelPoint(this->cur_track_bboxs.front()[i], 1);
          cv::Point2f cur_pixel = getPixelPoint(this->cur_track_bboxs.back()[i], 1);
          // // 计算真实世界坐标
          cv::Point3f wd_pre = cameraToWorld(pre_pixel);
          cv::Point3f wd_cur = cameraToWorld(cur_pixel);

          float delta = getDistBetweenTwoDetec(i);
          float speed = delta / float(this->object_detec_interval / float(output_video_fps));

          // 计算物体离相机的距离
          float dist_pre = sqrt(pow(wd_pre.x-this->camera_coord.x, 2) + 
                                pow(wd_pre.y-this->camera_coord.y, 2));
          float dist = sqrt(pow(wd_cur.x-this->camera_coord.x, 2) + 
                                  pow(wd_cur.y-this->camera_coord.y, 2));
          // 定义远离相机速度为正，靠近相机速度为负
          if (dist < dist_pre) {speed *= -1;}
          detec_info->track_speeds.push_back(speed);
          detec_info->track_distances.push_back(dist);
          detec_info->location.push_back(wd_cur);
        }
        cv::Mat detecImg;
        image.copyTo(detecImg);
        this->detecEvent(detecImg);          // 进行交通事件检测
      }
    }
    
  }
  
  detec_info->index = (detec_info->index + 1) % 25;
  // 如果当前帧進行了物體檢測，那麼 track_boxes 保存的是當前幀的結果
  // 如果当前帧没有检测，track_boxes 保存的是先前幀的結果，這樣可以保證方框的連續性
  for (unsigned int i=0; i< detec_info->track_boxes.size(); i++) {
    int x = detec_info->track_boxes[i].x;
    int y = detec_info->track_boxes[i].y;
    int width = detec_info->track_boxes[i].width;
    int height = detec_info->track_boxes[i].height;
    objectD->drawPred(detec_info->track_classIds[i], detec_info->track_confidences[i], 
            detec_info->track_speeds[i], detec_info->track_distances[i], x, y, x+width, y+height, image);
  }
}


// 根据 this->cur_track_bboxs 计算两次物体检测期间的累计距离
float Run::getDistBetweenTwoDetec(int index) {
  float total_dist = 0.0;
  // 计算第一个时刻的世界坐标
  cv::Point2f pixel = getPixelPoint(this->cur_track_bboxs[0][index], 1);
  cv::Point3f wd_pre = cameraToWorld(pixel);
  for (int i=1; i<this->cur_track_bboxs.size(); i++) {
    cv::Point3f wd_cur = cameraToWorld(getPixelPoint(cur_track_bboxs[i][index], 1));
    float dist = sqrt(pow(wd_cur.x-wd_pre.x, 2) + pow(wd_cur.y-wd_pre.y, 2));
    total_dist += dist;
    wd_pre.x = wd_cur.x;    // 更新 wd_pre
    wd_pre.y = wd_cur.y;
  }
  return total_dist;
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "detection_fusion_node");
  Run run;
  
  ros::spin();
  return 0;
}