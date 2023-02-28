

#include "object_detection.hpp"

ObjectDetection::ObjectDetection() {
    std::cout << "ObjectDetection" << std::endl;
    
    detecRes = new DetectionInfo(0);
    
    // 讀取目標分類文本
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) 
		classes.push_back(line);

	// 加载模型
	net = readNetFromDarknet(modelConfiguration, modelWeights);
	cout << "Using CPU device" << endl;
    net.setPreferableBackend(DNN_TARGET_CPU);

    // // 创建跟踪对象
    multiTracker = cv::MultiTracker::create();
}

ObjectDetection::~ObjectDetection() {

}

void ObjectDetection::hello() {
    std::cout << "hello" << std::endl;
}

Ptr<Tracker> ObjectDetection::createTrackerByName(string trackerType) 
{
    Ptr<Tracker> tracker;
    if (trackerType ==  trackerTypes[0])
        tracker = TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
        std::cout << " " << *it << endl;
    }
    return tracker;
}


void ObjectDetection::runTrackerModel(cv::Mat& frame) {
    // 跟踪
    multiTracker->update(frame);

    // draw tracked objects
    for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
    {
      rectangle(frame, multiTracker->getObjects()[i], Scalar(255, 178, 50), 2, 1);
    }
    // imshow("Tracker", frame);
}

/**
 * @brief 物体检测主函数
 * 
 * @param frame 
 * @param cam_index 当前相机的 index
 */
void ObjectDetection::runODModel(cv::Mat& frame) {
	cv::Mat blob;
	// Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

	//Sets the input to the network
	net.setInput(blob);
	// Runs the forward pass to get output of the output layers
	std::vector<cv::Mat> outs;
	net.forward(outs, getOutputsNames(net));
	// Remove the bounding boxes with low confidence
    
    postprocess(frame, outs);
    // std::cout << "forward time: " << double(mid-start)/CLOCKS_PER_SEC << " after time: " << double(end-mid)/CLOCKS_PER_SEC << std::endl;
	// Put efficiency information. 
    // The function getPerfProfile returns the overall time for inference(t) 
    // and the timings for each of the layers(in layersTimes)
	// vector<double> layersTimes;
	// double freq = getTickFrequency() / 1000;
	// double t = net.getPerfProfile(layersTimes) / freq;
	// string label = format("Inference time for a frame : %.2f ms", t);
	// putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    // std::cout << "boxes size end: " << boxes.size() << std::endl;

}


/**
 * @brief Remove the bounding boxes with low confidence using non-maxima suppression
 * 
 * @param frame 
 * @param outs network output
 * @param cam_index the index of the camera
 */
void ObjectDetection::postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold ||  (confidence > 0.2 && classIdPoint.x > 0 && classIdPoint.x < 7))
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        if(classIds[idx] == 0) {     // 只保存检测到的行人和车辆（class见resources/coco.names）
            detecRes->track_boxes.push_back(boxes[idx]);
            detecRes->track_classIds.push_back(classIds[idx]);
            detecRes->track_confidences.push_back(confidences[idx]);
            // track_boxes.push_back(boxes[idx]);
            // track_classIds.push_back(classIds[idx]);
            // track_confidences.push_back(confidences[idx]);
            // 此时仅仅执行目标检测，在main_window文件中添加框
            // drawPred(classIds[idx], confidences[idx], box.x, box.y,
            //         box.x + box.width, box.y + box.height, frame);
        }
        if(classIds[idx] > 0 && classIds[idx] < 7) {     // 只保存检测到的行人和车辆（class见resources/coco.names）
            detecRes->track_boxes.push_back(boxes[idx]);
            detecRes->track_classIds.push_back(classIds[idx]);
            detecRes->track_confidences.push_back(confidences[idx]);
        }
    }
}


// Draw the predicted bounding box
void ObjectDetection::drawPred(int classId, float conf, float speed, float dist,
                                int left, int top, int right, int bottom, Mat& frame) {
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    string speed_label = format("%.2f", speed);
    string dist_label = format("%.2f", dist);
    if (!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        if(classId > 0 && classId < 7) classId = 1;     // 把检测到的各种车辆都显示为 car
        label = classes[classId] + ":" + speed_label + " " + dist_label;
        // label = classes[classId];
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 178, 50), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),2);
}


// Get the names of the output layers
vector<String> ObjectDetection::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}