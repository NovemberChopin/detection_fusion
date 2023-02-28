#include"ros/ros.h" 
#include"std_msgs/Float64MultiArray.h"          
#include"std_msgs/String.h" 
#include<sstream>
#include<vector>
#include<string.h> 
using namespace std; 

vector<double> array_test={0,1,2,3,4,5,6,7,8,9,10,11};

int number_array()
{
    for(int k=0;k<2;k++)
    {
	    array_test[k]=k;
	}	
}
int main(int argc, char** argv) 
{ 	
	ros::init(argc, argv, "talker"); // 节点名称 
	ros::NodeHandle n; 
	ros::Publisher chatter_pub = n.advertise<std_msgs::Float64MultiArray>("chatter", 1000); 
	//ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter",1000); 
	ros::Rate loop_rate(10); 
	int count = 0; 
	while( ros::ok() )
	 { 	
		number_array();
		std_msgs::Float64MultiArray msg;
       		msg.data = array_test;
	    ROS_INFO("data: %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f%6.2f\n",array_test[0],array_test[1],array_test[2],array_test[3],array_test[4],array_test[5],array_test[6],array_test[7],array_test[8],array_test[9],array_test[10],array_test[11]); 
		chatter_pub.publish(msg); 
		ros::spinOnce(); 
		loop_rate.sleep(); 
		++count; 
	} 
	return 0; 
}

