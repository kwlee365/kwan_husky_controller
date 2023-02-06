#include <ros/ros.h>    // ROS 노드 설정
#include <std_msgs/String.h> // ROS 송,수신 메시지 타입 정보

#include <thread>   // Thread

#include "husky_controller/husky_controller.h"
#include "husky_controller/mujoco_interface.h"

#define TorqueControl 1
#define PositionControl 0

/**********************************************************************************/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "husky_controller");   // node 초기화
    ros::NodeHandle nh; // 
    DataContainer dc;

    // int control_mode = TorqueControl;
    int control_mode = PositionControl;

    MujocoInterface mujoco_interface(nh, dc);   // Class 정의
    HuskyController husky_controller(nh, dc, control_mode);

    std::thread thread[3];

    thread[0] = std::thread(&MujocoInterface::stateUpdate, &mujoco_interface);
    thread[1] = std::thread(&HuskyController::compute, &husky_controller);
    thread[2] = std::thread(&MujocoInterface::sendCommand, &mujoco_interface, control_mode);

    for (int i = 0; i < 3; i++)
        thread[i].join();   // 다른 스레드가 종료될 때까지 기다림.

    return 0;
}
