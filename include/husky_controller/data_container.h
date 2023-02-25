#include <Eigen/Dense>
#include <ros/ros.h>

#ifndef DataContainer_H
#define DataContainer_H

class DataContainer
{

public:
    ros::NodeHandle nh;

    bool is_first_callback = true;

    double sim_time_;

    double hz_ = 100.0; // Simulation hz

    std::string sim_mode_ = "torque";

    int num_dof_;

    Eigen::VectorXd q_;
    Eigen::VectorXd q_dot_;
    Eigen::VectorXd effort_;

    Eigen::Vector3d force_;
    Eigen::Vector3d torque_;
    Eigen::Vector3d lin_vel_;
    Eigen::Vector3d ang_vel_;
    Eigen::Vector3d pos_;

    double contact_;

    Eigen::VectorXd control_input_;
};

#endif
