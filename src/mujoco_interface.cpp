#include "husky_controller/mujoco_interface.h"

MujocoInterface::MujocoInterface(ros::NodeHandle &nh, DataContainer &dc) : dc_(dc)
{
    mujoco_sim_status_sub_ = nh.subscribe("/mujoco_ros_interface/sim_status", 1, &MujocoInterface::simStatusCallback, this, ros::TransportHints().tcpNoDelay(true));
    mujoco_joint_set_pub_ = nh.advertise<mujoco_ros_msgs::JointSet>("/mujoco_ros_interface/joint_set", 100);

    /* ex */
    // ros::Publisher pub = nh.advertise<std_msgs::String>("topic_name", 5);
    // ros::Subscriber sub = nh.subscribe("my_topic", 1, callback);
}

MujocoInterface::~MujocoInterface()
{
}

void MujocoInterface::stateUpdate() // Thread 1
{
    ros::Rate r(dc_.hz_); // 2000 Hz
    while (ros::ok())
    {
        ros::spinOnce();
        r.sleep(); // 2000 Hz에서 남은 시간만큼 sleep
    }
}

void MujocoInterface::simStatusCallback(const mujoco_ros_msgs::SimStatusConstPtr &msg)
{
    if (is_first_callback) // 첫 콜백이 시작하면
    {
        // DataContainer class에 있는 변수를 message 값으로 할당

        dc_.num_dof_ = msg->name.size();

        dc_.sim_time_ = msg->time;

        dc_.q_.resize(dc_.num_dof_);
        dc_.q_dot_.resize(dc_.num_dof_);
        dc_.effort_.resize(dc_.num_dof_);

        dc_.q_.setZero();
        dc_.q_dot_.setZero();
        dc_.effort_.setZero();

        for (int j = 0; j < dc_.num_dof_; j++)
        {
            dc_.q_[j] = msg->position[j];
            dc_.q_dot_[j] = msg->velocity[j];
            dc_.effort_[j] = msg->effort[j];
        }

        dc_.control_input_.resize(dc_.num_dof_);
        dc_.control_input_.setZero();

        mujoco_joint_set_msg_.torque.resize(dc_.num_dof_); // msg 파일을 통해 받은 torque, position resize
        mujoco_joint_set_msg_.position.resize(dc_.num_dof_);

        is_first_callback = false;
        dc_.is_first_callback = is_first_callback;
    }
    else // 첫번째 콜백 이후에는 관절 위치, 속도, 힘, 시간만 받아옴.
    {
        for (int j = 0; j < dc_.num_dof_; j++)
        {
            dc_.q_[j] = msg->position[j];
            dc_.q_dot_[j] = msg->velocity[j];
            dc_.effort_[j] = msg->effort[j];
        }

        for (int i = 0; i < msg->sensor.size(); i++)
        {
            if (msg->sensor[i].name == "Force_sensor")
            {
                for (int j = 0; j < 3; j++)
                    dc_.force_[j] = msg->sensor[i].data[j];
            }
            if (msg->sensor[i].name == "Torque_sensor")
            {
                for (int j = 0; j < 3; j++)
                    dc_.torque_[j] = msg->sensor[i].data[j];
            }
            if (msg->sensor[i].name == "Contact_sensor")
            {
                dc_.contact_ = msg->sensor[i].data[0];
            }
            if (msg->sensor[i].name == "position_sensor")
            {
                for (int j = 0; j < 3; j++)
                    dc_.pos_[j] = msg->sensor[i].data[j];
            }
            if (msg->sensor[i].name == "linear_velocity_sensor")
            {
                for (int j = 0; j < 3; j++)
                    dc_.lin_vel_[j] = msg->sensor[i].data[j];
            }
            if (msg->sensor[i].name == "angular_velocity_sensor")
            {
                for (int j = 0; j < 3; j++)
                    dc_.ang_vel_[j] = msg->sensor[i].data[j];
            }
        }

        dc_.sim_time_ = msg->time;
    }
}

void MujocoInterface::sendCommand(int control_mode) // Thread 3 (compute -> MuJoCo)
{
    ros::Rate r(dc_.hz_);

    while (ros::ok())
    {
        if (!is_first_callback) // 첫 콜백 이후
        {
            if (control_mode == 1)
            {
                if (dc_.sim_mode_ == "torque")
                {
                    mujoco_joint_set_msg_.MODE = 1;

                    for (int i = 0; i < dc_.num_dof_; i++)
                    {
                        mujoco_joint_set_msg_.torque[i] = dc_.control_input_[i];
                    }
                }
                else
                {
                    std::cout << "COMMAND DISMATCH! -- mujoco mode : position, controller mode : torque!" << std::endl;
                }
            }
            else if (control_mode == 0)
            {
                if (dc_.sim_mode_ == "position")
                {
                    mujoco_joint_set_msg_.MODE = 0;

                    for (int i = 0; i < dc_.num_dof_; i++)
                    {
                        mujoco_joint_set_msg_.position[i] = dc_.control_input_[i];
                    }
                }
                else
                {
                    std::cout << "COMMAND DISMATCH! -- mujoco mode : torque, controller mode : position!" << std::endl;
                }
            }

            mujoco_joint_set_msg_.header.stamp = ros::Time::now();
            mujoco_joint_set_msg_.time = dc_.sim_time_;
            mujoco_joint_set_pub_.publish(mujoco_joint_set_msg_); // mujoco_joint_set_msg_에 다 집어넣은 다음에 퍼블리시
        }

        r.sleep();
    }
}