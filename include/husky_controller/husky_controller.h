#include <mutex>    // thread lock, unlock (thread의 순서를 정하는 느낌)
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>

#include "mujoco_ros_msgs/JointSet.h"

#include "husky_controller/mujoco_interface.h"

#include <rbdl/rbdl.h>
#include <rbdl/addons/urdfreader/urdfreader.h>

#include "math_type_define.h"

#include <fstream>

#include <std_msgs/Float32MultiArray.h>

# define MODE_INIT 105
# define MODE_HOME 104
# define MODE_CLIK 99 
# define MODE_STOP 115
# define MODE_NULL 110

#define WHEEL_DOF 4
#define PANDA_DOF 7
#define EE_DOF 2
#define VIRTUAL_DOF 6

class HuskyController
{
    public:
        HuskyController(ros::NodeHandle &nh, DataContainer &dc, int control_mode);
        ~HuskyController();
        void compute();
        void updateKinematicsDynamics();
        void computeControlInput();
        void moveJointPosition(Eigen::Vector7d target_position, double duration);
        void moveJointPositionTorque(Eigen::Vector7d target_position, double duration);
        void CLIK(Eigen::Vector3d target_position, Eigen::Matrix3d target_rotation, double duration);
        void CLIK_traj();
        unsigned int ReadTextFile(Eigen::VectorXd &x_traj, Eigen::VectorXd &z_traj, Eigen::VectorXd &xdot_traj, Eigen::VectorXd &zdot_traj);
        Eigen::MatrixXd JacobianUpdate(Eigen::Vector7d qd_);
        Eigen::Isometry3d PositionUpdate(Eigen::Vector7d qd_);

        void moveHuskyPositionVelocity(Eigen::Vector2d input_velocity);
        void moveEndEffector(bool mode);
        void printData();
    
    private:
        double hz_ = 2000;
        double cur_time_;
        double pre_time_;
        double init_time_;

        unsigned int traj_tick_ = 0;
        unsigned int tick_limit_;
        Eigen::VectorXd x_traj_, z_traj_, xdot_traj_, zdot_traj_;

        int mode_ = 0;
        double mode_init_time_ =  0.0;
        Eigen::VectorXd q_mode_init_;
        Eigen::VectorXd q_dot_mode_init_;
        Eigen::Isometry3d x_mode_init_;

        std::mutex m_dc_;   // thread의 lock, unlock을 위한 변수 선언.
        std::mutex m_ci_; 
        std::mutex m_ext_; 
        std::mutex m_buffer_; 
        std::mutex m_rbdl_;

        DataContainer &dc_;

        bool is_init_ = true;
        bool is_read_ = true;

        double sim_time_ = 0.0;

        // Robot State
        Eigen::Vector7d q_;
        Eigen::Vector7d q_init_;
        Eigen::Vector7d q_dot_;
        Eigen::Vector7d effort_;

        Eigen::Isometry3d x_;   // Rot + tran
        Eigen::VectorXd x_dot_;
        Eigen::MatrixXd j_temp_;
        Eigen::MatrixXd j_;
        Eigen::MatrixXd j_inverse_;
        Eigen::MatrixXd j_v_;
        Eigen::MatrixXd j_w_;

        // Control
        Eigen::VectorXd q_target_;
        Eigen::VectorXd q_ddot_desired_;
        Eigen::VectorXd q_dot_desired_;
        Eigen::VectorXd q_desired_;

        Eigen::VectorXd torque_desired_;

        Eigen::Isometry3d x_target_;
        Eigen::Isometry3d x_desired_;
        Eigen::VectorXd x_dot_desired_;
        Eigen::Isometry3d x_ddot_desired_;

        Eigen::MatrixXd kv, kp;
        Eigen::MatrixXd kv_task_, kp_task_;

        Eigen::VectorXd control_input_;
        Eigen::Vector7d control_input_filtered_;

        // Kinematics & Dynamics
        RigidBodyDynamics::Model robot_;
        Eigen::VectorXd non_linear_;
        Eigen::MatrixXd m_temp_;
        Eigen::VectorXd g_temp_;
        Eigen::MatrixXd m_;
        Eigen::VectorXd g_;
        Eigen::MatrixXd C_;
        
        Eigen::MatrixXd Lambda_;     

        // Mobile robot
        Eigen::VectorXd input_vel;
        Eigen::VectorXd wheel_vel;

        Eigen::Vector2d ee_;
   
};

// ##### keyboard ##### //
// https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=y851004&logNo=20063499242
// https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=dog9230&logNo=35543543

static struct termios initial_settings, new_settings;   // 인터페이스 제어에 사용되는 변수 (입력, 출력, 제어, 로컬, 특수 제어 문자)

static int peek_character = -1;

void init_keyboard()
{
    tcgetattr(0, &initial_settings);
    new_settings = initial_settings;
    new_settings.c_lflag &= ~ICANON;
    new_settings.c_lflag &= ~ECHO;
    new_settings.c_cc[VMIN] = 1;
    new_settings.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
}

void close_keyboard()
{
    tcsetattr(0, TCSANOW, &initial_settings);
}

int _kbhit()
{
    unsigned char ch;
    int nread;

    if(peek_character != -1)
        return 1;

    new_settings.c_cc[VMIN] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
    nread = read(0, &ch, 1);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0, TCSANOW, &new_settings);

    if(nread == 1)
    {
        peek_character = ch;
        return 1;
    }

    return 0;
}

int _getch()
{
    char ch;
    
    if(peek_character != -1)
    {
        ch = peek_character;
        peek_character = -1;
        return ch;
    }

    read(0, &ch, 1);
    return ch;
}

int _putch(int c)
{
    putchar(c);
    fflush(stdout);

    return c;
}

