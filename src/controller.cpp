#include "husky_controller/husky_controller.h"

using namespace Eigen;

HuskyController::HuskyController(ros::NodeHandle &nh, DataContainer &dc, int control_mode) : dc_(dc)
{
    if (control_mode == 0)
        dc_.sim_mode_ = "position";
    else if (control_mode == 1)
        dc_.sim_mode_ = "torque";

    // RBDL
    bool floating_base = false;
    bool verbose = false;

    std::string urdf_name = ros::package::getPath("husky_description") + "/robots/panda_arm.urdf";
    std::cout << "Model name: " << urdf_name << std::endl;
    RigidBodyDynamics::Addons::URDFReadFromFile(urdf_name.c_str(), &robot_, floating_base, verbose);
    // https://rbdl.github.io/d6/db6/namespace_rigid_body_dynamics_1_1_addons.html#af01cd376f05076b855ad55a625d95065

    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Keyboard
    init_keyboard();
}

HuskyController::~HuskyController()
{

}

void HuskyController::compute()
{
    ros::Rate r(hz_);

    while (ros::ok())
    {
        if (!dc_.is_first_callback)
        {
            if (is_init_)
            {
                // Robot state
                q_.setZero();
                q_dot_.setZero();
                effort_.setZero();

                j_temp_.resize(6, PANDA_DOF);
                j_temp_.setZero();
                j_.resize(6, PANDA_DOF);
                j_.setZero();
                j_inverse_.resize(PANDA_DOF, 6);
                j_.setZero();
                j_v_.resize(3, PANDA_DOF);
                j_v_.setZero();
                j_w_.resize(3, PANDA_DOF);
                j_w_.setZero();

                x_.linear().setZero();
                x_.translation().setZero();
                x_dot_.resize(6);
                x_dot_.setZero();
                x_dot_desired_.resize(6);
                x_dot_desired_.setZero();

                // Control
                q_ddot_desired_.resize(PANDA_DOF);
                q_ddot_desired_.setZero();
                q_dot_desired_.resize(PANDA_DOF);
                q_dot_desired_.setZero();
                q_desired_.resize(PANDA_DOF);
                q_desired_.setZero();
                torque_desired_.resize(PANDA_DOF);
                torque_desired_.setZero();
                q_target_.resize(PANDA_DOF);
                q_target_.setZero();

                kp.resize(PANDA_DOF, PANDA_DOF);
                kp.setZero();
                kv.resize(PANDA_DOF, PANDA_DOF);
                kv.setZero();
                kp_task_.resize(6, 6);
                kp_task_.setZero();
                kv_task_.resize(6, 6);
                kv_task_.setZero();

                // Control gain
                for (int i = 0; i < PANDA_DOF; i++)
                {
                    kp(i, i) = 400;
                    kv(i, i) = 100;
                }
                for (int i = 0; i < 6; i++)
                {
                    kp_task_(i, i) = 4900;
                    kv_task_(i, i) = 140;
                }

                control_input_.resize(dc_.num_dof_);
                control_input_.setZero();
                control_input_filtered_.setZero();

                non_linear_.resize(PANDA_DOF);
                non_linear_.setZero();
                g_.resize(PANDA_DOF);
                g_.setZero();
                m_.resize(PANDA_DOF, PANDA_DOF);
                m_.setZero();
                C_.resize(PANDA_DOF, PANDA_DOF);
                C_.setZero();
                Lambda_.resize(6, 6);
                Lambda_.setZero();

                // Mobile
                input_vel.resize(2);
                input_vel.setZero();
                wheel_vel.resize(4);
                wheel_vel.setZero();

                init_time_ = ros::Time::now().toSec(); // 초 단위로 시간을 반환.

                is_init_ = false;
            }
            else
            {
                // std::cout << "Hey, init, are you done?" << std::endl;

                cur_time_ = ros::Time::now().toSec() - init_time_;

                m_dc_.lock();
                sim_time_ = dc_.sim_time_;
                for (int i = 0; i < PANDA_DOF;i++)
                {
                    q_(i) = dc_.q_(i + WHEEL_DOF + VIRTUAL_DOF);
                    q_dot_(i) = dc_.q_dot_(i + WHEEL_DOF + VIRTUAL_DOF);
                    effort_(i) = dc_.effort_(i + WHEEL_DOF + VIRTUAL_DOF);
                }
                m_dc_.unlock();

                updateKinematicsDynamics();

                computeControlInput();

                pre_time_ = cur_time_;
            }

            if (_kbhit())
            {
                int ch = _getch();
                _putch(ch);
                mode_ = ch;

                mode_init_time_ = ros::Time::now().toSec() - init_time_;
                q_mode_init_ = q_;
                q_dot_mode_init_ = q_dot_;
                x_mode_init_ = x_;

                std::cout << "Mode changed to"; 
                // i: 105, r: 114, m: 109, s: 115, f:102, h: 104

                switch (mode_)
                {
                case (MODE_INIT):
                    std::cout << " Init Pose" << std::endl;
                    break;
                case (MODE_HOME):
                    std::cout << " Home Pose" << std::endl;
                    break;
                case (MODE_STOP):
                    std::cout << " Stop Pose" << std::endl;
                    break;
                }
            }

            ros::spinOnce();
            // r.sleep();
        }
    }
    close_keyboard();
}

void HuskyController::updateKinematicsDynamics()
{
    static const int BODY_ID = robot_.GetBodyId("panda_link7");

    // Forward kinematics
    x_.translation().setZero();
    x_.linear().setZero();

    x_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(robot_, q_, BODY_ID, Vector3d::Zero(), true);
    x_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(robot_, q_, BODY_ID, true).transpose();

    // Jacobian
    j_temp_.setZero();
    j_.setZero();
    RigidBodyDynamics::CalcPointJacobian6D(robot_, q_, BODY_ID, Vector3d::Zero(), j_temp_, true);

    for (int i = 0; i < 2; i++)
    {
        j_.block<3, PANDA_DOF>(i * 3, 0) = j_temp_.block<3, PANDA_DOF>(3 - i * 3, 0);
    }

    j_v_ = j_.block<3, PANDA_DOF>(0, 0);
    j_w_ = j_.block<3, PANDA_DOF>(3, 0);

    x_dot_ = j_ * q_dot_;


    // Coriollis + gravity
    non_linear_.setZero();
    RigidBodyDynamics::NonlinearEffects(robot_, q_, q_dot_, non_linear_);

    // Only gravity
    RigidBodyDynamics::NonlinearEffects(robot_, q_, Vector7d::Zero(), g_);

    // Mass matrix
    RigidBodyDynamics::CompositeRigidBodyAlgorithm(robot_, q_, m_, true);

    j_inverse_.setZero();
    Lambda_.setZero();
    Lambda_ = (j_ * m_.inverse() * j_.transpose()).inverse();
    j_inverse_ = m_.inverse() * j_.transpose() * Lambda_;

}

void HuskyController::computeControlInput()
{
    if (mode_ == MODE_INIT)
    {
        q_target_ << 0.0, 0.0, 0.0, -M_PI / 2., 0.0, 0, 0;
        input_vel << 0.0, 2.0;

        moveJointPositionTorque(q_target_, 5.0);
        moveHuskyPositionVelocity(input_vel);
    }
    else if (mode_ == MODE_HOME)
    {
        q_target_ << 0.0, -3.14 / 6, 0.0, -3.14 / 6, 0.0, 0.0, 0.0;
        input_vel << 0.2, 0.0;

        moveJointPositionTorque(q_target_, 5.0);
        moveHuskyPositionVelocity(input_vel);
    }
    else if (mode_ == MODE_STOP)
    {
        torque_desired_ = g_;
        wheel_vel.setZero();
    }
    else
    {
        torque_desired_ = g_;
        wheel_vel.setZero();
    }

    dc_.control_input_.setZero();
    dc_.control_input_.segment(0, WHEEL_DOF) = wheel_vel;
    dc_.control_input_.segment(WHEEL_DOF, PANDA_DOF) = torque_desired_;
    // Virtual DOFs are located in last 6 columns
}

void HuskyController::moveJointPositionTorque(Eigen::Vector7d target_position, double duration)
{
    Vector7d q_cubic, qd_cubic, zero_vector;

    zero_vector.setZero();
    q_cubic = DyrosMath::cubicVector<7>(cur_time_,
                                        mode_init_time_,
                                        mode_init_time_ + duration,
                                        q_mode_init_,
                                        target_position,
                                        zero_vector,
                                        zero_vector);
    torque_desired_ = m_ * (kp * (q_cubic - q_) + kv * (-q_dot_)) + g_;
}

void HuskyController::moveHuskyPositionVelocity(Eigen::Vector2d input_velocity)
{
    // http://wiki.ros.org/diff_drive_controller //
    double V = input_velocity(0);
    double w = input_velocity(1);

    double b = 0.28545;
    double r = 0.1651;

    double wL = (V - w * b / 2) / r;
    double wR = (V + w * b / 2) / r;

    wheel_vel(0) = wL;
    wheel_vel(2) = wL;
    wheel_vel(1) = wR;
    wheel_vel(3) = wR;
}

void HuskyController::printData()
{
    std::cout << "Is it node alive?" << std::endl;
}