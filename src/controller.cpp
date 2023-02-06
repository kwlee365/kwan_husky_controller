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
                for (int i = 0; i < PANDA_DOF; i++)
                {
                    q_(i) = dc_.q_(i + WHEEL_DOF + VIRTUAL_DOF);
                    q_dot_(i) = dc_.q_dot_(i + WHEEL_DOF + VIRTUAL_DOF);
                    effort_(i) = dc_.effort_(i + WHEEL_DOF + VIRTUAL_DOF);
                }
                m_dc_.unlock();

                updateKinematicsDynamics();

                computeControlInput();

                printData();
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
                q_desired_ = q_;

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
                case (MODE_CLIK):
                    std::cout << " CLIK control" << std::endl;
                    break;
                case (MODE_NULL):
                    std::cout << " NULL control" << std::endl;
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

    x_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(robot_, q_, BODY_ID, Vector3d(0.0, 0.0, 0.0), true);
    x_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(robot_, q_, BODY_ID, true).transpose();

    // Jacobian
    j_temp_.setZero();
    j_.setZero();
    RigidBodyDynamics::CalcPointJacobian6D(robot_, q_, BODY_ID, Vector3d(0.0, 0.0, 0.0), j_temp_, true);

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
        input_vel << 0.0, 0.0;

        moveJointPosition(q_target_, 3.0);
        // moveJointPositionTorque(q_target_, 5.0);
        moveHuskyPositionVelocity(input_vel);
    }
    else if (mode_ == MODE_HOME)
    {
        q_target_ << 0.0, 0.0, 0.0, -90 * DEG2RAD, 0.0, 90 * DEG2RAD, 45 * DEG2RAD;
        input_vel << 0.2, 0.0;

        moveJointPositionTorque(q_target_, 5.0);
        moveHuskyPositionVelocity(input_vel);
    }
    else if (mode_ == MODE_CLIK)
    {
        Eigen::Vector3d target_position;
        target_position << 0.3, 0.0, 0.8;

        Eigen::Matrix3d target_rotation;
        target_rotation << 0.707, -0.707, 0.0,
            -0.707, -0.707, 0.0,
            0.0, 0.0, -1.0;

        input_vel << 0.0, 0.0;

        // CLIK(target_position, target_rotation, 5.0);

        CLIK_traj();

        moveHuskyPositionVelocity(input_vel);
        moveEndEffector(true);
    }
    else if (mode_ == MODE_NULL)
    {
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

    if (dc_.sim_mode_ == "position")
        dc_.control_input_.segment(WHEEL_DOF, PANDA_DOF) = q_desired_;
    else if (dc_.sim_mode_ == "torque")
        dc_.control_input_.segment(WHEEL_DOF, PANDA_DOF) = torque_desired_;

    dc_.control_input_.segment(WHEEL_DOF + PANDA_DOF, EE_DOF) = ee_;
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

void HuskyController::moveJointPosition(Eigen::Vector7d target_position, double duration)
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

    q_desired_ = q_cubic;
}

void HuskyController::CLIK(Eigen::Vector3d target_position, Eigen::Matrix3d target_rotation, double duration)
{
    Vector6d xd_desired, x_error;

    for (int i = 0; i < 3; i++)
    {
        x_desired_.translation()(i) = DyrosMath::cubic(cur_time_,
                                                       mode_init_time_,
                                                       mode_init_time_ + duration,
                                                       x_mode_init_.translation()(i),
                                                       target_position(i),
                                                       0.0, 0.0);

        xd_desired(i) = DyrosMath::cubicDot(cur_time_,
                                            mode_init_time_,
                                            mode_init_time_ + duration,
                                            x_mode_init_.translation()(i),
                                            target_position(i),
                                            0.0, 0.0);
    }

    x_desired_.linear() = DyrosMath::rotationCubic(cur_time_,
                                                   mode_init_time_,
                                                   mode_init_time_ + duration,
                                                   x_mode_init_.linear(),
                                                   target_rotation);

    xd_desired.segment<3>(3) = DyrosMath::rotationCubicDot(cur_time_,
                                                           mode_init_time_,
                                                           mode_init_time_ + duration,
                                                           Vector3d::Zero(), Vector3d::Zero(),
                                                           x_mode_init_.linear(),
                                                           target_rotation);
    int num = 0;

    while (1)
    {
        Eigen::Isometry3d x_qd_;
        x_qd_ = PositionUpdate(q_desired_);

        x_error.head(3) = x_desired_.translation() - x_qd_.translation();
        x_error.tail(3) = -DyrosMath::getPhi(x_qd_.linear(), x_desired_.linear());

        Eigen::MatrixXd j_qd_;
        j_qd_.setZero();
        j_qd_.resize(6, 7);
        j_qd_ = JacobianUpdate(q_desired_);

        Eigen::Vector6d kp_diag;
        kp_diag.setZero();
        Eigen::Matrix6d kp_;
        kp_.setZero();

        kp_diag << 50, 50, 50, 10, 10, 10;
        kp_ = kp_diag.asDiagonal();

        Vector7d qd_desired = j_qd_.transpose() * (j_qd_ * j_qd_.transpose()).inverse() * (xd_desired + kp_ * x_error);

        if (cur_time_ < mode_init_time_ + duration)
        {
            q_desired_ = q_desired_ + qd_desired / hz_;
        }
        else if (cur_time_ >= mode_init_time_ + duration)
        {
            q_desired_ = q_desired_;
        }

        if (x_error.norm() < 0.01 || num > 10)
            break;

        num++;
    }

    std::cout << "xpos: " << x_.translation().transpose() << "\n"
              << std::endl;
    std::cout << "rotpos: " << x_.linear() << "\n"
              << std::endl;
    std::cout << "xdes: " << x_desired_.translation().transpose() << "\n"
              << std::endl;
    std::cout << "rot_des: " << x_desired_.linear() << "\n"
              << std::endl;
}

void HuskyController::CLIK_traj()
{
    Vector6d xd_desired, x_error;

    if (is_read_ == true)
    {
        tick_limit_ = ReadTextFile(x_traj_, z_traj_, xdot_traj_, zdot_traj_);

        is_read_ = false;
    }

    x_desired_.translation()(0) = x_mode_init_.translation()(0);
    x_desired_.translation()(1) = x_mode_init_.translation()(1) + x_traj_(traj_tick_) / 3;
    x_desired_.translation()(2) = x_mode_init_.translation()(2) + z_traj_(traj_tick_) / 3;

    xd_desired(0) = 0.0;
    xd_desired(1) = xdot_traj_(traj_tick_);
    xd_desired(2) = zdot_traj_(traj_tick_);

    x_desired_.linear() = x_mode_init_.linear();
    xd_desired.tail(3).setZero();

    int num = 0;

    while (1)
    {
        Eigen::Isometry3d x_qd_;
        x_qd_ = PositionUpdate(q_desired_);

        x_error.head(3) = x_desired_.translation() - x_qd_.translation();
        x_error.tail(3) = -DyrosMath::getPhi(x_qd_.linear(), x_desired_.linear());

        Eigen::MatrixXd j_qd_;
        j_qd_.setZero();
        j_qd_.resize(6, 7);
        j_qd_ = JacobianUpdate(q_desired_);

        Eigen::Vector6d kp_diag;
        kp_diag.setZero();
        Eigen::Matrix6d kp_;
        kp_.setZero();

        kp_diag << 50, 50, 50, 10, 10, 10;
        kp_ = kp_diag.asDiagonal();

        Vector7d qd_desired = j_qd_.transpose() * (j_qd_ * j_qd_.transpose()).inverse() * (xd_desired + kp_ * x_error);

        q_desired_ = q_desired_ + qd_desired / hz_;

        if (x_error.norm() < 0.01 || num > 10)
            break;

        num++;
    }

    traj_tick_++;
    if (traj_tick_ == tick_limit_)
        traj_tick_ = 0;
}

unsigned int HuskyController::ReadTextFile(Eigen::VectorXd &x_traj, Eigen::VectorXd &z_traj, Eigen::VectorXd &xdot_traj, Eigen::VectorXd &zdot_traj)
{
    std::string textfile_location = "/home/kwan/catkin_ws/src/husky_controller/traj.txt";

    FILE *traj_file = NULL;
    traj_file = fopen(textfile_location.c_str(), "r");
    int traj_length = 0;
    char tmp;

    if (traj_file == NULL)
    {
        printf("There is no txt file. Please edit code. ");
        return 0;
    }

    while (fscanf(traj_file, "%c", &tmp) != EOF)
    {
        if (tmp == '\n')
            traj_length++;
    }

    fseek(traj_file, 0L, SEEK_SET);
    traj_length -= 1;

    std::cout << traj_length << std::endl;

    double time[traj_length + 1],
        ref_position_x[traj_length + 1],
        ref_position_y[traj_length + 1],
        ref_position_xdot_[traj_length + 1],
        ref_position_ydot_[traj_length + 1];

    for (int i = 0; i < traj_length + 1; i++)
    {
        fscanf(traj_file, "%lf %lf %lf %lf %lf \n",
               &time[i],
               &ref_position_x[i],
               &ref_position_y[i],
               &ref_position_xdot_[i],
               &ref_position_ydot_[i]);
    }

    x_traj.resize(traj_length + 1);
    z_traj.resize(traj_length + 1);
    xdot_traj.resize(traj_length + 1);
    zdot_traj.resize(traj_length + 1);

    for (int i = 0; i < traj_length + 1; i++)
    {
        x_traj(i) = ref_position_x[i];
        z_traj(i) = ref_position_y[i];
        xdot_traj(i) = ref_position_xdot_[i];
        zdot_traj(i) = ref_position_ydot_[i];
    }

    std::cout << "ref_position_x[] " << ref_position_x[31132] << std::endl;
    std::cout << "x_traj_(traj_tick_) " << x_traj_(31132) << std::endl;

    fclose(traj_file);

    return (traj_length + 1);
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

void HuskyController::moveEndEffector(bool mode)
{
    if (mode == false)
        ee_ << 3.0, 3.0;
    else if (mode == true)
        ee_ << -10.0, -10.0;
}

void HuskyController::printData()
{
    // std::cout << "xpos: " << x_.translation().transpose() << "\n"
    //           << std::endl;
    // std::cout << "xdes: " << x_.linear() << "\n"
    //           << std::endl;
    // std::cout << "rot_init: " << x_mode_init_.linear() << "\n"
    //           << std::endl;
    // std::cout << "qpos: " << q_.transpose() << "\n"
    //           << std::endl;
    // std::cout << "qdes: " << q_desired_.transpose() << "\n"
    //           << std::endl;
}

Eigen::MatrixXd HuskyController::JacobianUpdate(Eigen::Vector7d qd_)
{
    static const int BODY_ID = robot_.GetBodyId("panda_link7");

    Eigen::MatrixXd j_temp_, j_qd;
    j_temp_.setZero();
    j_qd.setZero();
    j_temp_.resize(6, 7);
    j_qd.resize(6, 7);
    RigidBodyDynamics::CalcPointJacobian6D(robot_, qd_, BODY_ID, Vector3d::Zero(), j_temp_, true);
    for (int i = 0; i < 2; i++)
    {
        j_qd.block<3, 7>(i * 3, 0) = j_temp_.block<3, 7>(3 - i * 3, 0);
    }

    return j_qd;
}

Eigen::Isometry3d HuskyController::PositionUpdate(Eigen::Vector7d qd_)
{
    static const int BODY_ID = robot_.GetBodyId("panda_link7");

    Isometry3d x_qd;
    x_qd.translation().setZero();
    x_qd.linear().setZero();

    x_qd.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(robot_, qd_, BODY_ID, Vector3d::Zero(), true);
    x_qd.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(robot_, qd_, BODY_ID, true).transpose();

    return x_qd;
}