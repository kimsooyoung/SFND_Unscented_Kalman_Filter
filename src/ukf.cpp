#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  n_x_ = 5;
  n_aug_ = 7;
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];

      float p_x = rho * cos(phi);
      float p_y = rho * sin(phi);
      float v_x = rho_dot * cos(phi);
      float v_y = rho_dot * sin(phi);
      float v = sqrt(v_x * v_x + v_y * v_y);
      x_ << p_x, p_y, v, 0, 0;
    }
    else
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) // check radar usage
    UpdateRadar(meas_package);

  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) // check lidar usage
    UpdateLidar(meas_package);
}

void UKF::Prediction(double delta_t)
{
  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    float p_x = Xsig_aug(0, i);
    float p_y = Xsig_aug(1, i);
    float v = Xsig_aug(2, i);
    float yaw = Xsig_aug(3, i);
    float yawd = Xsig_aug(4, i);
    float nu_a = Xsig_aug(5, i);
    float nu_yawdd = Xsig_aug(6, i);

    // avoid division by zero
    if (yawd != 0)
    {
      Xsig_pred_(0, i) = p_x + (v * (sin(yaw + yawd * delta_t) - sin(yaw)) / yawd) + (.5 * delta_t * delta_t * cos(yaw) * nu_a);
      Xsig_pred_(1, i) = p_y + (v * (-cos(yaw + yawd * delta_t) + cos(yaw)) / yawd) + (.5 * delta_t * delta_t * sin(yaw) * nu_a);
      Xsig_pred_(2, i) = v + 0 + (delta_t * nu_a);
      Xsig_pred_(3, i) = yaw + (yawd * delta_t) + (0.5 * delta_t * delta_t * nu_yawdd);
      Xsig_pred_(4, i) = yawd + 0 + (delta_t * nu_yawdd);
    }
    else
    {
      Xsig_pred_(0, i) = p_x + (v * cos(yaw) * delta_t) + (.5 * delta_t * delta_t * cos(yaw) * nu_a);
      Xsig_pred_(1, i) = p_y + (v * sin(yaw) * delta_t) + (.5 * delta_t * delta_t * sin(yaw) * nu_a);
      Xsig_pred_(2, i) = v + 0 + (delta_t * nu_a);
      Xsig_pred_(3, i) = yaw + (yawd * delta_t) + (0.5 * delta_t * delta_t * nu_yawdd);
      Xsig_pred_(4, i) = yawd + 0 + (delta_t * nu_yawdd);
    }
  }

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{

}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{

}