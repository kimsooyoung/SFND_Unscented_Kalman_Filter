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
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 5;

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
  int n_z_ = 2;
  int vec_size = 2 * n_aug_ + 1;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
  for (int i = 0; i < vec_size; i++)
  {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  // calculate mean predicted measurement
  for (int i = 0; i < vec_size; i++)
  {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // transform sigma points into measurement space
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);
  // calculate innovation covariance matrix S
  for (int i = 0; i < vec_size; ++i)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z_, n_z_);
  R.fill(0.0);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;

  S += R;

  MatrixXd Tc = MatrixXd(n_x_, n_z_); // 5 * 2
  Tc.fill(0.0);
  VectorXd z = meas_package.raw_measurements_;
  // calculate cross correlation matrix
  for (int i = 0; i < vec_size; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_; // 5 * 1
    VectorXd z_diff = Zsig.col(i) - z_pred;   // 2 * 1

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse(); // 5 * 2

  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;            // (5 * 2) (2 * 1)
  P_ = P_ - K * S * K.transpose(); // (5 * 2) (2 * 2) (2 * 5)

  float NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
  // std::cout << "NIS_laser_" << NIS_laser_ << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  int n_z_ = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S.fill(0.0);

  int vec_size = 2 * n_aug_ + 1;

  // transform sigma points into measurement space
  for (int i = 0; i < vec_size; i++)
  {
    float p_x = Xsig_pred_(0, i);
    float p_y = Xsig_pred_(1, i);
    float v = Xsig_pred_(2, i);
    float yaw = Xsig_pred_(3, i);

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = v * (p_x * cos(yaw) + p_y * sin(yaw)) / Zsig(0, i);
  }

  // calculate mean predicted measurement
  for (int i = 0; i < vec_size; i++)
    z_pred += weights_(i) * Zsig.col(i);

  // calculate innovation covariance matrix S
  for (int i = 0; i < vec_size; ++i)
  { // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z_, n_z_);
  R.fill(0.0);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;

  S += R;

  MatrixXd Tc = MatrixXd(n_x_, n_z_); // 5 * 3
  Tc.fill(0.0);
  VectorXd z = meas_package.raw_measurements_;

  // calculate cross correlation matrix
  for (int i = 0; i < vec_size; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_; // 5 * 1
    VectorXd z_diff = Zsig.col(i) - z_pred;   // 3 * 1

    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse(); // 5 * 3

  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;            // (5 * 3) (3 * 1)
  P_ = P_ - K * S * K.transpose(); // (5 * 3) (3 * 3) (3 * 5)

  float NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  // std::cout << "NIS_radar_" << NIS_radar_ << std::endl;
}