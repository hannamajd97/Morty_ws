#ifndef UR3E_GRIPPER_3D_MOVEIT_COMMANDER__PLAN_REUSE_TOLERANCE_HPP_
#define UR3E_GRIPPER_3D_MOVEIT_COMMANDER__PLAN_REUSE_TOLERANCE_HPP_

// ── Probe-plan reuse: the pure comparison logic ──────────────────────
// 2026-08-28. The grasp executors keep the Pilz LIN trajectory that the
// candidate probe already paid 6-12 s for, and execute THAT in Step 2
// instead of asking for the identical plan a second time. A stored
// joint-space trajectory is only safe to execute if it is still the same
// pose and the arm is still standing on its first point, so these two
// predicates are the load-bearing part of the whole change.
//
// Deliberately free of ROS / MoveIt / Eigen types: there is exactly ONE
// copy of this logic, and the standalone tolerance check
// (test/plan_reuse_tolerance_check.cpp) compiles this very header with a
// plain g++ so the numbers in the report come from the shipped code and
// not from a re-typed approximation of it.

#include <cmath>
#include <cstddef>
#include <vector>

namespace plan_reuse
{

/// Euclidean distance between two positions, metres.
inline double position_error_m(
  double ax, double ay, double az,
  double bx, double by, double bz)
{
  const double dx = ax - bx;
  const double dy = ay - by;
  const double dz = az - bz;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// Smallest rotation angle between two quaternions, radians.
/// std::fabs on the dot product collapses the double cover: q and -q are
/// the SAME rotation, and TF / GraspGen hand us either sign freely — a
/// signed comparison would report a perfect match as 180 degrees off.
/// Non-unit or zero quaternions return pi, i.e. "never matches".
inline double orientation_error_rad(
  double ax, double ay, double az, double aw,
  double bx, double by, double bz, double bw)
{
  const double na = std::sqrt(ax * ax + ay * ay + az * az + aw * aw);
  const double nb = std::sqrt(bx * bx + by * by + bz * bz + bw * bw);
  if (!(na > 0.0) || !(nb > 0.0)) {
    return M_PI;
  }
  double dot = (ax * bx + ay * by + az * bz + aw * bw) / (na * nb);
  dot = std::fabs(dot);
  if (dot > 1.0) {
    dot = 1.0;  // guard acos() against FP overshoot
  }
  return 2.0 * std::acos(dot);
}

/// "Is this the very same pose the cached plan was built for?"
/// Both position AND orientation must be inside tolerance. The errors are
/// always written out, including on rejection, so the caller can log how
/// far off it was.
inline bool pose_matches(
  double ax, double ay, double az, double aqx, double aqy, double aqz, double aqw,
  double bx, double by, double bz, double bqx, double bqy, double bqz, double bqw,
  double pos_tol_m, double ori_tol_rad,
  double * pos_err_out = nullptr, double * ori_err_out = nullptr)
{
  const double pe = position_error_m(ax, ay, az, bx, by, bz);
  const double oe = orientation_error_rad(aqx, aqy, aqz, aqw, bqx, bqy, bqz, bqw);
  if (pos_err_out) { *pos_err_out = pe; }
  if (ori_err_out) { *ori_err_out = oe; }
  return pe <= pos_tol_m && oe <= ori_tol_rad;
}

/// "Is the arm still standing exactly where this trajectory starts?"
/// Element-wise |a-b| <= tol_rad. THE safety predicate: a false positive
/// here hands the controller a trajectory whose first point is somewhere
/// the arm is not, and the controller closes that gap at speed.
///
/// A size mismatch or an empty vector is a HARD reject — an unprovable
/// start state is a rejected start state. Callers must pair the joint
/// values by NAME before calling; /joint_states ordering is not the
/// trajectory's ordering.
inline bool joints_match(
  const std::vector<double> & a, const std::vector<double> & b,
  double tol_rad,
  double * max_err_out = nullptr, std::size_t * worst_idx_out = nullptr)
{
  if (max_err_out) { *max_err_out = 0.0; }
  if (worst_idx_out) { *worst_idx_out = 0; }
  if (a.empty() || a.size() != b.size()) {
    return false;
  }
  double worst = 0.0;
  std::size_t worst_i = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = std::fabs(a[i] - b[i]);
    if (e > worst) {
      worst = e;
      worst_i = i;
    }
  }
  if (max_err_out) { *max_err_out = worst; }
  if (worst_idx_out) { *worst_idx_out = worst_i; }
  return worst <= tol_rad;
}

}  // namespace plan_reuse

#endif  // UR3E_GRIPPER_3D_MOVEIT_COMMANDER__PLAN_REUSE_TOLERANCE_HPP_
