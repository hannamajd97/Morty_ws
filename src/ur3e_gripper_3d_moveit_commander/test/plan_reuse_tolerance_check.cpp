// Standalone check of the probe-plan reuse comparison logic.
//
// The robot cannot be driven to verify this change, so the pure part of
// it is verified here instead: this program #includes the SHIPPED header
// (../src/plan_reuse_tolerance.hpp) — not a copy — and feeds it synthetic
// values around the tolerances the executors use.
//
// Build and run without ROS:
//   g++ -std=c++17 -Wall -Wextra -o /tmp/plan_reuse_check
//         src/ur3e_gripper_3d_moveit_commander/test/plan_reuse_tolerance_check.cpp
//   /tmp/plan_reuse_check
// It is also built and run by `colcon test` (see CMakeLists.txt).

#include "../src/plan_reuse_tolerance.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

// The exact defaults declared by both grasp executors.
constexpr double kPosTolM = 1e-4;          // cached_plan_pos_tol_m
constexpr double kOriTolRad = 1e-3;        // cached_plan_ori_tol_rad
constexpr double kJointTolRad = 0.01;      // cached_plan_start_joint_tol_rad
                                           // = MoveIt allowed_start_tolerance

int failures = 0;
int checks = 0;

void expect(bool got, bool want, const std::string & what, const std::string & detail)
{
  ++checks;
  const bool ok = (got == want);
  if (!ok) { ++failures; }
  std::printf("  [%s] %-58s want=%-6s got=%-6s  %s\n",
    ok ? "PASS" : "FAIL", what.c_str(),
    want ? "accept" : "reject", got ? "accept" : "reject", detail.c_str());
}

// Quaternion for a rotation of `ang` radians about +Z.
void quat_z(double ang, double q[4])
{
  q[0] = 0.0;
  q[1] = 0.0;
  q[2] = std::sin(ang / 2.0);
  q[3] = std::cos(ang / 2.0);
}

bool pose_ok(
  const double pa[3], const double qa[4],
  const double pb[3], const double qb[4],
  double * pe, double * oe)
{
  return plan_reuse::pose_matches(
    pa[0], pa[1], pa[2], qa[0], qa[1], qa[2], qa[3],
    pb[0], pb[1], pb[2], qb[0], qb[1], qb[2], qb[3],
    kPosTolM, kOriTolRad, pe, oe);
}

}  // namespace

int main()
{
  std::printf("plan_reuse tolerance check\n");
  std::printf("  position    tol = %.6f m   (%.3f mm)\n", kPosTolM, kPosTolM * 1000.0);
  std::printf("  orientation tol = %.6f rad (%.4f deg)\n",
    kOriTolRad, kOriTolRad * 180.0 / M_PI);
  std::printf("  start joint tol = %.6f rad (MoveIt allowed_start_tolerance)\n\n",
    kJointTolRad);

  char detail[200];

  // ── POSE ─────────────────────────────────────────────────────────
  std::printf("POSE equality (accept only the SAME pose)\n");
  const double p0[3] = {0.075, -0.308, 0.622};   // a real logged grasp
  double q0[4];
  quat_z(0.7, q0);

  {
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p0, q0, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.6f mm / %.6f deg",
      pe * 1000.0, oe * 180.0 / M_PI);
    expect(got, true, "identical pose (bit-for-bit)", detail);
  }
  {
    // 1 mm away in x — a DIFFERENT grasp; must replan.
    double p1[3] = {p0[0] + 0.001, p0[1], p0[2]};
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p1, q0, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.4f mm / %.6f deg (10x the tol)",
      pe * 1000.0, oe * 180.0 / M_PI);
    expect(got, false, "position 1 mm away", detail);
  }
  {
    // 1 degree of yaw — a different wrist angle; must replan.
    double q1[4];
    quat_z(0.7 + 1.0 * M_PI / 180.0, q1);
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p0, q1, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.6f mm / %.4f deg (%.1fx the tol)",
      pe * 1000.0, oe * 180.0 / M_PI, (1.0 * M_PI / 180.0) / kOriTolRad);
    expect(got, false, "orientation 1 deg away", detail);
  }
  {
    // Just inside the position tolerance: 0.09 mm.
    double p1[3] = {p0[0] + 0.00009, p0[1], p0[2]};
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p1, q0, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.4f mm", pe * 1000.0);
    expect(got, true, "position 0.09 mm away (inside 0.1 mm tol)", detail);
  }
  {
    // Just outside: 0.11 mm.
    double p1[3] = {p0[0] + 0.00011, p0[1], p0[2]};
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p1, q0, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.4f mm", pe * 1000.0);
    expect(got, false, "position 0.11 mm away (outside 0.1 mm tol)", detail);
  }
  {
    // q and -q are the SAME rotation. A signed dot would call this 180
    // deg off and throw the plan away on every second pick.
    double qneg[4] = {-q0[0], -q0[1], -q0[2], -q0[3]};
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p0, qneg, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.6f deg", oe * 180.0 / M_PI);
    expect(got, true, "negated quaternion (double cover, same rotation)", detail);
  }
  {
    // 180 degrees of yaw is the wrist-twin; genuinely different here.
    double q1[4];
    quat_z(0.7 + M_PI, q1);
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p0, q1, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.2f deg", oe * 180.0 / M_PI);
    expect(got, false, "orientation 180 deg away (wrist twin)", detail);
  }
  {
    // A zero quaternion is not a rotation: never claim a match.
    double qzero[4] = {0.0, 0.0, 0.0, 0.0};
    double pe = 0, oe = 0;
    const bool got = pose_ok(p0, q0, p0, qzero, &pe, &oe);
    std::snprintf(detail, sizeof(detail), "err %.2f deg", oe * 180.0 / M_PI);
    expect(got, false, "degenerate zero quaternion", detail);
  }

  // ── JOINTS ───────────────────────────────────────────────────────
  std::printf("\nSTART-STATE equality (accept only if the arm has not moved)\n");
  const std::vector<double> j0 =
    {1.5708, -1.2217, 1.0472, -1.3963, -1.5708, 0.0};

  {
    double me = 0; std::size_t wi = 0;
    const bool got = plan_reuse::joints_match(j0, j0, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "max err %.6f rad", me);
    expect(got, true, "identical joint vector", detail);
  }
  {
    // 0.005 rad on every joint: controller settling noise, still "here".
    std::vector<double> j1 = j0;
    for (auto & v : j1) { v += 0.005; }
    double me = 0; std::size_t wi = 0;
    const bool got = plan_reuse::joints_match(j0, j1, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "max err %.4f rad on joint %zu (half the tol)", me, wi);
    expect(got, true, "every joint 0.005 rad off", detail);
  }
  {
    // 0.05 rad: ~2.9 deg. The arm has moved. Executing the cached
    // trajectory would close that gap at trajectory speed.
    std::vector<double> j1 = j0;
    for (auto & v : j1) { v += 0.05; }
    double me = 0; std::size_t wi = 0;
    const bool got = plan_reuse::joints_match(j0, j1, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "max err %.4f rad on joint %zu (5x the tol)", me, wi);
    expect(got, false, "every joint 0.05 rad off", detail);
  }
  {
    // ONE joint moved. Must reject: it is a max-norm, not a mean.
    std::vector<double> j1 = j0;
    j1[3] += 0.05;
    double me = 0; std::size_t wi = 0;
    const bool got = plan_reuse::joints_match(j0, j1, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "max err %.4f rad on joint %zu", me, wi);
    expect(got, false, "a single joint 0.05 rad off, rest identical", detail);
  }
  {
    // The comparison is inclusive (<=). Checked against 0.0 so the
    // difference is EXACTLY the double nearest 0.01 with no rounding of
    // our own: adding 0.01 to a joint angle like 1.5708 lands a few ulps
    // ABOVE 0.01 and would make this test measure C++ arithmetic rather
    // than the predicate.
    double me = 0; std::size_t wi = 0;
    bool got = plan_reuse::joints_match({0.0}, {kJointTolRad}, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "max err %.17g rad", me);
    expect(got, true, "exactly the tolerance apart (boundary is inclusive)", detail);
    got = plan_reuse::joints_match({0.0}, {kJointTolRad + 1e-6}, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "max err %.6f rad", me);
    expect(got, false, "one micro-radian past the tolerance", detail);
    // ...and the realistic version, which shows the ulp wobble: nominally
    // 0.01 rad off a real joint angle, and REJECTED, because the true
    // difference is 0.010000000000000231. The guard fails safe.
    std::vector<double> j1 = j0; j1[0] += kJointTolRad;
    got = plan_reuse::joints_match(j0, j1, kJointTolRad, &me, &wi);
    std::snprintf(detail, sizeof(detail), "true err %.17g rad (nominal 0.01)", me);
    expect(got, false, "1.5708 + 0.01: a few ulps over, so rejected", detail);
  }
  {
    // Cannot compare ⇒ cannot execute.
    std::vector<double> shortv(j0.begin(), j0.end() - 1);
    double me = 0; std::size_t wi = 0;
    bool got = plan_reuse::joints_match(j0, shortv, kJointTolRad, &me, &wi);
    expect(got, false, "size mismatch (6 vs 5 joints)", "unprovable => rejected");
    got = plan_reuse::joints_match({}, {}, kJointTolRad, &me, &wi);
    expect(got, false, "both vectors empty", "unprovable => rejected");
  }

  std::printf("\n%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
