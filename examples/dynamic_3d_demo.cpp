#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <solver/tiny_api.hpp>
#include <solver/psd_support.hpp>

#ifndef TINYSDP_SCENARIO_SLUG
#define TINYSDP_SCENARIO_SLUG "sweeping_barrier"
#endif

namespace {

constexpr int NX0 = 6;   // x, y, z, vx, vy, vz
constexpr int NU0 = 3;   // ax, ay, az
constexpr int N   = 10;

using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

struct MovingSphere {
    tinytype cx0;
    tinytype cy0;
    tinytype cz0;
    tinytype vx;
    tinytype vy;
    tinytype vz;
    tinytype radius;
    tinytype wobble_x;
    tinytype wobble_x_freq;
    tinytype wobble_x_phase;
    tinytype wobble_y;
    tinytype wobble_y_freq;
    tinytype wobble_y_phase;
    tinytype wobble_z;
    tinytype wobble_z_freq;
    tinytype wobble_z_phase;

    std::array<tinytype, 4> sphere_at_time(tinytype t) const {
        double td = static_cast<double>(t);
        double xarg = static_cast<double>(wobble_x_freq) * td + static_cast<double>(wobble_x_phase);
        double yarg = static_cast<double>(wobble_y_freq) * td + static_cast<double>(wobble_y_phase);
        double zarg = static_cast<double>(wobble_z_freq) * td + static_cast<double>(wobble_z_phase);
        tinytype cx = cx0 + vx * t + wobble_x * tinytype(std::sin(xarg));
        tinytype cy = cy0 + vy * t + wobble_y * tinytype(std::cos(yarg));
        tinytype cz = cz0 + vz * t + wobble_z * tinytype(std::sin(zarg));
        return {cx, cy, cz, radius};
    }
};

bool sphere_has_motion(const MovingSphere& s) {
    auto nz = [](tinytype v) { return std::abs(static_cast<double>(v)) > 1e-9; };
    return nz(s.vx) || nz(s.vy) || nz(s.vz)
        || nz(s.wobble_x) || nz(s.wobble_y) || nz(s.wobble_z)
        || nz(s.wobble_x_freq) || nz(s.wobble_y_freq) || nz(s.wobble_z_freq);
}

struct DynamicSpheres {
    std::vector<MovingSphere> agents;
    tinytype dt = tinytype(1.0);

    std::vector<std::array<tinytype, 4>> spheres_at_step(int step) const {
        tinytype t = dt * tinytype(step);
        std::vector<std::array<tinytype, 4>> spheres;
        spheres.reserve(agents.size());
        for (const auto& agent : agents) {
            spheres.push_back(agent.sphere_at_time(t));
        }
        return spheres;
    }

    std::vector<std::vector<std::array<tinytype, 4>>> horizon_spheres_per_stage(
        int step,
        int horizon,
        tinytype inflation_rate) const {
        std::vector<std::vector<std::array<tinytype, 4>>> per_stage;
        per_stage.reserve(horizon);
        for (int h = 0; h < horizon; ++h) {
            auto spheres = spheres_at_step(step + h);
            tinytype inflate = inflation_rate * tinytype(std::sqrt(static_cast<double>(h)));
            for (auto& s : spheres) {
                s[3] += inflate;
            }
            per_stage.push_back(std::move(spheres));
        }
        return per_stage;
    }
};

struct ScenarioSpec {
    std::string name;
    std::string slug;
    DynamicSpheres obstacles;
    std::vector<Eigen::Vector3d> guide_points;
    tinytype prediction_inflation = tinytype(0.02);
    double activation_on = 0.75;
    double activation_off = 0.95;
    int total_steps = 28;
};

struct PsdCertificate {
    double trace_gap;
    double eta_min;
    double true_dist2_min;
    bool certified;
};

struct PlanCache {
    std::vector<Vec> states;
    std::vector<Vec> inputs;
    int start_step = 0;
    int last_iters = 0;
};

struct ScenarioResult {
    std::string name;
    std::string slug;
    bool success = false;
    int goal_step = -1;
    double min_point_sd = std::numeric_limits<double>::infinity();
    double min_seg_sd = std::numeric_limits<double>::infinity();
    double min_eta = std::numeric_limits<double>::infinity();
    double max_trace_gap = 0.0;
    bool all_certified = true;
    double final_goal_dist = std::numeric_limits<double>::infinity();
    double final_vel_norm = std::numeric_limits<double>::infinity();
    int planner_solves = 0;
    int tracker_solves = 0;
    double planner_total_us = 0.0;
    double tracker_total_us = 0.0;
};

struct RunConfig {
    int replan_stride = N - 1;
    std::string file_prefix = "tinysdp_3d_";
    bool log_plan_states = false;
    bool mocap_mode = false;
    std::vector<std::string> scenario_filter;
};

Vec build_lifted(const Vec& base_state) {
    const int nxL = NX0 + NX0 * NX0;
    Vec lifted(nxL);
    lifted.setZero();
    lifted.topRows(NX0) = base_state;
    Mat outer = base_state * base_state.transpose();
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            lifted(NX0 + j * NX0 + i) = outer(i, j);
        }
    }
    return lifted;
}

double signed_distance_point_spheres(const Vec& x,
                                     const std::vector<std::array<tinytype,4>>& spheres) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& s : spheres) {
        double dx = static_cast<double>(x(0) - s[0]);
        double dy = static_cast<double>(x(1) - s[1]);
        double dz = static_cast<double>(x(2) - s[2]);
        double r  = static_cast<double>(s[3]);
        best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - r);
    }
    return best;
}

double signed_distance_segment_spheres(const Vec& p0,
                                       const Vec& p1,
                                       const std::vector<std::array<tinytype,4>>& spheres) {
    double best = std::numeric_limits<double>::infinity();
    double x0 = static_cast<double>(p0(0));
    double y0 = static_cast<double>(p0(1));
    double z0 = static_cast<double>(p0(2));
    double x1 = static_cast<double>(p1(0));
    double y1 = static_cast<double>(p1(1));
    double z1 = static_cast<double>(p1(2));
    double ddx = x1 - x0;
    double ddy = y1 - y0;
    double ddz = z1 - z0;
    double len2 = ddx * ddx + ddy * ddy + ddz * ddz;
    for (const auto& s : spheres) {
        double cx = static_cast<double>(s[0]);
        double cy = static_cast<double>(s[1]);
        double cz = static_cast<double>(s[2]);
        double r  = static_cast<double>(s[3]);
        double t = 0.0;
        if (len2 > 0.0) {
            t = ((cx - x0) * ddx + (cy - y0) * ddy + (cz - z0) * ddz) / len2;
            t = std::max(0.0, std::min(1.0, t));
        }
        double px = x0 + t * ddx;
        double py = y0 + t * ddy;
        double pz = z0 + t * ddz;
        double sd = std::sqrt((px - cx) * (px - cx) +
                              (py - cy) * (py - cy) +
                              (pz - cz) * (pz - cz)) - r;
        best = std::min(best, sd);
    }
    return best;
}

PsdCertificate compute_psd_certificate(const Vec& x_lifted,
                                       const std::vector<std::array<tinytype,4>>& spheres) {
    PsdCertificate cert;
    cert.trace_gap = 0.0;
    cert.eta_min = std::numeric_limits<double>::infinity();
    cert.true_dist2_min = std::numeric_limits<double>::infinity();
    cert.certified = false;

    Eigen::Vector3d z;
    z << static_cast<double>(x_lifted(0)),
         static_cast<double>(x_lifted(1)),
         static_cast<double>(x_lifted(2));

    Eigen::Matrix<double, NX0, NX0> XX_full;
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            XX_full(i, j) = static_cast<double>(x_lifted(NX0 + j * NX0 + i));
        }
    }

    Eigen::Matrix3d XX = XX_full.topLeftCorner<3, 3>();
    cert.trace_gap = XX.trace() - z.squaredNorm();

    for (const auto& s : spheres) {
        Eigen::Vector3d c;
        c << static_cast<double>(s[0]),
             static_cast<double>(s[1]),
             static_cast<double>(s[2]);
        double r = static_cast<double>(s[3]);
        double lifted_dist2 = XX.trace() - 2.0 * c.dot(z) + c.squaredNorm();
        double eta = lifted_dist2 - r * r;
        double true_dist2 = (z - c).squaredNorm() - r * r;
        cert.eta_min = std::min(cert.eta_min, eta);
        cert.true_dist2_min = std::min(cert.true_dist2_min, true_dist2);
    }

    cert.certified = (cert.eta_min >= 0.0) && (std::abs(cert.trace_gap) <= cert.eta_min);
    return cert;
}

int clamp_index(int idx, int lo, int hi) {
    if (idx < lo) return lo;
    if (idx > hi) return hi;
    return idx;
}

Eigen::Vector3d sample_polyline(const std::vector<Eigen::Vector3d>& route, double alpha) {
    if (route.empty()) return Eigen::Vector3d::Zero();
    if (route.size() == 1) return route.front();
    double s = std::max(0.0, std::min(1.0, alpha));
    double scaled = s * static_cast<double>(route.size() - 1);
    int idx = static_cast<int>(std::floor(scaled));
    if (idx >= static_cast<int>(route.size()) - 1) {
        return route.back();
    }
    double local = scaled - static_cast<double>(idx);
    return (1.0 - local) * route[idx] + local * route[idx + 1];
}

Vec clamp_input(const Vec& u, tinytype limit) {
    Vec out = u;
    for (int i = 0; i < out.rows(); ++i) {
        out(i) = std::max(-limit, std::min(limit, out(i)));
    }
    return out;
}

void rollout_plan(const Mat& Ad,
                  const Mat& Bd,
                  const Vec& x_start,
                  TinySolver* solver,
                  PlanCache* cache) {
    cache->states.assign(N, Vec::Zero(NX0));
    cache->inputs.assign(N - 1, Vec::Zero(NU0));
    Vec x = x_start;
    cache->states[0] = x;
    for (int k = 0; k < N - 1; ++k) {
        Vec u = solver->solution->u.col(k).topRows(NU0);
        cache->inputs[k] = u;
        x = Ad * x + Bd * u;
        cache->states[k + 1] = x;
    }
    cache->last_iters = solver->solution->iter;
}

void set_base_tracking_refs(TinySolver* solver,
                            const PlanCache& cache,
                            int current_step) {
    Mat Xref = Mat::Zero(NX0, N);
    Mat Uref = Mat::Zero(NU0, N - 1);
    if (!cache.states.empty()) {
        const int max_idx = static_cast<int>(cache.states.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx);
            Xref.col(i) = cache.states[idx];
        }
    }
    if (!cache.inputs.empty()) {
        const int max_idx_u = static_cast<int>(cache.inputs.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N - 1; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx_u);
            Uref.col(i) = cache.inputs[idx];
        }
    }
    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);
}

void destroy_solver(TinySolver* solver) {
    if (!solver) return;
    delete solver->solution;
    delete solver->settings;
    delete solver->cache;
    delete solver->work;
    delete solver;
}

std::filesystem::path resolve_output_dir() {
    namespace fs = std::filesystem;
    const char* raw = std::getenv("TINYSDP_OUTPUT_DIR");
    fs::path out = (raw && raw[0]) ? fs::path(raw) : fs::current_path() / "outputs";
    std::error_code ec;
    fs::create_directories(out, ec);
    if (ec) {
        std::cout << "[TinySDP-3D] Could not create output directory " << out
                  << ": " << ec.message() << "\n";
    }
    return out;
}

int getenv_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    try {
        return std::max(1, std::stoi(raw));
    } catch (...) {
        return fallback;
    }
}

bool getenv_flag(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
    if (value == "0" || value == "false" || value == "no" || value == "off") return false;
    return fallback;
}

std::string getenv_string(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    return std::string(raw);
}

std::vector<std::string> getenv_list(const char* name) {
    std::vector<std::string> values;
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return values;
    std::string all(raw);
    std::size_t start = 0;
    while (start < all.size()) {
        std::size_t comma = all.find(',', start);
        std::string token = all.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(token);
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return values;
}

RunConfig load_run_config() {
    RunConfig cfg;
    cfg.replan_stride = getenv_int("TINYSDP_3D_REPLAN_STRIDE", N - 1);
    cfg.file_prefix = getenv_string("TINYSDP_3D_FILE_PREFIX", "tinysdp_3d_");
    cfg.log_plan_states = getenv_flag("TINYSDP_3D_LOG_PLAN_STATES", false);
    cfg.mocap_mode = getenv_flag("TINYSDP_3D_MOCAP_MODE", false);
    cfg.scenario_filter = getenv_list("TINYSDP_3D_SCENARIOS");
    if (cfg.scenario_filter.empty()) {
        cfg.scenario_filter.push_back(TINYSDP_SCENARIO_SLUG);
    }
    return cfg;
}

bool scenario_selected(const RunConfig& cfg, const std::string& slug) {
    if (cfg.scenario_filter.empty()) return true;
    return std::find(cfg.scenario_filter.begin(), cfg.scenario_filter.end(), slug) != cfg.scenario_filter.end();
}

std::vector<ScenarioSpec> build_scenarios() {
    std::vector<ScenarioSpec> scenarios;

    ScenarioSpec sweeper;
    sweeper.name = "Sweeping Barrier";
    sweeper.slug = "sweeping_barrier";
    sweeper.prediction_inflation = tinytype(0.016);
    sweeper.activation_on = 2.0;
    sweeper.activation_off = 2.25;
    sweeper.total_steps = 28;
    sweeper.guide_points = {
        Eigen::Vector3d(-3.0, -1.35, 0.95),
        Eigen::Vector3d(-1.8, -1.75, 1.10),
        Eigen::Vector3d(-0.8, -0.65, 0.55),
    };
    sweeper.obstacles.agents = {
        { tinytype(-3.3), tinytype( 0.0), tinytype(0.20), tinytype(0.00), tinytype( 0.00), tinytype(0.00),
          tinytype(0.65), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0) },
        { tinytype(-1.8), tinytype( 1.30), tinytype(0.35), tinytype(0.03), tinytype(-0.16), tinytype(0.00),
          tinytype(0.56), tinytype(0.03), tinytype(0.30), tinytype(0.2), tinytype(0.05), tinytype(0.30), tinytype(0.4), tinytype(0.04), tinytype(0.40), tinytype(0.5) },
        { tinytype(-1.35), tinytype( 1.55), tinytype(0.95), tinytype(0.03), tinytype(-0.16), tinytype(0.00),
          tinytype(0.55), tinytype(0.03), tinytype(0.30), tinytype(0.8), tinytype(0.05), tinytype(0.30), tinytype(0.7), tinytype(0.04), tinytype(0.40), tinytype(1.0) }
    };
    scenarios.push_back(sweeper);

    ScenarioSpec gate;
    gate.name = "Rising Gate";
    gate.slug = "vertical_gate";
    gate.prediction_inflation = tinytype(0.015);
    gate.activation_on = 1.45;
    gate.activation_off = 1.70;
    gate.total_steps = 26;
    gate.guide_points = {
        Eigen::Vector3d(-3.1,  0.0, 2.45),
        Eigen::Vector3d(-1.6,  0.0, 2.85),
        Eigen::Vector3d(-0.5,  0.0, 1.00),
    };
    gate.obstacles.agents = {
        { tinytype(-2.9), tinytype( 0.90), tinytype(0.15), tinytype(0.00), tinytype(0.00), tinytype(0.00),
          tinytype(0.60), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0) },
        { tinytype(-2.4), tinytype(-0.90), tinytype(0.15), tinytype(0.00), tinytype(0.00), tinytype(0.00),
          tinytype(0.60), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0) },
        { tinytype(-1.35), tinytype(0.00), tinytype(0.35), tinytype(0.00), tinytype(0.00), tinytype(0.00),
          tinytype(0.55), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.75), tinytype(0.48), tinytype(-1.1) }
    };
    scenarios.push_back(gate);

    return scenarios;
}

ScenarioResult run_scenario(const ScenarioSpec& scenario,
                            const RunConfig& cfg,
                            const std::filesystem::path& output_dir,
                            const Mat& Ad,
                            const Mat& Bd,
                            const Mat& A,
                            const Mat& B,
                            const Mat& Q,
                            const Mat& R,
                            const Mat& x_min,
                            const Mat& x_max,
                            const Mat& u_min,
                            const Mat& u_max,
                            const Vec& fdyn,
                            tinytype rho_base,
                            tinytype rho_psd_penalty) {
    ScenarioResult result;
    result.name = scenario.name;
    result.slug = scenario.slug;

    const int nxL = A.rows();
    const int nuL = B.cols();
    const tinytype goal_pos_tol = tinytype(0.15);
    const tinytype goal_vel_tol = tinytype(0.05);
    const tinytype tracker_input_limit = tinytype(3.0);
    const double seg_guard = 0.02;
    const int replan_stride = cfg.replan_stride;
    const int horizon_guard = 1;
    bool psd_constraints_active = !cfg.mocap_mode;
    const int total_steps = cfg.mocap_mode ? (scenario.total_steps + 12) : scenario.total_steps;
    const tinytype terminal_capture_radius = cfg.mocap_mode ? tinytype(3.0) : tinytype(1.0);

    Vec x0(NX0);
    x0 << -4.5, 0.0, 1.0, 0.0, 0.0, 0.0;
    Vec x_track = x0;
    PlanCache plan;

    auto goal_reached = [&](const Vec& state) -> bool {
        return state.topRows(3).norm() < goal_pos_tol &&
               state.bottomRows(3).norm() < goal_vel_tol;
    };

    TinySolver* solver_psd = nullptr;
    if (tiny_setup(&solver_psd, A, B, fdyn, Q, R, rho_base, nxL, nuL, N, /*verbose=*/0)) {
        std::cout << "[TinySDP-3D] Failed to setup planner for " << scenario.slug << "\n";
        return result;
    }
    solver_psd->settings->adaptive_rho = 0;
    solver_psd->settings->max_iter = 1200;
    solver_psd->settings->abs_pri_tol = tinytype(5e-2);
    solver_psd->settings->abs_dua_tol = tinytype(5e-2);
    solver_psd->settings->check_termination = 1;
    tiny_set_bound_constraints(solver_psd, x_min, x_max, u_min, u_max);
    tiny_enable_psd(solver_psd, NX0, NU0, rho_psd_penalty);
    tiny_set_x_ref(solver_psd, Mat::Zero(nxL, N));
    tiny_set_u_ref(solver_psd, Mat::Zero(nuL, N - 1));

    TinySolver* solver_track = nullptr;
    Mat Q_track = Mat::Zero(NX0, NX0);
    Q_track(0,0) = 55.0; Q_track(1,1) = 55.0; Q_track(2,2) = 55.0;
    Q_track(3,3) = 6.0;  Q_track(4,4) = 6.0;  Q_track(5,5) = 6.0;
    Mat R_track = Mat::Zero(NU0, NU0);
    R_track.diagonal().array() = tinytype(0.25);
    Vec fdyn_track = Vec::Zero(NX0);
    Mat x_min_track = Mat::Constant(NX0, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max_track = Mat::Constant(NX0, N,  std::numeric_limits<tinytype>::infinity());
    x_min_track.topRows(NX0).setConstant(-30.0);
    x_max_track.topRows(NX0).setConstant( 30.0);
    Mat u_min_track = Mat::Constant(NU0, N - 1, -3.0);
    Mat u_max_track = Mat::Constant(NU0, N - 1,  3.0);
    if (tiny_setup(&solver_track, Ad, Bd, fdyn_track, Q_track, R_track, rho_base, NX0, NU0, N, /*verbose=*/0)) {
        destroy_solver(solver_psd);
        std::cout << "[TinySDP-3D] Failed to setup tracker for " << scenario.slug << "\n";
        return result;
    }
    solver_track->settings->adaptive_rho = 0;
    solver_track->settings->max_iter = 120;
    solver_track->settings->abs_pri_tol = tinytype(1e-3);
    solver_track->settings->abs_dua_tol = tinytype(1e-3);
    solver_track->settings->check_termination = 1;
    tiny_set_bound_constraints(solver_track, x_min_track, x_max_track, u_min_track, u_max_track);
    tiny_set_x_ref(solver_track, Mat::Zero(NX0, N));
    tiny_set_u_ref(solver_track, Mat::Zero(NU0, N - 1));

    std::ofstream csv_track(output_dir / (cfg.file_prefix + scenario.slug + "_tracking.csv"));
    if (csv_track.is_open()) {
        csv_track << "k,x,y,z,vx,vy,vz,u1,u2,u3,signed_dist,seg_signed_dist,plan_age,solver_iter\n";
    }
    std::ofstream csv_spheres(output_dir / (cfg.file_prefix + scenario.slug + "_spheres.csv"));
    if (csv_spheres.is_open()) {
        csv_spheres << "k,sphere,cx,cy,cz,r\n";
    }
    std::ofstream csv_plan(output_dir / (cfg.file_prefix + scenario.slug + "_plan_log.csv"));
    if (csv_plan.is_open()) {
        csv_plan << "replan_step,plan_type,iter,num_spheres,min_sd_seed,threshold_on,threshold_off,goal_dist,certified_future,status\n";
    }
    std::ofstream csv_cert(output_dir / (cfg.file_prefix + scenario.slug + "_certificate.csv"));
    if (csv_cert.is_open()) {
        csv_cert << "k,trace_gap,eta_min,true_dist2_min,certified\n";
    }
    std::ofstream csv_plan_states;
    if (cfg.log_plan_states) {
        csv_plan_states.open(output_dir / (cfg.file_prefix + scenario.slug + "_plan_states.csv"));
        if (csv_plan_states.is_open()) {
            csv_plan_states << "replan_step,h,x,y,z,vx,vy,vz\n";
        }
    }

    auto log_spheres = [&](int step, const std::vector<std::array<tinytype,4>>& spheres_now) {
        if (!csv_spheres.is_open()) return;
        for (std::size_t j = 0; j < spheres_now.size(); ++j) {
            csv_spheres << step << "," << j
                       << "," << spheres_now[j][0]
                       << "," << spheres_now[j][1]
                       << "," << spheres_now[j][2]
                       << "," << spheres_now[j][3] << "\n";
        }
    };

    auto log_tracking_row = [&](int step, const Vec& state, const Vec& input,
                                double sd_point, double sd_segment,
                                int plan_age, int iters) {
        if (!csv_track.is_open()) return;
        csv_track << step
                  << "," << state(0) << "," << state(1) << "," << state(2)
                  << "," << state(3) << "," << state(4) << "," << state(5)
                  << "," << input(0) << "," << input(1) << "," << input(2)
                  << "," << sd_point << "," << sd_segment
                  << "," << plan_age << "," << iters << "\n";
    };

    auto log_certificate = [&](int step,
                               const Vec& x_lifted,
                               const std::vector<std::array<tinytype,4>>& spheres_now) {
        PsdCertificate cert = compute_psd_certificate(x_lifted, spheres_now);
        result.min_eta = std::min(result.min_eta, cert.eta_min);
        result.max_trace_gap = std::max(result.max_trace_gap, std::abs(cert.trace_gap));
        result.all_certified = result.all_certified && cert.certified;
        if (!csv_cert.is_open()) return;
        csv_cert << step << "," << cert.trace_gap << "," << cert.eta_min << ","
                 << cert.true_dist2_min << "," << (cert.certified ? 1 : 0) << "\n";
    };

    auto choose_safe_input = [&](const Vec& current_state,
                                 const std::vector<Vec>& candidates,
                                 const std::vector<std::array<tinytype,4>>& next_spheres,
                                 const Vec& target_state,
                                 Vec* u_safe) -> bool {
        static const std::array<double, 7> scales = {1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15};
        bool found = false;
        double best_score = std::numeric_limits<double>::infinity();
        Vec best_u = Vec::Zero(NU0);

        for (const Vec& u_nominal : candidates) {
            for (double scale : scales) {
                Vec u_try = clamp_input(tinytype(scale) * u_nominal, tracker_input_limit);
                Vec x1_try = Ad * current_state + Bd * u_try;
                PsdCertificate cert_try = compute_psd_certificate(build_lifted(x1_try), next_spheres);
                double seg_try = signed_distance_segment_spheres(current_state, x1_try, next_spheres);
                if (!cert_try.certified || seg_try < seg_guard) {
                    continue;
                }

                double goal_score = x1_try.topRows(3).norm();
                double plan_score = (x1_try - target_state).norm();
                double effort_score = u_try.norm();
                double score = goal_score + 0.55 * plan_score + 0.05 * effort_score;
                if (!found || score < best_score) {
                    found = true;
                    best_score = score;
                    best_u = u_try;
                }
            }
        }

        if (!found) {
            return false;
        }
        *u_safe = best_u;
        return true;
    };

    auto replan_psd = [&](int step, const Vec& x_seed) {
        auto spheres_now_all = scenario.obstacles.spheres_at_step(step);
        std::vector<std::array<tinytype, 4>> spheres_now_static;
        std::vector<std::array<tinytype, 4>> spheres_now_dynamic;
        spheres_now_static.reserve(spheres_now_all.size());
        spheres_now_dynamic.reserve(spheres_now_all.size());
        for (std::size_t i = 0; i < spheres_now_all.size(); ++i) {
            if (i < scenario.obstacles.agents.size() && sphere_has_motion(scenario.obstacles.agents[i])) {
                spheres_now_dynamic.push_back(spheres_now_all[i]);
            } else {
                spheres_now_static.push_back(spheres_now_all[i]);
            }
        }

        double sd_seed = signed_distance_point_spheres(x_seed, spheres_now_all);
        double sd_seed_dynamic = spheres_now_dynamic.empty()
            ? std::numeric_limits<double>::infinity()
            : signed_distance_point_spheres(x_seed, spheres_now_dynamic);
        double goal_dist = x_seed.topRows(3).norm();

        double on_thresh = scenario.activation_on;
        double off_thresh = scenario.activation_off;
        if (cfg.mocap_mode) {
            on_thresh = std::min(on_thresh, goal_dist + 0.30);
            off_thresh = std::max(on_thresh + 0.20, std::min(off_thresh, goal_dist + 0.70));
            if (!psd_constraints_active && sd_seed_dynamic < on_thresh) {
                psd_constraints_active = true;
            } else if (psd_constraints_active && sd_seed_dynamic > off_thresh) {
                psd_constraints_active = false;
            }
        }

        std::vector<std::array<tinytype, 4>> planner_spheres_now = spheres_now_static;
        if (!cfg.mocap_mode || psd_constraints_active) {
            planner_spheres_now.insert(planner_spheres_now.end(),
                                       spheres_now_dynamic.begin(),
                                       spheres_now_dynamic.end());
        }

        std::vector<std::vector<std::array<tinytype, 4>>> predicted_true;
        std::vector<std::vector<std::array<tinytype, 4>>> predicted_inflated;
        predicted_true.reserve(N);
        predicted_inflated.reserve(N);
        if (cfg.mocap_mode) {
            for (int h = 0; h < N; ++h) {
                predicted_true.push_back(planner_spheres_now);
                predicted_inflated.push_back(planner_spheres_now);
            }
        } else if (psd_constraints_active) {
            if (cfg.mocap_mode) {
                for (int h = 0; h < N; ++h) {
                    predicted_true.push_back(planner_spheres_now);
                    predicted_inflated.push_back(planner_spheres_now);
                }
            } else {
                predicted_true = scenario.obstacles.horizon_spheres_per_stage(step, N, tinytype(0.0));
                predicted_inflated =
                    scenario.obstacles.horizon_spheres_per_stage(step, N, scenario.prediction_inflation);
            }
        } else {
            for (int h = 0; h < N; ++h) {
                predicted_true.emplace_back();
                predicted_inflated.emplace_back();
            }
        }

        Mat Xref_plan = Mat::Zero(nxL, N);
        Mat Uref_plan = Mat::Zero(nuL, N - 1);
        std::vector<Eigen::Vector3d> route;
        route.reserve(scenario.guide_points.size() + 2);
        route.push_back(Eigen::Vector3d(
            static_cast<double>(x_seed(0)),
            static_cast<double>(x_seed(1)),
            static_cast<double>(x_seed(2))));
        if (!cfg.mocap_mode || psd_constraints_active) {
            for (const auto& p : scenario.guide_points) {
                if (p.x() > static_cast<double>(x_seed(0)) + 0.05) {
                    route.push_back(p);
                }
            }
        }
        route.push_back(Eigen::Vector3d::Zero());
        for (int h = 0; h < N; ++h) {
            double alpha = (N > 1) ? static_cast<double>(h) / static_cast<double>(N - 1) : 1.0;
            Eigen::Vector3d pref = sample_polyline(route, alpha);
            Xref_plan(0, h) = tinytype(pref.x());
            Xref_plan(1, h) = tinytype(pref.y());
            Xref_plan(2, h) = tinytype(pref.z());
            Xref_plan(3, h) = tinytype(0.0);
            Xref_plan(4, h) = tinytype(0.0);
            Xref_plan(5, h) = tinytype(0.0);
        }
        tiny_set_x_ref(solver_psd, Xref_plan);
        tiny_set_u_ref(solver_psd, Uref_plan);
        solver_psd->settings->en_psd = planner_spheres_now.empty() ? 0 : 1;
        if (!planner_spheres_now.empty()) {
            if (cfg.mocap_mode) {
                tiny_set_lifted_spheres(solver_psd, planner_spheres_now);
            } else {
                tiny_set_lifted_spheres_per_stage(solver_psd, predicted_inflated);
            }
        }
        tiny_set_x0(solver_psd, build_lifted(x_seed));

        auto t0 = std::chrono::high_resolution_clock::now();
        tiny_solve(solver_psd);
        auto t1 = std::chrono::high_resolution_clock::now();
        result.planner_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        result.planner_solves++;

        bool plan_valid = true;
        int certified_future = 0;
        int first_fail = -1;
        for (int h = 1; h < N - 1; ++h) {
            PsdCertificate cert_h = compute_psd_certificate(solver_psd->solution->x.col(h), predicted_true[h]);
            if (cert_h.certified) {
                certified_future++;
            } else {
                plan_valid = false;
                if (first_fail < 0) first_fail = h;
                break;
            }
        }

        if (plan_valid) {
            rollout_plan(Ad, Bd, x_seed, solver_psd, &plan);
            plan.start_step = step;
            if (csv_plan_states.is_open()) {
                for (int h = 0; h < static_cast<int>(plan.states.size()); ++h) {
                    const Vec& xh = plan.states[h];
                    csv_plan_states << step << "," << h
                                    << "," << xh(0) << "," << xh(1) << "," << xh(2)
                                    << "," << xh(3) << "," << xh(4) << "," << xh(5) << "\n";
                }
            }
        } else {
            std::cout << "[TinySDP-3D][" << scenario.slug << "] Planner reject at k=" << step
                      << " first_fail_stage=" << first_fail << "\n";
        }

        if (csv_plan.is_open()) {
            csv_plan << step << "," << (cfg.mocap_mode
                                            ? (psd_constraints_active ? "tinysdp_current" : "static_only")
                                            : "tinysdp_horizon")
                     << "," << solver_psd->solution->iter
                     << "," << planner_spheres_now.size()
                     << "," << sd_seed
                     << "," << on_thresh
                     << "," << off_thresh
                     << "," << goal_dist
                     << "," << certified_future
                     << "," << (plan_valid ? "accepted" : "rejected") << "\n";
        }
    };

    auto spheres0 = scenario.obstacles.spheres_at_step(0);
    log_spheres(0, spheres0);
    double sd0 = signed_distance_point_spheres(x_track, spheres0);
    Vec zero_u = Vec::Zero(NU0);
    log_tracking_row(0, x_track, zero_u, sd0, sd0, 0, 0);
    log_certificate(0, build_lifted(x_track), spheres0);
    result.min_point_sd = sd0;
    result.min_seg_sd = sd0;

    replan_psd(0, x_track);

    Vec prev_state = x_track;
    for (int k = 0; k < total_steps; ++k) {
        bool terminal_capture = x_track.topRows(3).norm() < terminal_capture_radius;
        bool need_replan = !terminal_capture &&
                           ((k == 0) ||
                            (k - plan.start_step >= replan_stride) ||
                            (k >= plan.start_step + N - horizon_guard));
        if (need_replan && k > 0) {
            replan_psd(k, x_track);
        }

        auto next_spheres = scenario.obstacles.spheres_at_step(k + 1);
        Vec u0 = Vec::Zero(NU0);
        int applied_iters = 0;

        if (terminal_capture) {
            Vec u_fb(NU0);
            u_fb = -tinytype(0.9) * x_track.topRows(3) - tinytype(1.6) * x_track.bottomRows(3);
            u_fb = clamp_input(u_fb, tracker_input_limit);
            if (!choose_safe_input(x_track, {u_fb}, next_spheres, Vec::Zero(NX0), &u0)) {
                u0.setZero();
            }
        } else if (!plan.inputs.empty()) {
            int offset = clamp_index(k - plan.start_step, 0, static_cast<int>(plan.inputs.size()) - 1);
            Vec plan_target = plan.states[clamp_index(offset + 1, 0, static_cast<int>(plan.states.size()) - 1)];
            Vec u_plan = plan.inputs[offset];
            set_base_tracking_refs(solver_track, plan, k);
            tiny_set_x0(solver_track, x_track);
            auto t0 = std::chrono::high_resolution_clock::now();
            tiny_solve(solver_track);
            auto t1 = std::chrono::high_resolution_clock::now();
            result.tracker_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.tracker_solves++;
            applied_iters = solver_track->solution->iter;

            Vec u_nominal = solver_track->solution->u.col(0);
            Vec u_fb(NU0);
            u_fb = -tinytype(0.9) * x_track.topRows(3) - tinytype(1.6) * x_track.bottomRows(3);
            u_fb = clamp_input(u_fb, tracker_input_limit);

            std::vector<Vec> candidates;
            candidates.push_back(u_nominal);
            candidates.push_back(u_plan);
            candidates.push_back(tinytype(0.5) * (u_nominal + u_plan));
            candidates.push_back(u_fb);

            if (!choose_safe_input(x_track, candidates, next_spheres, plan_target, &u0)) {
                Vec u_brake(NU0);
                u_brake = -tinytype(1.6) * x_track.bottomRows(3);
                u_brake = clamp_input(u_brake, tracker_input_limit);
                if (!choose_safe_input(x_track, {u_brake}, next_spheres, plan_target, &u0)) {
                    u0.setZero();
                }
            }
        } else {
            Vec u_brake(NU0);
            u_brake = -tinytype(1.6) * x_track.bottomRows(3);
            u_brake = clamp_input(u_brake, tracker_input_limit);
            if (!choose_safe_input(x_track, {u_brake}, next_spheres, Vec::Zero(NX0), &u0)) {
                u0.setZero();
            }
        }

        prev_state = x_track;
        x_track = Ad * x_track + Bd * u0;

        int step_idx = k + 1;
        auto spheres_now = scenario.obstacles.spheres_at_step(step_idx);
        log_spheres(step_idx, spheres_now);
        double sd_point = signed_distance_point_spheres(x_track, spheres_now);
        double sd_segment = signed_distance_segment_spheres(prev_state, x_track, spheres_now);
        result.min_point_sd = std::min(result.min_point_sd, sd_point);
        result.min_seg_sd = std::min(result.min_seg_sd, sd_segment);

        int plan_age = step_idx - plan.start_step;
        log_tracking_row(step_idx, x_track, u0, sd_point, sd_segment, plan_age, applied_iters);
        log_certificate(step_idx, build_lifted(x_track), spheres_now);

        if (goal_reached(x_track)) {
            result.success = true;
            result.goal_step = step_idx;
            break;
        }
    }

    result.final_goal_dist = x_track.topRows(3).norm();
    result.final_vel_norm = x_track.bottomRows(3).norm();

    destroy_solver(solver_psd);
    destroy_solver(solver_track);
    return result;
}

}  // namespace

int main() {
    const std::filesystem::path output_dir = resolve_output_dir();
    const RunConfig cfg = load_run_config();

    Mat Ad(NX0, NX0);
    Ad << 1, 0, 0, 1, 0, 0,
          0, 1, 0, 0, 1, 0,
          0, 0, 1, 0, 0, 1,
          0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 1;
    Mat Bd(NX0, NU0);
    Bd << 0.5, 0,   0,
          0,   0.5, 0,
          0,   0,   0.5,
          1,   0,   0,
          0,   1,   0,
          0,   0,   1;

    Mat A, B;
    tiny_build_lifted_from_base(Ad, Bd, A, B);
    const int nxL = A.rows();
    const int nuL = B.cols();

    Mat Q = Mat::Zero(nxL, nxL);
    Q(0,0) = 40.0; Q(1,1) = 40.0; Q(2,2) = 40.0;
    Q(3,3) = 4.0;  Q(4,4) = 4.0;  Q(5,5) = 4.0;
    Q.diagonal().segment(NX0, NX0 * NX0).array() = tinytype(1e-4);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int nuu = NU0 * NU0;
    R.diagonal().head(NU0).array() = tinytype(0.2);
    R.diagonal().segment(NU0, nxu).array() = tinytype(1.0);
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(1.0);
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(10.0);

    Vec fdyn = Vec::Zero(nxL);
    const tinytype rho_base = tinytype(5.0);
    const tinytype rho_psd_penalty = tinytype(0.95);

    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    x_min.middleRows(NX0, NX0 * NX0).setConstant(-1500.0);
    x_max.middleRows(NX0, NX0 * NX0).setConstant( 1500.0);

    Mat u_min = Mat::Constant(nuL, N - 1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N - 1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    u_min.bottomRows(nxu + nux + nuu).setConstant(-120.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 120.0);

    auto scenarios = build_scenarios();
    std::ofstream csv_summary(output_dir / (cfg.file_prefix + "summary.csv"));
    if (csv_summary.is_open()) {
        csv_summary << "scenario,slug,success,goal_step,min_point_sd,min_seg_sd,min_eta,max_trace_gap,"
                       "all_certified,final_goal_dist,final_vel_norm,planner_solves,planner_total_ms,"
                       "tracker_solves,tracker_total_ms,tracker_avg_us\n";
    }

    int selected_count = 0;
    for (const auto& scenario : scenarios) {
        if (scenario_selected(cfg, scenario.slug)) {
            selected_count++;
        }
    }
    std::cout << "[TinySDP-3D] Running " << selected_count << " dynamic 3D scenario(s)\n";
    int success_count = 0;
    for (const auto& scenario : scenarios) {
        if (!scenario_selected(cfg, scenario.slug)) {
            continue;
        }
        ScenarioResult result = run_scenario(
            scenario, cfg, output_dir, Ad, Bd, A, B, Q, R, x_min, x_max, u_min, u_max, fdyn, rho_base, rho_psd_penalty);
        success_count += result.success ? 1 : 0;
        if (csv_summary.is_open()) {
            csv_summary << result.name << "," << result.slug << ","
                        << (result.success ? 1 : 0) << ","
                        << result.goal_step << ","
                        << result.min_point_sd << ","
                        << result.min_seg_sd << ","
                        << result.min_eta << ","
                        << result.max_trace_gap << ","
                        << (result.all_certified ? 1 : 0) << ","
                        << result.final_goal_dist << ","
                        << result.final_vel_norm << ","
                        << result.planner_solves << ","
                        << result.planner_total_us / 1000.0 << ","
                        << result.tracker_solves << ","
                        << result.tracker_total_us / 1000.0 << ","
                        << (result.tracker_solves > 0 ? result.tracker_total_us / result.tracker_solves : 0.0)
                        << "\n";
        }

        std::cout << "[TinySDP-3D] " << result.slug
                  << " success=" << (result.success ? "YES" : "NO")
                  << " goal_step=" << result.goal_step
                  << " min_seg=" << result.min_seg_sd
                  << " min_eta=" << result.min_eta
                  << " planner_ms=" << result.planner_total_us / 1000.0
                  << " tracker_avg_us=" << (result.tracker_solves > 0
                      ? result.tracker_total_us / result.tracker_solves : 0.0)
                  << "\n";
    }

    std::cout << "[TinySDP-3D] Completed " << success_count << "/" << selected_count
              << " scenarios successfully\n";
    return (success_count == selected_count) ? 0 : 1;
}
