#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

namespace {

// Start position presets
enum class StartPreset { INSIDE, EDGE_UP, EDGE_DOWN, OUTSIDE_CENTER, ABOVE, BELOW };

struct StartConfig {
    double x;
    double y;
    double rho_psd;
    const char* name;
};

StartConfig get_start_config(StartPreset preset) {
    switch (preset) {
        case StartPreset::EDGE_UP:
            return {6.0, 1.0, 0.96, "edge_up"};
        case StartPreset::EDGE_DOWN:
            return {6.0, -1.0, 0.96, "edge_down"};
        case StartPreset::OUTSIDE_CENTER:
            return {6.0, 0.05, 1.2, "outside_center"};
        case StartPreset::ABOVE:
            return {4.0, 3.0, 1.5, "above"};
        case StartPreset::BELOW:
            return {4.0, -3.0, 0.96, "below"};
        case StartPreset::INSIDE:
        default:
            return {5.0, 0.05, 2.09, "inside"};
    }
}

StartPreset parse_start_arg(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            std::string val = argv[i + 1];
            if (val == "edge_up") return StartPreset::EDGE_UP;
            if (val == "edge_down") return StartPreset::EDGE_DOWN;
            if (val == "inside") return StartPreset::INSIDE;
            if (val == "outside_center") return StartPreset::OUTSIDE_CENTER;
            if (val == "above") return StartPreset::ABOVE;
            if (val == "below") return StartPreset::BELOW;
        }
    }
    return StartPreset::INSIDE;  // default
}

constexpr int NX0 = 4;
constexpr int NU0 = 2;
constexpr int N   = 45;

using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

Vec build_lifted(const Vec& base_state) {
    const int nxL = NX0 + NX0 * NX0;
    Vec lifted(nxL);
    lifted.setZero();
    lifted.topRows(NX0) = base_state;
    Mat outer = base_state * base_state.transpose();
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            lifted(NX0 + j*NX0 + i) = outer(i, j);
        }
    }
    return lifted;
}

std::pair<Mat, Mat> build_diag_refs(TinySolver* solver) {
    const int nxL = solver->work->nx;
    const int nuL = solver->work->nu;
    Mat Xref = Mat::Zero(nxL, N);
    Mat Uref = Mat::Zero(nuL, N-1);
    const tinytype q_xx = tinytype(1.0);
    const tinytype r_uu = tinytype(10.0);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int baseUU = NU0 + nxu + nux;

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX0; ++i) {
            int idx = NX0 + i*NX0 + i;
            tinytype denom = solver->work->Q(idx);
            if (denom != tinytype(0)) {
                Xref(idx, k) = -q_xx / denom;
            }
        }
    }
    for (int k = 0; k < N-1; ++k) {
        for (int j = 0; j < NU0; ++j) {
            int idx = baseUU + j*NU0 + j;
            tinytype denom = solver->work->R(idx);
            if (denom != tinytype(0)) {
                Uref(idx, k) = -r_uu / denom;
            }
        }
    }
    return {Xref, Uref};
}

double signed_distance_point_disks(const Vec& x,
                                   const std::vector<std::array<tinytype,3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& d : disks) {
        double dx = static_cast<double>(x(0) - d[0]);
        double dy = static_cast<double>(x(1) - d[1]);
        double r  = static_cast<double>(d[2]);
        double sd = std::sqrt(dx*dx + dy*dy) - r;
        if (sd < best) {
            best = sd;
        }
    }
    return best;
}

double signed_distance_segment_disks(const Vec& p0,
                                     const Vec& p1,
                                     const std::vector<std::array<tinytype,3>>& disks) {
    double best = std::numeric_limits<double>::infinity();
    double x0 = static_cast<double>(p0(0));
    double y0 = static_cast<double>(p0(1));
    double x1 = static_cast<double>(p1(0));
    double y1 = static_cast<double>(p1(1));
    double dx = x1 - x0;
    double dy = y1 - y0;
    double len2 = dx*dx + dy*dy;
    for (const auto& d : disks) {
        double cx = static_cast<double>(d[0]);
        double cy = static_cast<double>(d[1]);
        double r  = static_cast<double>(d[2]);
        double t = 0.0;
        if (len2 > 0.0) {
            t = ((cx - x0)*dx + (cy - y0)*dy) / len2;
            t = std::max(0.0, std::min(1.0, t));
        }
        double px = x0 + t*dx;
        double py = y0 + t*dy;
        double sd = std::sqrt((px - cx)*(px - cx) + (py - cy)*(py - cy)) - r;
        if (sd < best) {
            best = sd;
        }
    }
    return best;
}

// Check if the straight-line path from current position to goal is blocked by any disk
// Returns the minimum signed distance along the path (negative = blocked)
double path_to_goal_clearance(const Vec& x,
                              const std::vector<std::array<tinytype,3>>& disks) {
    double x0 = static_cast<double>(x(0));
    double y0 = static_cast<double>(x(1));
    double x1 = 0.0, y1 = 0.0;  // goal at origin
    double dx = x1 - x0;
    double dy = y1 - y0;
    double len2 = dx*dx + dy*dy;
    
    double min_clearance = std::numeric_limits<double>::infinity();
    for (const auto& d : disks) {
        double cx = static_cast<double>(d[0]);
        double cy = static_cast<double>(d[1]);
        double r  = static_cast<double>(d[2]);
        double t = 0.0;
        if (len2 > 0.0) {
            t = ((cx - x0)*dx + (cy - y0)*dy) / len2;
            t = std::max(0.0, std::min(1.0, t));
        }
        double px = x0 + t*dx;
        double py = y0 + t*dy;
        double sd = std::sqrt((px - cx)*(px - cx) + (py - cy)*(py - cy)) - r;
        if (sd < min_clearance) {
            min_clearance = sd;
        }
    }
    return min_clearance;
}

// Rank-1 gap certificate computation
struct PsdCertificate {
    double trace_gap;      // Δ = trace(XX) - ||z||^2
    double eta_min;        // min lifted margin over all disks
    double true_dist2_min; // min true squared distance to disk boundary
    bool certified;        // true if |trace_gap| <= eta_min and eta_min >= 0
};

PsdCertificate compute_psd_certificate(const Vec& x_lifted,
                                       const std::vector<std::array<tinytype,3>>& disks) {
    PsdCertificate cert;
    cert.trace_gap = 0.0;
    cert.eta_min = std::numeric_limits<double>::infinity();
    cert.true_dist2_min = std::numeric_limits<double>::infinity();
    cert.certified = false;

    // Extract position z = (x1, x2)
    Eigen::Vector2d z;
    z << static_cast<double>(x_lifted(0)), static_cast<double>(x_lifted(1));

    // Extract full XX matrix from lifted state
    Eigen::Matrix<double, NX0, NX0> XX_full;
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            XX_full(i, j) = static_cast<double>(x_lifted(NX0 + j * NX0 + i));
        }
    }

    // Extract top-left 2x2 block (position covariance)
    Eigen::Matrix2d XX = XX_full.topLeftCorner<2, 2>();

    // Trace gap: Δ = trace(XX_{2x2}) - ||z||^2
    cert.trace_gap = XX.trace() - z.squaredNorm();

    // Compute lifted margin for each disk
    for (const auto& d : disks) {
        Eigen::Vector2d c;
        c << static_cast<double>(d[0]), static_cast<double>(d[1]);
        double r = static_cast<double>(d[2]);

        // Lifted distance squared: trace(XX) - 2 c^T z + ||c||^2
        double lifted_dist2 = XX.trace() - 2.0 * c.dot(z) + c.squaredNorm();

        // Lifted margin: η = lifted_dist2 - r^2
        double eta = lifted_dist2 - r * r;

        // True distance squared
        double true_dist2 = (z - c).squaredNorm() - r * r;

        if (eta < cert.eta_min) {
            cert.eta_min = eta;
        }
        if (true_dist2 < cert.true_dist2_min) {
            cert.true_dist2_min = true_dist2;
        }
    }

    cert.certified = (cert.eta_min >= 0.0) && (std::abs(cert.trace_gap) <= cert.eta_min);

    return cert;
}

int clamp_index(int idx, int lo, int hi) {
    if (idx < lo) return lo;
    if (idx > hi) return hi;
    return idx;
}

struct PlanCache {
    std::vector<Vec> states;
    std::vector<Vec> inputs;
    int start_step = 0;
    int last_iters = 0;
};

void rollout_plan(const Mat& Ad,
                  const Mat& Bd,
                  const Vec& x_start,
                  TinySolver* solver,
                  PlanCache* cache) {
    if (!cache) return;
    cache->states.assign(N, Vec::Zero(NX0));
    cache->inputs.assign(N-1, Vec::Zero(NU0));
    Vec x = x_start;
    cache->states[0] = x;
    for (int k = 0; k < N-1; ++k) {
        Vec u = solver->solution->u.col(k).topRows(NU0);
        cache->inputs[k] = u;
        x = Ad * x + Bd * u;
        cache->states[k+1] = x;
    }
    cache->last_iters = solver->solution->iter;
}

void set_tracking_refs(TinySolver* solver,
                       const PlanCache& cache,
                       int current_step,
                       const Mat& stab_Xref,
                       const Mat& stab_Uref) {
    Mat Xref = stab_Xref;
    Mat Uref = stab_Uref;
    if (!cache.states.empty()) {
        const int max_idx = static_cast<int>(cache.states.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx);
            Xref.col(i).topRows(NX0) = cache.states[idx];
        }
    }
    if (!cache.inputs.empty()) {
        const int max_idx_u = static_cast<int>(cache.inputs.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N-1; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx_u);
            Uref.col(i).topRows(NU0) = cache.inputs[idx];
        }
    }
    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);
}

}  // namespace

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    const char* raw_output_dir = std::getenv("TINYSDP_OUTPUT_DIR");
    fs::path output_dir = (raw_output_dir && raw_output_dir[0])
        ? fs::path(raw_output_dir)
        : fs::current_path() / "outputs";
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::cout << "[TinySDP-U] Could not create output directory " << output_dir
                  << ": " << ec.message() << "\n";
    }

    StartPreset preset = parse_start_arg(argc, argv);
    StartConfig cfg = get_start_config(preset);
    std::cout << "[TinySDP-U] Start preset: " << cfg.name
              << " x0=(" << cfg.x << ", " << cfg.y << ")"
              << " rho_psd=" << cfg.rho_psd << "\n";

    Mat Ad(NX0, NX0);
    Ad << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;
    Mat Bd(NX0, NU0);
    Bd << 0.5, 0,
          0,   0.5,
          1,   0,
          0,   1;

    Mat A, B;
    tiny_build_lifted_from_base(Ad, Bd, A, B);
    const int nxL = A.rows();
    const int nuL = B.cols();

    Mat Q = Mat::Zero(nxL, nxL);
    Q(0,0) = 8.0; Q(1,1) = 8.0; Q(2,2) = 0.8; Q(3,3) = 0.8;
    Q.diagonal().segment(NX0, NX0*NX0).array() = tinytype(5e-3);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int nuu = NU0 * NU0;
    R.diagonal().head(NU0).array() = tinytype(3.0);
    R.diagonal().segment(NU0, nxu).array() = tinytype(6.0);
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(6.0);
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(250.0);

    Vec fdyn = Vec::Zero(nxL);
    const tinytype rho_base = tinytype(5.0);
    const tinytype rho_psd_penalty = tinytype(cfg.rho_psd);

    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    x_min.middleRows(NX0, NX0*NX0).setConstant(-1500.0);
    x_max.middleRows(NX0, NX0*NX0).setConstant( 1500.0);

    Mat u_min = Mat::Constant(nuL, N-1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N-1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    u_min.bottomRows(nxu + nux + nuu).setConstant(-120.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 120.0);

    Vec x0(NX0);
    x0 << cfg.x, cfg.y, 0.0, 0.0;  // start from preset config

    // U-shaped obstacle from disks (two arms + base)
    const tinytype r_wall = tinytype(0.8);
    std::vector<std::array<tinytype,3>> disks = {
        { tinytype(2.5), tinytype( 0.0), r_wall },
        { tinytype(2.5), tinytype( 1.2), r_wall },
        { tinytype(2.5), tinytype(-1.2), r_wall },
        { tinytype(3.8), tinytype( 1.2), r_wall },
        { tinytype(3.8), tinytype(-1.2), r_wall },
        { tinytype(5.0), tinytype( 1.2), r_wall },
        { tinytype(5.0), tinytype(-1.2), r_wall }
    };

    // PSD planner solver
    TinySolver* solver_psd = nullptr;
    if (tiny_setup(&solver_psd, A, B, fdyn, Q, R, rho_base, nxL, nuL, N, /*verbose=*/0)) {
        return 1;
    }
    solver_psd->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_psd, x_min, x_max, u_min, u_max);
    tiny_enable_psd(solver_psd, NX0, NU0, rho_psd_penalty);
    tiny_set_lifted_disks(solver_psd, disks);
    auto diag_refs_psd = build_diag_refs(solver_psd);
    tiny_set_x_ref(solver_psd, diag_refs_psd.first);
    tiny_set_u_ref(solver_psd, diag_refs_psd.second);

    // Tracking solver (simple LQR, no PSD - same as dynamic demo)
    TinySolver* solver_track = nullptr;
    if (tiny_setup(&solver_track, A, B, fdyn, Q, R, rho_base, nxL, nuL, N, /*verbose=*/0)) {
        return 1;
    }
    solver_track->settings->adaptive_rho = 0;
    tiny_set_bound_constraints(solver_track, x_min, x_max, u_min, u_max);
    auto diag_refs_track = build_diag_refs(solver_track);

    PlanCache plan;
    Vec x_track = x0;

    const int total_steps = 60;
    const int replan_stride = 5;
    const int horizon_guard = 5;
    
    // Distance-based PSD activation with hysteresis + goal proximity exception
    const double base_on = 1.0;   // Base threshold to turn PSD ON
    const double base_off = 1.0;  // Base threshold to turn PSD OFF (hysteresis)
    bool psd_constraints_active = false;

    // Output files include preset name for easy comparison
    std::string suffix = std::string("_") + cfg.name;
    std::ofstream csv_plan(output_dir / ("tinysdp_ushape_plan_log" + suffix + ".csv"));
    if (csv_plan.is_open()) {
        csv_plan << "replan_step,iter,min_sd_seed,mode,threshold,goal_dist,status\n";
    }
    std::ofstream csv_track(output_dir / ("tinysdp_ushape_tracking" + suffix + ".csv"));
    if (csv_track.is_open()) {
        csv_track << "k,x1,x2,x3,x4,u1,u2,signed_dist,seg_signed_dist,plan_age,solver_iter\n";
    }
    std::ofstream csv_cert(output_dir / ("tinysdp_ushape_certificate" + suffix + ".csv"));
    if (csv_cert.is_open()) {
        csv_cert << "k,trace_gap,eta_min,true_dist2_min,certified\n";
    }

    const tinytype goal_pos_tol = tinytype(0.15);
    const tinytype goal_vel_tol = tinytype(0.05);

    // Certificate tracking variables
    double min_eta = std::numeric_limits<double>::infinity();
    double max_trace_gap = 0.0;
    bool all_certified = true;

    auto log_tracking_row = [&](int step, const Vec& state, const Vec& input,
                                double sd_point, double sd_segment,
                                int plan_age, int iters) {
        if (!csv_track.is_open()) return;
        csv_track << step << "," << state(0) << "," << state(1) << "," << state(2) << "," << state(3)
                  << "," << input(0) << "," << input(1) << "," << sd_point
                  << "," << sd_segment << "," << plan_age << "," << iters << "\n";
    };

    auto log_certificate = [&](int step, const Vec& x_lifted) {
        PsdCertificate cert = compute_psd_certificate(x_lifted, disks);

        if (cert.eta_min < min_eta) {
            min_eta = cert.eta_min;
        }
        if (std::abs(cert.trace_gap) > max_trace_gap) {
            max_trace_gap = std::abs(cert.trace_gap);
        }
        if (!cert.certified) {
            all_certified = false;
        }

        if (!csv_cert.is_open()) return;
        csv_cert << step << "," << cert.trace_gap << "," << cert.eta_min << ","
                 << cert.true_dist2_min << "," << (cert.certified ? 1 : 0) << "\n";
    };

    auto goal_reached = [&](const Vec& state) -> bool {
        tinytype pos_norm = state.topRows(2).norm();
        tinytype vel_norm = state.bottomRows(2).norm();
        return (pos_norm < goal_pos_tol) && (vel_norm < goal_vel_tol);
    };

    auto replan_psd = [&](int step, const Vec& x_seed) {
        double sd_seed = signed_distance_point_disks(x_seed, disks);
        double goal_dist = std::sqrt(static_cast<double>(x_seed(0)*x_seed(0) + x_seed(1)*x_seed(1)));
        
        // Goal proximity exception: shrink thresholds as we approach goal
        double on_thresh = std::min(base_on, goal_dist + 0.3);
        double off_thresh = std::max(on_thresh + 0.3, std::min(base_off, goal_dist + 0.8));
        
        // Hysteresis: different thresholds for ON vs OFF
        if (!psd_constraints_active && sd_seed < on_thresh) {
            psd_constraints_active = true;
        } else if (psd_constraints_active && sd_seed > off_thresh) {
            psd_constraints_active = false;
        }
        
        if (psd_constraints_active) {
            solver_psd->settings->en_psd = 1;
            tiny_set_lifted_disks(solver_psd, disks);
        } else {
            solver_psd->settings->en_psd = 0;
        }
        
        Vec x_lift = build_lifted(x_seed);
        tiny_set_x0(solver_psd, x_lift);
        tiny_solve(solver_psd);
        
        // GATE PLANNER: Check certificate before accepting new plan
        PsdCertificate cert_plan = compute_psd_certificate(solver_psd->solution->x.col(0), disks);
        bool plan_valid = cert_plan.certified;
        
        if (plan_valid) {
            rollout_plan(Ad, Bd, x_seed, solver_psd, &plan);
            plan.start_step = step;
        } else {
            std::cout << "[TinySDP-U] Planner rejected k=" << step
                      << " planner produced invalid solution!"
                      << " trace_gap=" << cert_plan.trace_gap
                      << " eta_min=" << cert_plan.eta_min
                      << " -> Keeping old plan.\n";
            // Keep old plan, don't update
        }

        const char* mode = psd_constraints_active ? "PSD" : "nominal";
        const char* status = plan_valid ? "accepted" : "REJECTED";
        if (csv_plan.is_open()) {
            csv_plan << step << "," << plan.last_iters << "," << sd_seed << "," << mode 
                     << "," << on_thresh << "," << goal_dist << "," << status << "\n";
        }
        std::cout << "[TinySDP-U] Replan at k=" << step
                  << " mode=" << mode
                  << " status=" << status
                  << " iter=" << plan.last_iters
                  << " sd=" << sd_seed 
                  << " on_thresh=" << on_thresh
                  << " off_thresh=" << off_thresh
                  << " goal_dist=" << goal_dist << "\n";
    };

    double sd0 = signed_distance_point_disks(x_track, disks);
    Vec zero_u = Vec::Zero(NU0);
    log_tracking_row(0, x_track, zero_u, sd0, sd0, /*plan_age=*/0, /*iters=*/0);

    // Log certificate at step 0
    Vec x_lift_init = build_lifted(x_track);
    log_certificate(0, x_lift_init);

    double min_sd_track = sd0;

    replan_psd(0, x_track);

    Vec prev_state = x_track;
    for (int k = 0; k < total_steps; ++k) {
        bool need_replan = (k == 0)
                        || (k - plan.start_step >= replan_stride)
                        || (k >= plan.start_step + N - horizon_guard);
        if (need_replan && k > 0) {
            replan_psd(k, x_track);
        }

        set_tracking_refs(solver_track, plan, k, diag_refs_track.first, diag_refs_track.second);
        tiny_set_x0(solver_track, build_lifted(x_track));
        tiny_solve(solver_track);
        Vec u0 = solver_track->solution->u.col(0).topRows(NU0);
        
        // ACTIONABLE GUARANTEE: Gate control on tracker's certificate EVERY step
        // Check tracker's lifted solution is rank-1 / safe before applying
        PsdCertificate cert_apply = compute_psd_certificate(solver_track->solution->x.col(0), disks);
        if (!cert_apply.certified) {
            std::cout << "[TinySDP-U] Emergency stop at k=" << k
                      << " tracker produced non-rank-1 / unsafe solution!"
                      << " trace_gap=" << cert_apply.trace_gap
                      << " eta_min=" << cert_apply.eta_min
                      << " -> Safe stop.\n";
            u0.setZero();  // Safe stop
        }
        
        prev_state = x_track;
        x_track = Ad * x_track + Bd * u0;

        int step_idx = k + 1;
        double sd_point = signed_distance_point_disks(x_track, disks);
        double sd_segment = signed_distance_segment_disks(prev_state, x_track, disks);
        if (sd_segment < min_sd_track) {
            min_sd_track = sd_segment;
        }

        int plan_age = step_idx - plan.start_step;
        log_tracking_row(step_idx, x_track, u0, sd_point, sd_segment,
                         plan_age, solver_track->solution->iter);

        // Log PSD certificate using tracker's lifted solution
        Vec x_lifted_track = solver_track->solution->x.col(0);
        log_certificate(step_idx, x_lifted_track);

        if (goal_reached(x_track)) {
            std::cout << "[TinySDP-U] Goal reached at step " << step_idx
                      << " (pos_norm=" << x_track.topRows(2).norm()
                      << ", vel_norm=" << x_track.bottomRows(2).norm() << ")\n";
            break;
        }
    }

    if (csv_plan.is_open()) csv_plan.close();
    if (csv_track.is_open()) csv_track.close();
    if (csv_cert.is_open()) csv_cert.close();

    std::cout << "\n[TinySDP-U] ===== CLOSED-LOOP CERTIFICATE SUMMARY =====\n";
    std::cout << "[TinySDP-U] Closed-loop min signed distance: " << min_sd_track << "\n";
    std::cout << "[TinySDP-U] Min eta (lifted margin):         " << min_eta << "\n";
    std::cout << "[TinySDP-U] Max |trace_gap|:                 " << max_trace_gap << "\n";
    std::cout << "[TinySDP-U] All steps certified:             " << (all_certified ? "YES" : "NO") << "\n";

    if (max_trace_gap < 1e-6) {
        std::cout << "[TinySDP-U] => Solutions are RANK-1 (trace_gap ~= 0)\n";
    } else {
        std::cout << "[TinySDP-U] => Solutions may NOT be rank-1 (trace_gap = " << max_trace_gap << ")\n";
    }

    // Check BOTH certificate AND actual geometric collision
    bool no_geometric_collision = (min_sd_track >= 0.0);
    
    if (all_certified && no_geometric_collision) {
        std::cout << "[TinySDP-U] => Closed-loop trajectory is collision-free.\n";
    } else if (all_certified && !no_geometric_collision) {
        std::cout << "[TinySDP-U] => WARNING: Certificate passed but GEOMETRIC COLLISION occurred!\n";
        std::cout << "[TinySDP-U] => (Lifted space safe, physical space violated by " << -min_sd_track << ")\n";
    } else if (!all_certified && no_geometric_collision) {
        std::cout << "[TinySDP-U] => Certificate FAILED but no geometric collision.\n";
    } else {
        std::cout << "[TinySDP-U] => UNSAFE: Certificate failed AND geometric collision occurred.\n";
    }

    return 0;
}
