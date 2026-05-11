#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <solver/tiny_api.hpp>

// ---------------- svec/smat (half-vectorization with sqrt(2) scaling) -----------------
// Packs a symmetric p x p matrix S into a length m = p(p+1)/2 vector v
// using column-wise lower-triangular order: [S(0,0), S(1,0)*sqrt2, ..., S(p-1,0)*sqrt2,
//                                            S(1,1), S(2,1)*sqrt2, ..., S(p-1,1)*sqrt2, ...]
// Off-diagonals are scaled by sqrt(2) so that:  trace(A^T B) = svec(A)^T svec(B)
inline int svec_size(int p) { return p * (p + 1) / 2; }

inline void smat_inplace(const tinyVector& v, int p, tinyMatrix& S) {
    S.setZero(p, p);
    const tinytype sqrt2 = tinytype(M_SQRT2);
    int idx = 0;
    for (int c = 0; c < p; ++c) {
        // diagonal
        S(c, c) = v(idx++);
        // below diagonal (rows r > c)
        for (int r = c + 1; r < p; ++r) {
            tinytype x = v(idx++) / sqrt2; // invert scaling
            S(r, c) = x;
            S(c, r) = x;
        }
    }
}

inline void svec_inplace(const tinyMatrix& S, tinyVector& v) {
    const int p = S.rows();
    const tinytype sqrt2 = tinytype(M_SQRT2);
    v.setZero(svec_size(p));
    int idx = 0;
    for (int c = 0; c < p; ++c) {
        v(idx++) = S(c, c);
        for (int r = c + 1; r < p; ++r) {
            v(idx++) = sqrt2 * S(r, c);
        }
    }
}

// Manual Kronecker product implementation
template<typename Derived1, typename Derived2>
Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, Eigen::Dynamic>
kronecker(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& B) {
    Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, Eigen::Dynamic> result(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            result.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j) * B;
        }
    }
    return result;
}

// Build lifted (A,B) exactly like the Julia file
inline void tiny_build_lifted_from_base(
    const Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& Ad,
    const Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& Bd,
    Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& A_out,
    Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>& B_out)
{
    using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
    const int nx0 = Ad.rows();
    const int nu0 = Bd.cols();
    const int nxx = nx0*nx0;
    const int nxu = nx0*nu0;
    const int nux = nu0*nx0;
    const int nuu = nu0*nu0;

    A_out.setZero(nx0 + nxx, nx0 + nxx);
    A_out.block(0,0,nx0,nx0)            = Ad;
    A_out.block(nx0,nx0,nxx,nxx)        = kronecker(Ad, Ad);

    B_out.setZero(nx0 + nxx, nu0 + nxu + nux + nuu);
    B_out.block(0,0,nx0,nu0)            = Bd;
    B_out.block(nx0,     nu0,           nxx, nxu) = kronecker(Bd, Ad);
    B_out.block(nx0,     nu0 + nxu,     nxx, nux) = kronecker(Ad, Bd);
    B_out.block(nx0,     nu0 + nxu+nux, nxx, nuu) = kronecker(Bd, Bd);

    std::cout << "[TinySDP] Built lifted A(" << A_out.rows() << "x" << A_out.cols()
              << ") B(" << B_out.rows() << "x" << B_out.cols() << ")\n";
}

// Enable PSD block (stores nx0, nu0 and allocates S/H buffers)
inline int tiny_enable_psd(TinySolver* solver, int nx0, int nu0, tinytype rho_psd)
{
    if (!solver) { std::cout << "tiny_enable_psd: solver nullptr\n"; return 1; }
    solver->settings->en_psd   = 1;
    solver->settings->nx0_psd  = nx0;
    solver->settings->nu0_psd  = nu0;
    solver->cache->rho_psd     = rho_psd;

    const int psd_dim = 1 + nx0 + nu0;
    const int N = solver->work->N;
    const int m = svec_size(psd_dim);

    solver->work->Spsd     = tinyMatrix::Zero(m, N);
    solver->work->Spsd_new = tinyMatrix::Zero(m, N);
    solver->work->Hpsd     = tinyMatrix::Zero(m, N);

    // std::cout << "[TinySDP] Enabled: nx0=" << nx0 << " nu0=" << nu0
    //           << " psd_dim=" << psd_dim << " (half m=" << m << ")"
    //           << " rho_psd=" << rho_psd << "\n";
    return 0;
}

// ---------------- Per-stage linear constraint helpers --------------------
inline int tiny_enable_per_stage_state_linear(TinySolver* solver, int n_constr) {
    if (!solver) { std::cout << "tiny_enable_per_stage_state_linear: solver nullptr\n"; return 1; }
    solver->settings->en_tv_state_linear = 1;
    solver->work->numtvStateLinear = n_constr;
    solver->work->tv_Alin_x = tinyMatrix::Zero(n_constr * solver->work->N, solver->work->nx);
    solver->work->tv_blin_x = tinyMatrix::Zero(n_constr, solver->work->N);
    solver->work->vlnew_tv = solver->work->x;
    solver->work->gl_tv    = tinyMatrix::Zero(solver->work->nx, solver->work->N);
    return 0;
}

// Enable non-time-varying linear state constraints with n_constr rows
inline int tiny_enable_state_linear(TinySolver* solver, int n_constr) {
    if (!solver) { std::cout << "tiny_enable_state_linear: solver nullptr\n"; return 1; }
    solver->settings->en_state_linear = 1;
    solver->work->numStateLinear = n_constr;
    solver->work->Alin_x = tinyMatrix::Zero(n_constr, solver->work->nx);
    solver->work->blin_x = tinyVector::Zero(n_constr);
    // Initialize associated slack/dual storage
    solver->work->vlnew = solver->work->x;
    solver->work->gl    = tinyMatrix::Zero(solver->work->nx, solver->work->N);
    return 0;
}

// Add lifted disk constraints.
// Each disk j with center (ox,oy) and effective radius r contributes a row.
// m^T [x; vec(XX)] >= n  where m = [-2ox, -2oy, e11, e22] and n = r^2 - ||o||^2.
// We store a^T z <= b via a = -m, b = -n into Alin_x, blin_x and enable en_state_linear.
inline int tiny_set_lifted_disks(
    TinySolver* solver,
    const std::vector<std::array<tinytype,3>>& disks)
{
    if (!solver) { std::cout << "tiny_set_lifted_disks: solver nullptr\n"; return 1; }
    const int nx0 = solver->settings->nx0_psd;
    const int nxL = solver->work->nx;
    if (nx0 <= 0) { std::cout << "tiny_set_lifted_disks: nx0_psd not set (>0)\n"; return 1; }
    const int m = static_cast<int>(disks.size());
    if (m == 0) return 0;

    tinyMatrix Alin_x = tinyMatrix::Zero(m, nxL);
    tinyVector blin_x = tinyVector::Zero(m);

    const int idx_xx11 = nx0 + 0 + 0*nx0;
    const int idx_xx22 = nx0 + 1 + 1*nx0;

    for (int j = 0; j < m; ++j) {
        const tinytype ox = disks[j][0];
        const tinytype oy = disks[j][1];
        const tinytype r  = disks[j][2];

        tinyVector mrow = tinyVector::Zero(nxL);
        mrow(0) = -2 * ox; mrow(1) = -2 * oy;
        mrow(idx_xx11) = 1; mrow(idx_xx22) = 1;
        tinytype n = (r*r - (ox*ox + oy*oy));

        tinyVector a = -mrow;
        tinytype   b = -n;
        Alin_x.row(j) = a.transpose();
        blin_x(j) = b;
    }

    // Enable and set in solver
    tiny_enable_state_linear(solver, m);
    return tiny_set_linear_constraints(
        solver,
        Alin_x,
        blin_x,
        tinyMatrix::Zero(0, solver->work->nu),
        tinyVector::Zero(0));
}

// Add lifted sphere constraints in 3D.
// For each sphere with center o=(ox,oy,oz) and effective radius r, the constraint is:
//   (x-o)^T(x-o) >= r^2  =>  x^T x - 2 o^T x >= r^2 - o^T o
// In lifted form, using entries for XX_11, XX_22, XX_33 and base x[0:3):
//   m^T [x; vec(XX)] >= n where m has -2*o in base positions and 1's on XX_11,XX_22,XX_33
// We store a^T z <= b via a = -m, b = -n into Alin_x, blin_x.
inline int tiny_set_lifted_spheres(
    TinySolver* solver,
    const std::vector<std::array<tinytype,4>>& spheres)
{
    if (!solver) { std::cout << "tiny_set_lifted_spheres: solver nullptr\n"; return 1; }
    const int nx0 = solver->settings->nx0_psd;
    const int nxL = solver->work->nx;
    if (nx0 < 3) { std::cout << "tiny_set_lifted_spheres: nx0_psd must be >= 3\n"; return 1; }
    const int m = static_cast<int>(spheres.size());
    if (m == 0) return 0;

    tinyMatrix Alin_x = tinyMatrix::Zero(m, nxL);
    tinyVector blin_x = tinyVector::Zero(m);

    const int idx_xx11 = nx0 + 0 + 0*nx0;
    const int idx_xx22 = nx0 + 1 + 1*nx0;
    const int idx_xx33 = nx0 + 2 + 2*nx0;

    for (int j = 0; j < m; ++j) {
        const tinytype ox = spheres[j][0];
        const tinytype oy = spheres[j][1];
        const tinytype oz = spheres[j][2];
        const tinytype r  = spheres[j][3];

        tinyVector mrow = tinyVector::Zero(nxL);
        mrow(0) = -2 * ox; mrow(1) = -2 * oy; mrow(2) = -2 * oz;
        mrow(idx_xx11) = 1; mrow(idx_xx22) = 1; mrow(idx_xx33) = 1;
        tinytype n = (r*r - (ox*ox + oy*oy + oz*oz));

        tinyVector a = -mrow;
        tinytype   b = -n;
        Alin_x.row(j) = a.transpose();
        blin_x(j) = b;
    }

    // Enable and set in solver
    tiny_enable_state_linear(solver, m);
    return tiny_set_linear_constraints(
        solver,
        Alin_x,
        blin_x,
        tinyMatrix::Zero(0, solver->work->nu),
        tinyVector::Zero(0));
}

inline tinyVector build_lifted_sphere_row(int nx0,
                                          int nxL,
                                          tinytype ox,
                                          tinytype oy,
                                          tinytype oz) {
    tinyVector mrow = tinyVector::Zero(nxL);
    mrow(0) = -2 * ox;
    mrow(1) = -2 * oy;
    mrow(2) = -2 * oz;
    const int idx_xx11 = nx0 + 0 + 0*nx0;
    const int idx_xx22 = nx0 + 1 + 1*nx0;
    const int idx_xx33 = nx0 + 2 + 2*nx0;
    mrow(idx_xx11) = 1;
    mrow(idx_xx22) = 1;
    mrow(idx_xx33) = 1;
    return mrow;
}

inline int tiny_set_lifted_spheres_per_stage(
    TinySolver* solver,
    const std::vector<std::vector<std::array<tinytype,4>>>& spheres_per_stage)
{
    if (!solver) { std::cout << "tiny_set_lifted_spheres_per_stage: solver nullptr\n"; return 1; }
    const int nx0 = solver->settings->nx0_psd;
    const int nxL = solver->work->nx;
    const int N   = solver->work->N;
    if (nx0 < 3) { std::cout << "tiny_set_lifted_spheres_per_stage: nx0_psd must be >= 3\n"; return 1; }
    if (spheres_per_stage.empty()) return 0;

    int per_stage_rows = 0;
    const int stages = std::min<int>(N, spheres_per_stage.size());
    for (int k = 0; k < stages; ++k) {
        per_stage_rows = std::max(per_stage_rows,
                                  static_cast<int>(spheres_per_stage[k].size()));
    }
    if (per_stage_rows == 0) return 0;

    tiny_enable_per_stage_state_linear(solver, per_stage_rows);
    solver->settings->en_tv_state_linear = 1;

    const tinytype relaxed_upper = tinytype(1e6);
    for (int k = 0; k < N; ++k) {
        const auto& stage_spheres = (k < static_cast<int>(spheres_per_stage.size()))
                                  ? spheres_per_stage[k]
                                  : std::vector<std::array<tinytype,4>>{};
        for (int j = 0; j < per_stage_rows; ++j) {
            const int row = k * per_stage_rows + j;
            if (j < static_cast<int>(stage_spheres.size())) {
                const auto& s = stage_spheres[j];
                tinyVector mrow = build_lifted_sphere_row(nx0, nxL, s[0], s[1], s[2]);
                tinytype n = s[3]*s[3] - (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
                solver->work->tv_Alin_x.row(row) = (-mrow).transpose();
                solver->work->tv_blin_x(j, k) = -n;
            } else {
                solver->work->tv_Alin_x.row(row).setZero();
                solver->work->tv_blin_x(j, k) = relaxed_upper;
            }
        }
    }
    return 0;
}
