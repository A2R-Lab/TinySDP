#include <iostream>

#include <Eigen/Dense>
#include <solver/admm.hpp>
#include <solver/psd_support.hpp>
#include <cmath>

#define DEBUG_MODULE "TINYALG"


/**
 * Update linear terms from Riccati backward pass
*/
void backward_pass_grad(TinySolver *solver)
{
    for (int i = solver->work->N - 2; i >= 0; i--)
    {
        (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i) + solver->cache->BPf);
        (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)) + solver->cache->APf; 
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < solver->work->N - 1; i++)
    {
        (solver->work->u.col(i)).noalias() = -solver->cache->Kinf.lazyProduct(solver->work->x.col(i)) - solver->work->d.col(i);
        (solver->work->x.col(i + 1)).noalias() = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i)) + solver->work->fdyn;
    }
}

/**
 * Project a vector s onto the second order cone defined by mu
 * @param s, mu
 * @return projection onto cone if s is outside cone. Return s if s is inside cone.
*/
tinyVector project_soc(tinyVector s, float mu) {
    tinytype u0 = s(Eigen::last) * mu;
    tinyVector u1 = s.head(s.rows()-1);
    float a = u1.norm();
    tinyVector cone_origin(s.rows());
    cone_origin.setZero();

    if (a <= -u0) { // below cone
        return cone_origin;
    }
    else if (a <= u0) { // in cone
        return s;
    }
    else if (a >= abs(u0)) { // outside cone
        Matrix<tinytype, 3, 1> u2(u1.size() + 1);
        u2 << u1, a/mu;
        return 0.5 * (1 + u0/a) * u2;
    }
    else {
        return cone_origin;
    }
}

// Project onto half-space { z | a^T z <= b } with guards
static inline tinyVector project_halfspace_leq(const tinyVector& z,
                                               const tinyVector& a,
                                               tinytype b) {
    tinytype anorm2 = a.squaredNorm();
    if (!(std::isfinite(anorm2)) || anorm2 <= tinytype(1e-12)) {
        return z; // ill-posed row; skip
    }
    tinytype val = a.dot(z);
    if (!(std::isfinite(val))) return z;
    if (val <= b) return z; // already feasible
    tinytype step = (val - b) / anorm2;
    if (!(std::isfinite(step))) return z;
    // Optional clamp to avoid extreme jumps
    const tinytype clamp_val = 1e3;
    if (step > clamp_val) step = clamp_val;
    if (step < -clamp_val) step = -clamp_val;
    return z - step * a;
}

// ---------- PSD utilities (svec/smat live in psd_support.hpp) ----------

void assemble_psd_block(TinySolver* solver, int k, tinyMatrix& M, bool last)
{
    const int nx0 = solver->settings->nx0_psd;
    const int nu0 = solver->settings->nu0_psd;
    const int psd_dim = 1 + nx0 + nu0;
    M.setZero(psd_dim, psd_dim);
    M(0,0) = 1.0;

    const int nxL = solver->work->nx;
    const int nuL = solver->work->nu;
    const int nxx = nx0*nx0, nxu = nx0*nu0, nux = nu0*nx0, nuu = nu0*nu0;

    // x_bar = [x; vec(XX)] built from consensus/slack if primal is non-finite
    tinyVector xsafe = solver->work->x.col(k);
    if (!xsafe.allFinite()) {
        if (solver->work->vnew.col(k).allFinite()) xsafe = solver->work->vnew.col(k);
        else xsafe.setZero();
    }
    tinyVector x  = xsafe.topRows(nx0);
    tinyVector vXX = xsafe.middleRows(nx0, nxx);
    tinyMatrix XX = Eigen::Map<const tinyMatrix>(vXX.data(), nx0, nx0);

    M.block(0,1,1,nx0) = x.transpose();
    M.block(1,0,nx0,1) = x;
    M.block(1,1,nx0,nx0) = 0.5*(XX + XX.transpose());  // numeric symm

    if (!last) {
        // u_bar = [u; vec(XU); vec(UX); vec(UU)] using consensus/slack if needed
        tinyVector usafe = solver->work->u.col(k);
        if (!usafe.allFinite()) {
            if (solver->work->znew.col(k).allFinite()) usafe = solver->work->znew.col(k);
            else usafe.setZero();
        }
        tinyVector u  = usafe.topRows(nu0);
        tinyVector vXU = usafe.middleRows(nu0, nxu);
        tinyVector vUX = usafe.middleRows(nu0+nxu, nux);
        tinyVector vUU = usafe.bottomRows(nuu);

        tinyMatrix XU = Eigen::Map<const tinyMatrix>(vXU.data(), nx0, nu0);
        tinyMatrix UX = Eigen::Map<const tinyMatrix>(vUX.data(), nu0, nx0);
        tinyMatrix UU = Eigen::Map<const tinyMatrix>(vUU.data(), nu0, nu0);

        M.block(0,1+nx0,1,nu0)         = u.transpose();
        M.block(1+nx0,0,nu0,1)         = u;
        M.block(1,1+nx0,nx0,nu0)       = XU;
        M.block(1+nx0,1,nu0,nx0)       = UX;
        M.block(1+nx0,1+nx0,nu0,nu0)   = 0.5*(UU + UU.transpose());
    }
}

void update_psd_slack(TinySolver *solver)
{
    if (!solver->settings->en_psd) return;
    const int nx0 = solver->settings->nx0_psd;
    const int nu0 = solver->settings->nu0_psd;
    const int psd_dim = 1 + nx0 + nu0;

    for (int k = 0; k < solver->work->N; ++k)
    {
        const bool last = (k == solver->work->N-1);

        tinyMatrix M(psd_dim, psd_dim);
        assemble_psd_block(solver, k, M, last);

        tinyMatrix Hk(psd_dim, psd_dim);
        smat_inplace(solver->work->Hpsd.col(k), psd_dim, Hk);

        // Guard against non-finite input - keep previous S
        if (!M.allFinite() || !Hk.allFinite()) {
            std::cout << "[TinySDP] k=" << k << " non-finite M/H, keeping previous S" << "\n";
            continue;
        }

        // Build Raw and enforce exact symmetry
        tinyMatrix Raw = M + Hk;
        if (!Raw.allFinite()) {
            std::cout << "[TinySDP] k=" << k << " Raw non-finite, keeping previous S" << "\n";
            continue;
        }
        Raw = tinytype(0.5) * (Raw + Raw.transpose());

        // Scale Raw to keep eigensolver in a safe numeric range
        const tinytype RAW_CLIP = 1e6;
        tinytype max_abs = Raw.cwiseAbs().maxCoeff();
        tinytype scale = tinytype(1.0);
        if (std::isfinite(max_abs) && max_abs > RAW_CLIP) {
            scale = max_abs / RAW_CLIP;
        }

        // Add STRONGER jitter on diagonal to make eigensolver robust
        tinytype max_abs_safe = std::isfinite(max_abs) ? max_abs : tinytype(1.0);
        tinytype jitter = std::max(tinytype(1e-10), tinytype(1e-9) * max_abs_safe);  // STRONGER
        tinyMatrix Raw_safe = Raw / scale + jitter * tinyMatrix::Identity(psd_dim, psd_dim);

        // Eigensolver with error checking
        Eigen::SelfAdjointEigenSolver<tinyMatrix> es;
        es.compute(Raw_safe, Eigen::ComputeEigenvectors);
        
        // Check if eigensolver succeeded
        if (es.info() != Eigen::Success) {
            std::cout << "[TinySDP] k=" << k << " eigensolve FAILED; keeping previous S, reducing rho_psd" << "\n";
            // Calm down by reducing rho_psd
            if (solver->cache->rho_psd > 0.1) {
                solver->cache->rho_psd *= 0.5;
                std::cout << "  -> rho_psd reduced to " << solver->cache->rho_psd << "\n";
            }
            continue;
        }

        // Check if eigenvalues/vectors are finite
        tinyVector lam = es.eigenvalues();
        if (!lam.allFinite() || !es.eigenvectors().allFinite()) {
            std::cout << "[TinySDP] k=" << k << " eigen results non-finite; keeping previous S" << "\n";
            continue;
        }

        // EPSILON FLOOR on eigenvalues to avoid rank-deficient outputs
        const tinytype eps_floor = tinytype(1e-12);
        tinytype minlam_raw = lam.minCoeff() * (1.0/scale);
        lam = lam.cwiseMax(eps_floor);  // Floor at eps, not 0!
        
        tinyMatrix Mproj = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
        Mproj *= scale;

        // Final sanity check
        if (!Mproj.allFinite()) {
            std::cout << "[TinySDP] k=" << k << " Mproj non-finite; keeping previous S" << "\n";
            continue;
        }

        tinyVector col(svec_size(psd_dim));
        svec_inplace(Mproj, col);
        solver->work->Spsd_new.col(k) = col;

        // std::cout << "[TinySDP] k=" << k
        //           << " eigmin(raw M+H)=" << minlam_raw
        //           << " -> eigmin(proj)=" << lam.minCoeff() << "\n";
    }
}

void update_psd_dual(TinySolver *solver)
{
    if (!solver->settings->en_psd) return;
    const int nx0 = solver->settings->nx0_psd;
    const int nu0 = solver->settings->nu0_psd;
    const int psd_dim = 1 + nx0 + nu0;

    for (int k = 0; k < solver->work->N; ++k)
    {
        const bool last = (k == solver->work->N-1);

        tinyMatrix M(psd_dim, psd_dim);
        assemble_psd_block(solver, k, M, last);

        tinyMatrix Hk(psd_dim, psd_dim);
        tinyMatrix Snew(psd_dim, psd_dim);
        smat_inplace(solver->work->Hpsd.col(k), psd_dim, Hk);
        smat_inplace(solver->work->Spsd_new.col(k), psd_dim, Snew);

        // Under-relaxed dual update to improve stability
        const tinytype gamma_psd = 0.2;
        Hk = Hk + gamma_psd * (M - Snew);

        // Clip dual magnitude to avoid numerical blow-up
        const tinytype H_CLIP = 1e3;
        for (int r = 0; r < Hk.rows(); ++r) {
            for (int c = 0; c < Hk.cols(); ++c) {
                tinytype v = Hk(r,c);
                if (!std::isfinite(v)) v = 0;
                if (v > H_CLIP) v = H_CLIP;
                if (v < -H_CLIP) v = -H_CLIP;
                Hk(r,c) = v;
            }
        }
        tinyVector tmpH(svec_size(psd_dim));
        svec_inplace(Hk, tmpH);
        solver->work->Hpsd.col(k) = tmpH;
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(TinySolver *solver)
{
    
    // Update bound constraint slack variables for state
    solver->work->vnew = solver->work->x + solver->work->g;
    
    // Update bound constraint slack variables for input
    solver->work->znew = solver->work->u + solver->work->y;

    // Box constraints on state
    if (solver->settings->en_state_bound) {
        solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
    }

    // Box constraints on input
    if (solver->settings->en_input_bound) {
        solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
    }

    
    // Update second order cone slack variables for state
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->vcnew = solver->work->x + solver->work->gc;
    }

    // Update second order cone slack variables for input
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        solver->work->zcnew = solver->work->u + solver->work->yc;
    }

    // Cone constraints on state
    if (solver->settings->en_state_soc) {
        for (int i=0; i<solver->work->N; i++) {
            for (int k=0; k<solver->work->numStateCones; k++) {
                int start = solver->work->Acx(k);
                int num_xs = solver->work->qcx(k);
                tinytype mu = solver->work->cx(k);
                tinyVector col = solver->work->vcnew.block(start, i, num_xs, 1);
                solver->work->vcnew.block(start, i, num_xs, 1) = project_soc(col, mu);
            }
        }
    }

    // Cone constraints on input
    if (solver->settings->en_input_soc) {
        for (int i=0; i<solver->work->N-1; i++) {
            for (int k=0; k<solver->work->numInputCones; k++) {
                int start = solver->work->Acu(k);
                int num_us = solver->work->qcu(k);
                tinytype mu = solver->work->cu(k);
                tinyVector col = solver->work->zcnew.block(start, i, num_us, 1);
                solver->work->zcnew.block(start, i, num_us, 1) = project_soc(col, mu);
            }
        }
    }
    
    // Update linear constraint slack variables for state
    if (solver->settings->en_state_linear) {
        solver->work->vlnew = solver->work->x + solver->work->gl;
    }

    // Update linear constraint slack variables for input
    if (solver->settings->en_input_linear) {
        solver->work->zlnew = solver->work->u + solver->work->yl;
    }

    // Linear constraints on state
    if (solver->settings->en_state_linear) {
        for (int i=0; i<solver->work->N; i++) {
            for (int k=0; k<solver->work->numStateLinear; k++) {
                tinyVector a = solver->work->Alin_x.row(k).transpose();
                tinytype b = solver->work->blin_x(k);
                solver->work->vlnew.col(i) = project_halfspace_leq(solver->work->vlnew.col(i), a, b);
            }
        }
    }

    // Linear constraints on input
    if (solver->settings->en_input_linear) {
        for (int i=0; i<solver->work->N-1; i++) {
            for (int k=0; k<solver->work->numInputLinear; k++) {
                tinyVector a = solver->work->Alin_u.row(k).transpose();
                tinytype b = solver->work->blin_u(k);
                solver->work->zlnew.col(i) = project_halfspace_leq(solver->work->zlnew.col(i), a, b);
            }
        }
    }

    // Update time-varying linear constraint slack variables for state
    if (solver->settings->en_tv_state_linear) {
        solver->work->vlnew_tv = solver->work->x + solver->work->gl_tv;
    }

    // Update time-varying linear constraint slack variables for input
    if (solver->settings->en_tv_input_linear) {
        solver->work->zlnew_tv = solver->work->u + solver->work->yl_tv;
    }

    // Time-varying Linear constraints on state
    if (solver->settings->en_tv_state_linear) {
        for (int i=0; i<solver->work->N; i++) {
            // Sanitize current column to avoid NaN propagation
            if (!solver->work->vlnew_tv.col(i).allFinite()) {
                if (solver->work->x.col(i).allFinite()) {
                    solver->work->vlnew_tv.col(i) = solver->work->x.col(i);
                } else {
                    solver->work->vlnew_tv.col(i).setZero();
                }
            }
            const int nc = solver->work->numtvStateLinear;
            for (int k=0; k<nc; k++) {
                int row = i*nc + k;
                tinyVector a = solver->work->tv_Alin_x.row(row).transpose();
                tinytype b = solver->work->tv_blin_x(k,i);
                solver->work->vlnew_tv.col(i) = project_halfspace_leq(solver->work->vlnew_tv.col(i), a, b);
            }
        }
    }

    // Time-varying Linear constraints on input
    if (solver->settings->en_tv_input_linear) {
        for (int i=0; i<solver->work->N-1; i++) {
            const int nc = solver->work->numtvInputLinear;
            for (int k=0; k<nc; k++) {
                int row = i*nc + k;
                tinyVector a = solver->work->tv_Alin_u.row(row).transpose();
                tinytype b = solver->work->tv_blin_u(k,i);
                solver->work->zlnew_tv.col(i) = project_halfspace_leq(solver->work->zlnew_tv.col(i), a, b);
            }
        }
    }
    
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(TinySolver *solver)
{
    // Update bound constraint dual variables for state
    solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;

    // Update bound constraint dual variables for input
    solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
    
    // Update second order cone dual variables for state
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->gc = solver->work->gc + solver->work->x - solver->work->vcnew;
    }

    // Update second order cone dual variables for input
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        solver->work->yc = solver->work->yc + solver->work->u - solver->work->zcnew;
    }
    
    // Update linear constraint dual variables for state
    if (solver->settings->en_state_linear) {
        solver->work->gl = solver->work->gl + solver->work->x - solver->work->vlnew;
    }

    // Update linear constraint dual variables for input
    if (solver->settings->en_input_linear) {
        solver->work->yl = solver->work->yl + solver->work->u - solver->work->zlnew;
    }
        
    // Update time-varying linear constraint dual variables for state
    if (solver->settings->en_tv_state_linear) {
        solver->work->gl_tv = solver->work->gl_tv + solver->work->x - solver->work->vlnew_tv;
    }

    // Update time-varying linear constraint dual variables for input
    if (solver->settings->en_tv_input_linear) {
        solver->work->yl_tv = solver->work->yl_tv + solver->work->u - solver->work->zlnew_tv;
}
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(TinySolver *solver)
{

    // Update state cost terms
    solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
    (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vcnew - solver->work->gc);
    }
    if (solver->settings->en_state_linear) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vlnew - solver->work->gl);
    }
    if (solver->settings->en_tv_state_linear) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vlnew_tv - solver->work->gl_tv);
    }

    // Update input cost terms
    solver->work->r = -(solver->work->Uref.array().colwise() * solver->work->R.array());
    (solver->work->r).noalias() -= solver->cache->rho * (solver->work->znew - solver->work->y);
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zcnew - solver->work->yc);
    }
    if (solver->settings->en_input_linear) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zlnew - solver->work->yl);
    }
    if (solver->settings->en_tv_input_linear) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zlnew_tv - solver->work->yl_tv);
    }

    // Update terminal cost
    solver->work->p.col(solver->work->N - 1) = -(solver->work->Xref.col(solver->work->N - 1).transpose().lazyProduct(solver->cache->Pinf));
    (solver->work->p.col(solver->work->N - 1)).noalias() -= solver->cache->rho * (solver->work->vnew.col(solver->work->N - 1) - solver->work->g.col(solver->work->N - 1));

    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vcnew.col(solver->work->N - 1) - solver->work->gc.col(solver->work->N - 1));
    }
    if (solver->settings->en_state_linear) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vlnew.col(solver->work->N - 1) - solver->work->gl.col(solver->work->N - 1));
    }
    if (solver->settings->en_tv_state_linear) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vlnew_tv.col(solver->work->N - 1) - solver->work->gl_tv.col(solver->work->N - 1));
    }

    // --- PSD pullback: q/r -= rho_psd * L^T( Snew - H ) -------------------------
    if (solver->settings->en_psd)
    {
        const int nx0 = solver->settings->nx0_psd;
        const int nu0 = solver->settings->nu0_psd;
        const int psd_dim = 1 + nx0 + nu0;
        const int nxx = nx0*nx0, nxu = nx0*nu0, nux = nu0*nx0, nuu = nu0*nu0;

        for (int k = 0; k < solver->work->N; ++k)
        {
            const bool last = (k == solver->work->N-1);
            tinyMatrix Snew(psd_dim, psd_dim);
            tinyMatrix Hk(psd_dim, psd_dim);
            smat_inplace(solver->work->Spsd_new.col(k), psd_dim, Snew);
            smat_inplace(solver->work->Hpsd.col(k), psd_dim, Hk);

            const tinyMatrix T = Snew - Hk; // "znew - y" analog
            if (!T.allFinite()) continue; // guard

            // State pullback
            solver->work->q.col(k).topRows(nx0)     .array() -= solver->cache->rho_psd * T.block(1,0, nx0,1).array();

            tinyVector vXX(nxx);
            Eigen::Map<tinyMatrix>(vXX.data(), nx0, nx0) = T.block(1,1, nx0,nx0);
            solver->work->q.col(k).middleRows(nx0, nxx).array() -= solver->cache->rho_psd * vXX.array();

            if (!last) {
                // Input/lifted-input pullback
                solver->work->r.col(k).topRows(nu0)     .array() -= solver->cache->rho_psd * T.block(1+nx0,0, nu0,1).array();

                tinyVector vXU(nxu), vUX(nux), vUU(nuu);
                Eigen::Map<tinyMatrix>(vXU.data(), nx0, nu0)    = T.block(1,1+nx0, nx0,nu0);
                Eigen::Map<tinyMatrix>(vUX.data(), nu0, nx0)    = T.block(1+nx0,1, nu0,nx0);
                Eigen::Map<tinyMatrix>(vUU.data(), nu0, nu0)    = T.block(1+nx0,1+nx0, nu0,nu0);

                solver->work->r.col(k).middleRows(nu0, nxu)             .array() -= solver->cache->rho_psd * vXU.array();
                solver->work->r.col(k).middleRows(nu0 + nxu, nux)       .array() -= solver->cache->rho_psd * vUX.array();
                solver->work->r.col(k).bottomRows(nuu)                  .array() -= solver->cache->rho_psd * vUU.array();
            }
        }
    }
}

/**
 * Check for termination condition by evaluating whether the largest absolute
 * primal and dual residuals for states and inputs are below threhold.
*/
bool termination_condition(TinySolver *solver)
{
    if (solver->work->iter % solver->settings->check_termination == 0)
    {
        // Primal residuals include box slacks and (if enabled) linear and TV-linear slacks
        tinytype pri_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
        tinytype pri_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();

        if (solver->settings->en_state_linear) {
            tinytype r = (solver->work->x - solver->work->vlnew).cwiseAbs().maxCoeff();
            if (r > pri_state) pri_state = r;
        }
        if (solver->settings->en_input_linear) {
            tinytype r = (solver->work->u - solver->work->zlnew).cwiseAbs().maxCoeff();
            if (r > pri_input) pri_input = r;
        }
        if (solver->settings->en_tv_state_linear) {
            tinytype r = (solver->work->x - solver->work->vlnew_tv).cwiseAbs().maxCoeff();
            if (r > pri_state) pri_state = r;
        }
        if (solver->settings->en_tv_input_linear) {
            tinytype r = (solver->work->u - solver->work->zlnew_tv).cwiseAbs().maxCoeff();
            if (r > pri_input) pri_input = r;
        }

        solver->work->primal_residual_state = pri_state;
        solver->work->primal_residual_input = pri_input;

        // Dual residuals use change in slack variables scaled by rho
        tinytype dua_state = (solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff() * solver->cache->rho;
        tinytype dua_input = (solver->work->z - solver->work->znew).cwiseAbs().maxCoeff() * solver->cache->rho;

        if (solver->settings->en_state_linear) {
            tinytype r = (solver->work->vl - solver->work->vlnew).cwiseAbs().maxCoeff() * solver->cache->rho;
            if (r > dua_state) dua_state = r;
        }
        if (solver->settings->en_input_linear) {
            tinytype r = (solver->work->zl - solver->work->zlnew).cwiseAbs().maxCoeff() * solver->cache->rho;
            if (r > dua_input) dua_input = r;
        }
        if (solver->settings->en_tv_state_linear) {
            tinytype r = (solver->work->vl_tv - solver->work->vlnew_tv).cwiseAbs().maxCoeff() * solver->cache->rho;
            if (r > dua_state) dua_state = r;
        }
        if (solver->settings->en_tv_input_linear) {
            tinytype r = (solver->work->zl_tv - solver->work->zlnew_tv).cwiseAbs().maxCoeff() * solver->cache->rho;
            if (r > dua_input) dua_input = r;
        }

        solver->work->dual_residual_state = dua_state;
        solver->work->dual_residual_input = dua_input;

        if (solver->work->primal_residual_state < solver->settings->abs_pri_tol &&
            solver->work->primal_residual_input < solver->settings->abs_pri_tol &&
            solver->work->dual_residual_state < solver->settings->abs_dua_tol &&
            solver->work->dual_residual_input < solver->settings->abs_dua_tol)
        {
            return true;                 
        }
    }
    return false;
}

int solve(TinySolver *solver)
{
    // Initialize variables
    solver->solution->solved = 0;
    solver->solution->iter = 0;
    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 0;

    // Store previous values for residuals (used for dual residual and adaptive rho)
    tinyMatrix v_prev = solver->work->vnew;
    tinyMatrix z_prev = solver->work->znew;
    
    // Initialize SOC slack variables if needed
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->vcnew = solver->work->x;
    }
    
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        solver->work->zcnew = solver->work->u;
    }

    // Initialize linear constraint slack variables if needed
    if (solver->settings->en_state_linear) {
        solver->work->vlnew = solver->work->x;
    }
    
    if (solver->settings->en_input_linear) {
        solver->work->zlnew = solver->work->u;
    }

    // Initialize time-varying linear constraint slack variables if needed
    if (solver->settings->en_tv_state_linear) {
        solver->work->vlnew_tv = solver->work->x;
    }
    
    if (solver->settings->en_tv_input_linear) {
        solver->work->zlnew_tv = solver->work->u;
    }

    for (int i = 0; i < solver->settings->max_iter; i++)
    {
        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost(solver);

        // Solve linear system with Riccati and roll out to get new trajectory
        backward_pass_grad(solver);

        forward_pass(solver);

        // Project slack variables into feasible domain
        update_slack(solver);

        // NEW: PSD projector (prints per-stage eigmins). Guard non-finite input.
        update_psd_slack(solver);

        // Compute next iteration of dual variables
        update_dual(solver);

        // NEW: PSD dual update
        update_psd_dual(solver);

        solver->work->iter += 1;

        // Handle adaptive rho if enabled - OSQP-style using native residuals.
        if (solver->settings->adaptive_rho && i > 0 && i % 25 == 0) {
            // Use the solver's native residuals, already computed for termination.
            tinytype pri_res = std::max(
                (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff(),
                (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff()
            );
            tinytype dua_res = std::max(
                solver->cache->rho * (solver->work->vnew - v_prev).cwiseAbs().maxCoeff(),
                solver->cache->rho * (solver->work->znew - z_prev).cwiseAbs().maxCoeff()
            );
            
            // OSQP formula: rho_new = rho * sqrt(pri_res / dual_res)
            const tinytype eps = 1e-10;
            if (pri_res > eps && dua_res > eps) {
                tinytype old_rho = solver->cache->rho;
                tinytype ratio = pri_res / dua_res;
                tinytype new_rho = old_rho * std::sqrt(ratio);
                
                // Clamp to reasonable bounds
                new_rho = std::min(std::max(new_rho, solver->settings->adaptive_rho_min), 
                                  solver->settings->adaptive_rho_max);
                
                // Only update if change is significant (avoid tiny oscillations)
                if (std::abs(new_rho - old_rho) / old_rho > 0.1) {
                    solver->cache->rho = new_rho;
                    if (i % 50 == 0) {
                        std::cout << "[ADAPT-RHO] iter=" << i 
                                  << " rho: " << old_rho << " -> " << new_rho
                                  << " pri=" << pri_res << " dua=" << dua_res << "\n";
                    }
                }
            }
        }

        
            
        // Store previous values for next iteration
        z_prev = solver->work->znew;
        v_prev = solver->work->vnew;

        // Check for whether cost is minimized by calculating residuals
        if (termination_condition(solver)) {
            solver->work->status = 1; // TINY_SOLVED

            // Save solution
            solver->solution->iter = solver->work->iter;
            solver->solution->solved = 1;
            solver->solution->x = solver->work->vnew;
            solver->solution->u = solver->work->znew;

            std::cout << "Solver converged in " << solver->work->iter << " iterations" << std::endl;

            return 0;
        }

        // Save previous slack variables for residual computation in next iter
        solver->work->v = solver->work->vnew;
        solver->work->z = solver->work->znew;
        if (solver->settings->en_state_linear)  solver->work->vl   = solver->work->vlnew;
        if (solver->settings->en_input_linear)  solver->work->zl   = solver->work->zlnew;
        if (solver->settings->en_tv_state_linear)  solver->work->vl_tv = solver->work->vlnew_tv;
        if (solver->settings->en_tv_input_linear)  solver->work->zl_tv = solver->work->zlnew_tv;
      
    }
    
    solver->solution->iter = solver->work->iter;
    solver->solution->solved = 0;
    solver->solution->x = solver->work->vnew;
    solver->solution->u = solver->work->znew;
    return 1;
}
