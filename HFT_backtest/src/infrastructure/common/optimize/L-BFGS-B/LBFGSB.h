// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSB_H
#define LBFGSB_H

#include "infrastructure/common/eigen/Eigen"
#include "infrastructure/common/optimize/L-BFGS-B/LBFGSpp/BFGSMat.h"
#include "infrastructure/common/optimize/L-BFGS-B/LBFGSpp/Cauchy.h"
#include "infrastructure/common/optimize/L-BFGS-B/LBFGSpp/LineSearchMoreThuente.h"
#include "infrastructure/common/optimize/L-BFGS-B/LBFGSpp/Param.h"
#include "infrastructure/common/optimize/L-BFGS-B/LBFGSpp/SubspaceMin.h"

#include <stdexcept>  // std::invalid_argument
#include <vector>


namespace LBFGSpp
{
/* -------------------------------------------------------------------------------------------------
L-BFGS-B solver for box-constrained numerical optimization
------------------------------------------------------------------------------------------------- */
template <typename Scalar, template <class> class LineSearch = LineSearchMoreThuente>
class LBFGSBSolver
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1>              Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector>                                    MapVec;
    typedef std::vector<int>                                      IndexSet;

  public:
    LBFGSBSolver(const LBFGSBParam<Scalar> &param) : m_param(param)
    {
        m_param.check_param();
    }

    /* ---------------------------------------------------------------------------------------------
    Minimizing a multivariate function subject to box constraints, using the L-BFGS-B algorithm.
    \param f  A function object such that `f(x, grad)` returns the objective function value at `x`,
              and overwrites `grad` with the gradient.
    \param x  In: An initial guess of the optimal point.
              Out: The best point found.
    \param fx Out: The objective function value at `x`.
    \param lb Lower bounds for `x`.
    \param ub Upper bounds for `x`.
    \return Number of iterations used.
    --------------------------------------------------------------------------------------------- */
    template <typename Foo>
    inline long unsigned int minimize(Foo &f, Vector &x, Scalar &fx, const Vector &lb,
                                      const Vector &ub)
    {
        // Dimension of the vector
        const long int n{x.size()};
        if (lb.size() != n || ub.size() != n)
        {
            throw std::invalid_argument("'lb' and 'ub' must have the same size as 'x'");
        }

        // Check whether the initial vector is within the bounds. If not, project to the feasible
        // set
        force_bounds(x, lb, ub);

        // Initialization
        reset(n);

        // The length of lag for objective function value to test convergence
        const unsigned long int fpast{static_cast<long unsigned int>(m_param.past)};

        // Evaluate function and compute gradient
        fx = f(x, m_grad);

        Scalar projgnorm = proj_grad_norm(x, m_grad, lb, ub);
        if (fpast > 0)
        {
            m_fx[0] = fx;
        }

        // Early exit if the initial x is already a minimizer
        if (projgnorm <= m_param.epsilon || projgnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Compute generalized Cauchy point
        Vector   xcp(n);
        Vector   vecc;
        IndexSet newact_set;
        IndexSet fv_set;
        Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);

        // Initial direction
        m_drt.noalias() = xcp - x;
        m_drt.normalize();
        // Tolerance for s'y >= eps * (y'y)
        const Scalar eps{std::numeric_limits<Scalar>::epsilon()};
        // s and y vectors
        Vector vecs(n);
        Vector vecy(n);

        // Number of iterations used
        long unsigned int k{1};
        for (;;)
        {
            // Save the curent x and gradient
            m_xp.noalias()    = x;
            m_gradp.noalias() = m_grad;

            // Line search to update x, fx and gradient
            Scalar step_max = std::min(m_param.max_step, max_step_size(x, m_drt, lb, ub));
            Scalar step     = std::min(Scalar(1.0), step_max);
            LineSearch<Scalar>::LineSearch(f, fx, x, m_grad, step, step_max, m_drt, m_xp, m_param);

            // New projected gradient norm
            projgnorm = proj_grad_norm(x, m_grad, lb, ub);

            /* std::cout << "** Iteration " << k << std::endl;
            std::cout << "   x = " << x.transpose() << std::endl;
            std::cout << "   f(x) = " << fx << ", ||proj_grad|| = " << projgnorm << std::endl <<
            std::endl; */

            // Convergence test -- gradient
            if (projgnorm <= m_param.epsilon || projgnorm <= m_param.epsilon_rel * x.norm())
            {
                return k;
            }
            // Convergence test -- objective function value
            if (fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];
                if (k >= fpast &&
                    std::abs(fxd - fx) <=
                        m_param.delta * std::max(std::max(std::abs(fx), std::abs(fxd)), Scalar(1)))
                {
                    return k;
                }

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            vecs.noalias() = x - m_xp;
            vecy.noalias() = m_grad - m_gradp;
            if (vecs.dot(vecy) > eps * vecy.squaredNorm())
                m_bfgs.add_correction(vecs, vecy);

            force_bounds(x, lb, ub);
            Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set,
                                             fv_set);

            /*Vector gcp(n);
            Scalar fcp = f(xcp, gcp);
            Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
            std::cout << "xcp = " << xcp.transpose() << std::endl;
            std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl <<
            std::endl;*/

            SubspaceMin<Scalar>::subspace_minimize(m_bfgs, x, xcp, m_grad, lb, ub, vecc, newact_set,
                                                   fv_set, m_param.max_submin, m_drt);

            /*Vector gsm(n);
            Scalar fsm = f(x + m_drt, gsm);
            Scalar projgsmnorm = proj_grad_norm(x + m_drt, gsm, lb, ub);
            std::cout << "xsm = " << (x + m_drt).transpose() << std::endl;
            std::cout << "f(xsm) = " << fsm << ", ||proj_grad|| = " << projgsmnorm << std::endl <<
            std::endl;*/

            k++;
        }

        return k;
    }

  private:
    const LBFGSBParam<Scalar> &m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar, true>      m_bfgs;   // Approximation to the Hessian matrix
    Vector                     m_fx;     // History of the objective function values
    Vector                     m_xp;     // Old x
    Vector                     m_grad;   // New gradient
    Vector                     m_gradp;  // Old gradient
    Vector                     m_drt;    // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    inline void reset(int n)
    {
        m_bfgs.reset(n, m_param.m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if (m_param.past > 0)
        {
            m_fx.resize(m_param.past);
        }
    }

    // Project the vector x to the bound constraint set
    static void force_bounds(Vector &x, const Vector &lb, const Vector &ub)
    {
        x.noalias() = x.cwiseMax(lb).cwiseMin(ub);
    }

    // Norm of the projected gradient: ||P(x-g, l, u) - x||_inf
    static Scalar proj_grad_norm(const Vector &x, const Vector &g, const Vector &lb,
                                 const Vector &ub)
    {
        return ((x - g).cwiseMax(lb).cwiseMin(ub) - x).cwiseAbs().maxCoeff();
    }

    // The maximum step size alpha such that x0 + alpha * d stays within the bounds
    static Scalar max_step_size(const Vector &x0, const Vector &drt, const Vector &lb,
                                const Vector &ub)
    {
        Scalar step{std::numeric_limits<Scalar>::infinity()};
        for (long int i{0}; i < x0.size(); ++i)
        {
            if (drt[i] > Scalar(0.0))
            {
                step = std::min(step, (ub[i] - x0[i]) / drt[i]);
            }
            else if (drt[i] < Scalar(0.0))
            {
                step = std::min(step, (lb[i] - x0[i]) / drt[i]);
            }
        }
        return step;
    }
};
}  // namespace LBFGSpp

#endif  // LBFGSB_H
