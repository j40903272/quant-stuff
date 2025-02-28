// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LINE_SEARCH_BACKTRACKING_H
#define LINE_SEARCH_BACKTRACKING_H

#include "infrastructure/common/eigen/Eigen"
#include "infrastructure/common/optimize/L-BFGS-B/LBFGSpp/Param.h"

#include <stdexcept>  // std::runtime_error


namespace LBFGSpp
{
///
/// The backtracking line search algorithm for L-BFGS. Mainly for internal use.
///
template <typename Scalar>
class LineSearchBacktracking
{
  public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f      A function object such that `f(x, grad)` returns the
    ///               objective function value at `x`, and overwrites `grad` with
    ///               the gradient.
    /// \param fx     In: The objective function value at the current point.
    ///               Out: The function value at the new point.
    /// \param x      Out: The new point moved to.
    /// \param grad   In: The current gradient vector. Out: The gradient at the
    ///               new point.
    /// \param step   In: The initial step length. Out: The calculated step length.
    /// \param drt    The current moving direction.
    /// \param xp     The current point.
    /// \param param  Parameters for the LBFGS algorithm
    ///
    template <typename Foo>
    static void LineSearch(Foo &f, Scalar &fx, Vector &x, Vector &grad, Scalar &step,
                           const Vector &drt, const Vector &xp, const LBFGSParam<Scalar> &param)
    {
        // Decreasing and increasing factors
        const Scalar dec{0.5};
        const Scalar inc{2.1};

        // Check the value of step
        if (step <= Scalar(0.0))
        {
            throw std::invalid_argument("'step' must be positive");
        }

        // Save the function value at the current x
        const Scalar fx_init{fx};

        // Projection of gradient on the search direction
        const Scalar dg_init{grad.dot(drt)};

        // Make sure d points to a descent direction
        if (dg_init > 0.0)
        {
            throw std::logic_error("the moving direction increases the objective function value");
        }

        const Scalar test_decr{param.ftol * dg_init};
        Scalar       width;

        size_t iter{0};
        for (; iter < param.max_linesearch; ++iter)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;

            // Evaluate this candidate
            fx = f(x, grad);

            if (fx > fx_init + step * test_decr || (fx != fx))
            {
                width = dec;
            }
            else
            {
                // Armijo condition is met
                if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
                {
                    break;
                }

                const Scalar dg{grad.dot(drt)};
                if (dg < param.wolfe * dg_init)
                {
                    width = inc;
                }
                else
                {
                    // Regular Wolfe condition is met
                    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
                    {
                        break;
                    }

                    if (dg > -param.wolfe * dg_init)
                    {
                        width = dec;
                    }
                    else
                    {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            if (step < param.min_step)
            {
                // throw std::runtime_error(
                //     "the line search step became smaller than the minimum value allowed");
                std::cout << "the line search step [" << step
                          << "] became smaller than the minimum value allowed [" << param.min_step
                          << "]\n";
                return;
            }

            if (step > param.max_step)
            {
                // throw std::runtime_error(
                //     "the line search step became larger than the maximum value allowed");
                std::cout << "the line search step [" << step
                          << "] became larger than the maximum value allowed " << param.max_step
                          << "]\n";
                return;
            }

            step *= width;
        }

        if (iter >= param.max_linesearch)
        {
            // throw std::runtime_error(
            //     "the line search routine reached the maximum number of iterations");
            std::cout << "the line search routine [" << iter
                      << "] reached the maximum number of iterations [" << param.max_linesearch
                      << "]\n";
            return;
        }
    }

  private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
}  // namespace LBFGSpp

#endif  // LINE_SEARCH_BACKTRACKING_H
