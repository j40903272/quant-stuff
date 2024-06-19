#ifndef _SMOOTHNUMBER_H_
#define _SMOOTHNUMBER_H_

#include "infrastructure/common/util/Branch.h"

#include <boost/circular_buffer.hpp>
#include <iostream>
#include <numeric>

namespace alphaone
{
class SmoothNumberBase
{
  public:
    SmoothNumberBase() : number_{0.0}
    {
    }

    virtual ~SmoothNumberBase() = default;

    virtual void update(double outcome) = 0;

    virtual double obtain() const
    {
        return number_;
    }

    virtual double *obtain_ptr()
    {
        return &number_;
    }

    virtual void reset()
    {
        number_ = 0.0;
    }

  protected:
    double number_;
};

class SimpleMovingAverage : public SmoothNumberBase
{
  public:
    SimpleMovingAverage(const size_t window) : circle_{window}
    {
    }

    ~SimpleMovingAverage() = default;

    void update(double outcome)
    {
        if (circle_.full())
            sum_ -= circle_.front();
        sum_ += outcome;
        circle_.push_back(outcome);
        number_ = sum_ / circle_.size();
    }

    void reset()
    {
        sum_    = 0.0;
        number_ = 0.0;
    }

  private:
    double                         sum_;
    boost::circular_buffer<double> circle_;
};

class ExponentialMovingAverage : public SmoothNumberBase
{
  public:
    ExponentialMovingAverage(const double coefficient) noexcept
        : coefficient_{coefficient}
        , complementary_coefficient_{1.0 - coefficient}
        , adjustment_{1.0}
        , value_{0.0}
        , prev_outcome_{0.0}
        , is_updated_{false}
    {
    }

    ~ExponentialMovingAverage() = default;

    void update(double outcome)
    {
        value_ = coefficient_ * outcome + complementary_coefficient_ * value_;
        adjustment_ *= (adjustment_ < 0.0000000001 ? 0.0 : complementary_coefficient_);
        number_ = value_ / (1.0 - adjustment_);

        prev_outcome_ = outcome;

        is_updated_ = true;
    }

    void revise(double outcome)
    {
        if (BRANCH_LIKELY(is_updated_))
        {
            value_  = coefficient_ * (outcome - prev_outcome_) + value_;
            number_ = value_ / (1.0 - adjustment_);

            prev_outcome_ = outcome;
        }
        else
        {
            update(outcome);
        }
    }

    void reset()
    {
        value_      = 0.0;
        adjustment_ = 1.0;

        number_ = 0.0;

        is_updated_ = false;
    }

  private:
    const double coefficient_;
    const double complementary_coefficient_;
    double       adjustment_;
    double       value_;
    double       prev_outcome_;
    bool         is_updated_;
};
}  // namespace alphaone

#endif
