#ifndef MATH_H
#define MATH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <xmmintrin.h>

namespace alphaone
{
// README (2020-02-11) [Andrew Kuo]:
// ├─  use these functions:
// │   ├─  y_exp
// │   ├─  y_log
// │   ├─  y_pow
// │   ├─  y_sqrt
// │   ├─  y_invsqrt
// │   ├─  y_sign
// │   └─  y_recip
// ├─  use adjusted versions y_q{} when applying functions below on numbers that might < 1.0:
// │   ├─  y_qpow: y_qpow(v, 0.5) will do y_pow(1.0 + v, 0.5)
// │   └─  y_qsqrt
// ├─  underneath they use:
// │   ├─  exp: c_expf / c_expd
// │   ├─  log: c_logf
// │   ├─  pow: fastpow
// │   ├─  sqrt c_sqrtf
// │   └─  invsqrt: c_invsqrtd
// └─  credits:
//     ├─ c_{} functions are attributed to herumi@nifty.com
//     │  ├─  https://github.com/herumi/fmath/blob/master/fmath.hpp
//     │  └─  http://herumi.in.coocan.jp/soft/fmath.html
//     └─ fast{} functions are attributed to paul@mineiro.com
//        └─  see Copyright notice below

union fi
{
    float        f;
    unsigned int i;
};

union di
{
    double   d;
    uint64_t i;
};

inline unsigned int mask(int x)
{
    return (1U << x) - 1;
}

inline uint64_t mask64(int x)
{
    return (1ULL << x) - 1;
}

static constexpr const size_t EXP_TABLE_SIZE{10};
static constexpr const size_t EXPD_TABLE_SIZE{11};
static constexpr const size_t LOG_TABLE_SIZE{12};

template <size_t N = EXP_TABLE_SIZE>
struct c_expvar
{
    enum
    {
        s   = N,
        n   = 1 << s,
        f88 = 0x42b00000 /* 88.0 */
    };
    float        minX[8];
    float        maxX[8];
    float        a[8];
    float        b[8];
    float        f1[8];
    unsigned int i127s[8];
    unsigned int mask_s[8];
    unsigned int i7fffffff[8];
    unsigned int tbl[n];
    c_expvar()
    {
        float log_2 = ::logf(2.0f);
        for (int i = 0; i < 8; i++)
        {
            maxX[i]      = 88;
            minX[i]      = -88;
            a[i]         = static_cast<double>(n) / log_2;
            b[i]         = log_2 / static_cast<double>(n);
            f1[i]        = 1.0f;
            i127s[i]     = 127 << s;
            i7fffffff[i] = 0x7fffffff;
            mask_s[i]    = mask(s);
        }
        for (int i = 0; i < n; i++)
        {
            float y = pow(2.0f, (float)i / static_cast<double>(n));
            fi    fi;
            fi.f   = y;
            tbl[i] = fi.i & mask(23);
        }
    }
};

template <size_t sbit_ = EXPD_TABLE_SIZE>
struct c_expdvar
{
    enum
    {
        sbit = sbit_,
        s    = 1UL << sbit,
        adj  = (1UL << (sbit + 10)) - (1UL << sbit)
    };

    // A = 1, B = 1, C = 1/2, D = 1/6
    double   C1[2];  // A
    double   C2[2];  // D
    double   C3[2];  // C/D
    uint64_t tbl[s];
    double   a;
    double   ra;
    c_expdvar() : a(static_cast<double>(s) / ::log(2.0)), ra(1 / a)
    {
        for (int i = 0; i < 2; i++)
        {
            C1[i] = 1.0;
            C2[i] = 0.16666666685227835064;
            C3[i] = 3.0000000027955394;
        }
        for (int i = 0; i < s; i++)
        {
            di di;
            di.d   = ::pow(2.0, i * (1.0 / static_cast<double>(s)));
            tbl[i] = di.i & mask64(52);
        }
    }
};

template <size_t N = LOG_TABLE_SIZE>
struct c_logvar
{
    enum
    {
        LEN = N - 1
    };

    unsigned int m1[4];  // 0
    unsigned int m2[4];  // 16
    unsigned int m3[4];  // 32
    float        m4[4];  // 48
    unsigned int m5[4];  // 64

    struct
    {
        float app;
        float rev;
    } tbl[1 << LEN];

    float c_log2;

    c_logvar() : c_log2(::logf(2.0f) / (1 << 23))
    {
        const double e = 1 / double(1 << 24);
        const double h = 1 / double(1 << LEN);
        const size_t n = 1U << LEN;
        for (size_t i = 0; i < n; i++)
        {
            double x   = 1 + double(i) / n;
            double a   = ::log(x);
            tbl[i].app = (float)a;
            if (i < n - 1)
            {
                double b   = ::log(x + h - e);
                tbl[i].rev = (float)((b - a) / ((h - e) * (1 << 23)));
            }
            else
            {
                tbl[i].rev = (float)(1 / (x * (1 << 23)));
            }
        }

        for (int i = 0; i < 4; i++)
        {
            m1[i] = mask(8) << 23;
            m2[i] = mask(LEN) << (23 - LEN);
            m3[i] = mask(23 - LEN);
            m4[i] = c_log2;
            m5[i] = 127U << 23;
        }
    }
};

template <size_t EXP_N = EXP_TABLE_SIZE, size_t LOG_N = LOG_TABLE_SIZE,
          size_t EXPD_N = EXPD_TABLE_SIZE>
struct C
{
    C()
    {
    }
    c_expvar<EXP_N>   expVar;
    c_logvar<LOG_N>   logVar;
    c_expdvar<EXPD_N> expdVar;
};
const C<EXP_TABLE_SIZE, LOG_TABLE_SIZE, EXPD_TABLE_SIZE> const_C;

inline float c_expf(float x)
{
    const c_expvar<> &expVar = const_C.expVar;
    x                        = std::min(x, expVar.maxX[0]);
    x                        = std::max(x, expVar.minX[0]);
    float       t            = x * expVar.a[0];
    const float magic        = (1 << 23) + (1 << 22);  // to round
    t += magic;
    fi fi;
    fi.f           = t;
    t              = x - (t - magic) * expVar.b[0];
    int          u = ((fi.i + (127 << expVar.s)) >> expVar.s) << 23;
    unsigned int v = fi.i & mask(expVar.s);
    fi.i           = u | expVar.tbl[v];
    return (1 + t) * fi.f;
}

inline double c_expd(double x)
{
    if (x <= -708.39641853226408)
        return 0;
    if (x >= 709.78271289338397)
        return std::numeric_limits<double>::infinity();

    const c_expdvar<> &c = const_C.expdVar;
    const uint64_t     b = 3ULL << 51;
    di                 di;
    di.d         = x * c.a + b;
    uint64_t iax = c.tbl[di.i & mask(c.sbit)];

    double   t = (di.d - b) * c.ra - x;
    uint64_t u = (uint64_t)((di.i + c.adj) >> c.sbit) << 52;
    double   y = (c.C3[0] - t) * (t * t) * c.C2[0] - t + c.C1[0];

    di.i = u | iax;
    return y * di.d;
}

inline float c_logf(float x)
{
    const c_logvar<> &logVar = const_C.logVar;
    const size_t      loglen = logVar.LEN;

    fi fi;
    fi.f             = x;
    int          a   = fi.i & (mask(8) << 23);
    unsigned int b1  = fi.i & (mask(loglen) << (23 - loglen));
    unsigned int b2  = fi.i & mask(23 - loglen);
    int          idx = b1 >> (23 - loglen);
    float        f   = float(a - (127 << 23)) * logVar.c_log2 + logVar.tbl[idx].app +
              float(b2) * logVar.tbl[idx].rev;
    return f;
}

// Reference: https://software.intel.com/en-us/node/524256
// Note: putting a very small number into here (say -2e-40) will get you a nan
inline float c_sqrtf(float x)  // more accurate than c_sqrtd
{
    if (x <= 0.0 or std::fabs(x) < FLT_MIN)
        return 0.0;

    float  r = 0.0f;
    __m128 i = _mm_load_ss(&x);
    _mm_store_ss(&r, _mm_mul_ss(i, _mm_rsqrt_ss(i)));
    return r;
}

// Note: more accurate than c_invsqrtd but not very fast. even 1./sqrt(a) seems faster
inline float c_invsqrtf(float x)
{
    float  r = 0.0f;
    __m128 i = _mm_load_ss(&x);
    _mm_store_ss(&r, _mm_rsqrt_ss(i));
    return r;
}

// Note: fast but you lose some accuracy
inline float c_invf(float x)
{
    float  r = 0.0f;
    __m128 i = _mm_load_ss(&x);
    _mm_store_ss(&r, _mm_rcp_ss(i));
    return r;
}

// Note: only marginally faster than a/b, although it is precise
inline float c_invf_precise(float x)
{
    __m128 i = _mm_load_ss(&x);
    float  r = 0.0f;
    _mm_store_ss(&r, _mm_rcp_ss(i));
    float t = r + r;
    float p = x * r * r;
    return t - p;
}

// Reference: https://gamedev.net/forums/topic/500229-optimized-invsqrt-double/
inline double c_invsqrtd(double x)
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
    double  y     = x;
    double  xhalf = (double)0.5 * y;
    int64_t i     = *(int64_t *)(&y);
    i             = int64_t(0x5fe6ec85e7de30da) - (i >> 1);
    y             = *(double *)(&i);
    y             = y * ((double)1.5 - xhalf * y * y);
    return y;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

// Reference: http://stackoverflow.com/questions/1838368/calculating-the-amount-of-combinations
inline int c_choose(int n, int k)
{
    if (k > n)
    {
        return 0;
    }

    int r{1};

    for (int d{1}; d <= k; ++d)
    {
        r *= n--;
        r /= d;
    }

    return r;
}

// BEGIN CODE USED WITH PERMISSION FROM PAUL MINEIRO <paul@mineiro.com>

/*=====================================================================*
 *                   Copyright (C) 2012 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

inline float fastpow2(float p)
{
    float offset = (p < 0) ? 1.0f : 0.0f;
    float clipp  = (p < -126) ? -126.0f : p;
    int   w      = clipp;
    float z      = clipp - w + offset;
    union
    {
        uint32_t i;
        float    f;
    } v = {static_cast<uint32_t>(
        (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z))};

    return v.f;
}

// Note: this is deprecated because of the inaccuracy
inline float fastexp(float p)
{
    return fastpow2(1.442695040f * p);
}

// Note: this is deprecated because of being off by extension to fastlog()
inline float fastlog2(float x)
{
    union
    {
        float    f;
        uint32_t i;
    } vx = {x};
    union
    {
        uint32_t i;
        float    f;
    } mx    = {(vx.i & 0x007FFFFF) | 0x3f000000};
    float y = vx.i;
    y *= 1.1920928955078125e-7f;

    return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
}

// Note: this is deprecated because of being not right for log(1); use c::logf instead
inline double fastlog(double x, double power)
{
    return 0.69314718f * fastlog2(x);
}

// Note: slower for integer powers, but faster for decimals in powers
inline double fastpow(double x, double power)
{
    return fastpow2(power * fastlog2(x));
}

inline double fastsqrt(double x)
{
    return fastpow(x, 0.5);
}

// Note: this is deprecated because of being off
inline float fasterpow2(double p)
{
    float clipp = (p < -126) ? -126.0f : static_cast<float>(p);
    union
    {
        uint32_t i;
        float    f;
    } v = {static_cast<uint32_t>((1 << 23) * (clipp + 126.94269504f))};
    return v.f;
}

// Note: this is deprecated because of being off and slow
inline float fasterlog2(double x)
{
    union
    {
        float    f;
        uint32_t i;
    } vx    = {static_cast<float>(x)};
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 126.94269504f;
}

// Note: this is deprecated because of being off and slow
inline float fasterlog(float x)
{
    union
    {
        float    f;
        uint32_t i;
    } vx    = {x};
    float y = vx.i;
    y *= 8.2629582881927490e-8f;
    return y - 87.989971088f;
}

// Note: this is deprecated because of the inaccuracy
inline double fasterexp(double p)
{
    return fasterpow2(1.442695040f * p);
}

// Note: this is deprecated because of being off
inline double fasterpow(double x, double p)
{
    return fasterpow2(p * fasterlog2(x));
}

// Note: this is deprecated because of being off
inline double fastersqrt(double x)
{
    return fasterpow(x, 0.5);
}

// END CODE USED WITH PERMISSION FROM PAUL MINEIRO <paul@mineiro.com>

// when calculating impluse signals with with a square-root relation observed, e.g., price impact of
// traded size, if the quantity is sometimes below one, then simply applying sqrt on it without
// further considerations will incidentally ruin your results [Andrew Kuo]

static constexpr bool ADJUST_POWERS_FOR_QUANTITIES_BELOW_ONE = false;
// however, after doing some experiemnts, setting ADJUST_POWERS_FOR_QUANTITIES_BELOW_ONE to true
// seems somehow worsen our results -- why? please only set this parameter back to true when the
// performance problem has been solved -- [Andrew Kuo]

inline double y_exp(double v)
{
    return std::isnan(v) ? v : c_expf(v);
}

inline double y_log(double v)
{
    return std::isnan(v) ? v : c_logf(v);
}

inline double y_pow(double v, double exp)
{
    return std::isnan(v) ? v : fastpow(v, exp);
}

inline double y_qpow(double v, double exp)
{
    if (!ADJUST_POWERS_FOR_QUANTITIES_BELOW_ONE)
    {
        return y_pow(v, exp);
    }
    else
    {
        if (std::isnan(v))
        {
            return v;
        }
        else
        {
            return fastpow(1.0 + v, exp);
        }
    }
}

inline double y_sqrt(double v)
{
    return std::isnan(v) ? v : c_sqrtf(v);
}

inline double y_signed_sqrt(double v)
{
    if (std::isnan(v))
    {
        return v;
    }
    else
    {
        return (v >= 0 ? 1.0 : -1.0) * c_sqrtf(std::fabs(v));
    }
}

inline double y_qsqrt(double v)
{
    if (std::isnan(v))
    {
        return v;
    }
    else
    {
        if (!ADJUST_POWERS_FOR_QUANTITIES_BELOW_ONE)
        {
            return y_sqrt(v);
        }
        else
        {
            return c_sqrtf(1.0 + v);
        }
    }
}

inline double y_signed_qsqrt(double v)
{
    if (std::isnan(v))
    {
        return v;
    }
    else
    {
        if (!ADJUST_POWERS_FOR_QUANTITIES_BELOW_ONE)
        {
            return y_signed_sqrt(v);
        }
        else
        {
            return (v >= 0 ? 1.0 : -1.0) * c_sqrtf(1.0 + std::fabs(v));
        }
    }
}

inline double y_invsqrt(double v)
{
    return std::isnan(v) ? v : c_invsqrtd(v);
}

inline int y_sign(double v)
{
    return (v > 0) - (v < 0);
}

inline double y_recip(float recip)
{
    float  res = 0.0;
    __m128 in  = _mm_load_ss(&recip);
    _mm_store_ss(&res, _mm_rcp_ss(in));
    return res;
}
}  // namespace alphaone

#endif
