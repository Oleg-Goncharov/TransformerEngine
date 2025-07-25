/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTILS_CUH_
#define TRANSFORMER_ENGINE_COMMON_UTILS_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#if !defined(__CUDACC_RTC__)
#include <cstdint>
#else
// Importing C++ standard headers is a pain with NVRTC
using uint8_t = unsigned char;
using uint16_t = unsigned short int;  // NOLINT(*)
using uint32_t = unsigned int;
using uint64_t = unsigned long long int;  // NOLINT(*)
static_assert(sizeof(uint8_t) == 1);
static_assert(sizeof(uint16_t) == 2);
static_assert(sizeof(uint32_t) == 4);
static_assert(sizeof(uint64_t) == 8);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t THREADS_PER_WARP = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 operator+(const float2 &a, const float2 &b) {  // NOLINT(*)
  return {a.x + b.x, a.y + b.y};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void operator+=(float2 &a, const float2 &b) {  // NOLINT(*)
  a.x += b.x;
  a.y += b.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Sum {
  inline __device__ Sum() {}
  inline __device__ T operator()(const T &a, const T &b) const { return a + b; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ T warp_shuffle_xor(const T &x, uint32_t idx) {
  return __shfl_xor_sync(static_cast<uint32_t>(-1), x, idx);
}

template <>
inline __device__ float2 warp_shuffle_xor<float2>(const float2 &x, uint32_t idx) {
  return {warp_shuffle_xor(x.x, idx), warp_shuffle_xor(x.y, idx)};
}

template <typename T>
inline __device__ T warp_shuffle_down(const T &x, uint32_t idx) {
  return __shfl_down_sync(static_cast<uint32_t>(-1), x, idx);
}

template <>
inline __device__ float2 warp_shuffle_down<float2>(const float2 &x, uint32_t idx) {
  return {warp_shuffle_down(x.x, idx), warp_shuffle_down(x.y, idx)};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace transformer_engine {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint16 {
  uint4 u;
  uint4 v;
  uint4 s;
  uint4 t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint8 {
  uint4 u;
  uint4 v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<64> {
  using Type = uint16;
  static_assert(sizeof(Type) == 64);
};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct TypeToVec2 {};

template <>
struct TypeToVec2<float> {
  using Type = float2;
};

template <>
struct TypeToVec2<half> {
  using Type = half2;
};

template <>
struct TypeToVec2<nv_bfloat16> {
  using Type = nv_bfloat162;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename IType, typename IType2, typename OType, typename CType>
struct CTDBiasDActParam {
  using InputType = IType;
  using InputType2 = IType2;
  using OutputType = OType;
  using ComputeType = CType;
  const IType *input;
  const IType2 *act_input;
  OType *output_c;
  OType *output_t;
  const CType *scale_ptr;
  CType *amax;
  CType *scale_inv;
  CType *workspace;
  CType *warp_scales_inv;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int INDEX>
struct Get {
  template <typename T, typename R>
  static inline __device__ R of(const T &vec);
};

template <>
template <typename T, typename R>
inline __device__ R Get<0>::of(const T &vec) {
  return vec.x;
}

template <>
template <typename T, typename R>
inline __device__ R Get<1>::of(const T &vec) {
  return vec.y;
}

template <>
template <typename T, typename R>
inline __device__ R Get<2>::of(const T &vec) {
  return vec.z;
}

template <>
template <typename T, typename R>
inline __device__ R Get<3>::of(const T &vec) {
  return vec.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Src, typename Dst>
struct Converter {
  static inline __device__ Dst convert(const Src &from) { return Dst(from); }
};

template <>
struct Converter<float2, half2> {
  static inline __device__ half2 convert(const float2 &x) { return __float22half2_rn(x); }
};

template <>
struct Converter<float2, nv_bfloat162> {
  static inline __device__ nv_bfloat162 convert(const float2 &x) {
#if __CUDA_ARCH__ >= 800
    return __float22bfloat162_rn(x);
#else
    union {
      nv_bfloat162 raw;
      nv_bfloat16 elt[2];
    } tmp;
    tmp.elt[0] = __float2bfloat16_rn(x.x);
    tmp.elt[1] = __float2bfloat16_rn(x.y);
    return tmp.raw;
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Zeros {
  static inline __device__ T get() { return T(0.f); }
};

template <>
struct Zeros<float2> {
  static inline __device__ float2 get() { return make_float2(0.f, 0.f); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Elt_type, uint32_t NUM_ELT>
struct Vec {
  enum { BYTES = NUM_ELT * sizeof(Elt_type) };

  using Vec_type = typename BytesToType<BYTES>::Type;
  using type = Elt_type;

  using Alias_type = union {
    Vec_type vec;
    Elt_type elt[NUM_ELT];
  };

  Alias_type data;

  template <typename S>
  inline __device__ void to(Vec<S, NUM_ELT> &other) {  // NOLINT(*)
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      other.data.elt[it] = S(this->data.elt[it]);
    }
  }

  template <typename Op>
  inline __device__ void assign(const Op &op) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = op(it);
    }
  }

  // Pointer is cast to vector type
  inline __device__ void load_from(const void *base_ptr, size_t idx = 0) {
    this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
  }

  // Pointer is cast to vector type
  inline __device__ void store_to(void *base_ptr, size_t idx = 0) const {
    static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
  }

  // Pointer is cast to element type. Loads min(count, NUM_ELT)
  // elements and any remaining elements are set to zero.
  inline __device__ void load_from_elts(const void *base_ptr, size_t idx = 0,
                                        size_t count = NUM_ELT) {
    const Elt_type *elt_ptr = static_cast<const Elt_type *>(base_ptr) + idx;
    if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        this->data.elt[it] = (it < count ? elt_ptr[it] : Elt_type(0.f));
      }
    } else {
      this->load_from(elt_ptr);
    }
  }

  // Pointer is cast to element type. Stores min(count, NUM_ELT)
  // elements.
  inline __device__ void store_to_elts(void *base_ptr, size_t idx = 0,
                                       size_t count = NUM_ELT) const {
    Elt_type *elt_ptr = static_cast<Elt_type *>(base_ptr) + idx;
    if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        if (it < count) {
          elt_ptr[it] = this->data.elt[it];
        }
      }
    } else {
      this->store_to(elt_ptr);
    }
  }

  inline __device__ void clear() {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = Elt_type(0.f);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct InterCTASync {
  inline __device__ InterCTASync(int *barrier, int group, int num_groups, int group_size)
      : phase_counter_(0),
        b0_(barrier + group)  // The barrier for this group of CTAs.
        ,
        b1_(barrier + group + num_groups)  // The barrier for this group of CTAs.
        ,
        group_size_(group_size) {
    // BARRIERS ARE ASSUMED TO BE INITIALIZED TO 0!
  }

  inline __device__ void spin_wait_(int *barrier, int step, int expected) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(barrier), "r"(step));
    for (int found = -1; found != expected;) {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(barrier));
    }
  }

  inline __device__ void sync() {
    // ALL THREADS MUST ENTER!

    // We switch barrier every iteration.
    int *barrier = phase_counter_ & 0x1 ? b1_ : b0_;
    // We decrement every other iteration.
    bool dec = phase_counter_ & 0x2;
    int step = dec ? -1 : 1;
    int expected = dec ? 0 : group_size_;
    // There are only 4 phases: up/down for b0/b1.
    phase_counter_ = (phase_counter_ + 1) & 0x3;

    if (threadIdx.x == 0) {
      spin_wait_(barrier, step, expected);
    }
    // CTA waits for thread 0
    __syncthreads();
  }

  int phase_counter_;
  int *b0_;
  int *b1_;
  int group_size_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t CTAS_PER_ROW, uint32_t WARPS_M, uint32_t WARPS_N>
struct Reducer : public Reducer<T, 1, WARPS_M, WARPS_N> {
  using Base = Reducer<T, 1, WARPS_M, WARPS_N>;
  using Type = typename Base::Type;

  enum { SMEM_BYTES = Base::SMEM_BYTES };

  enum { WS_BARRIER_BYTES = 2 * sizeof(int) };
  enum { WS_DATA_BYTES = WARPS_M * CTAS_PER_ROW * sizeof(T) };

  // size of the barriers + temporary result per CTA (multiply with CTAS_PER_ROW to get total)
  enum {
    WORKSPACE_BYTES_PER_GROUP = Base::WORKSPACE_BYTES_PER_GROUP + WS_BARRIER_BYTES + WS_DATA_BYTES
  };

  template <typename Params>
  inline __device__ Reducer(const Params &params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                            uint32_t warp_n, uint32_t lane, void *smem)
      : Base(params, bidm, bidn, warp_m, warp_n, lane, smem),
        inter_cta_(params.barrier, bidm, params.ctas_per_col, CTAS_PER_ROW),
        bidn_(bidn)  // CTA id within the group.
        ,
        w0_(static_cast<T *>(params.workspace) + (bidm * WARPS_M + warp_m) * CTAS_PER_ROW),
        w1_(w0_ + params.ctas_per_col * WARPS_M * CTAS_PER_ROW) {}

  template <typename Op>
  inline __device__ T allreduce(T data, const Op &op) {
    data = Base::reduce(data, op);
    // We switch workspace every iteration.
    T *const workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

    // Warp leaders 0 hold the CTA-local results.
    if (this->warp_n_ == 0 && this->lane_ == 0) {
      workspace[bidn_] = data;
    }
    inter_cta_.sync();
    static_assert(CTAS_PER_ROW <= 32);
    T total = Zeros<T>::get();
    if (this->lane_ < CTAS_PER_ROW) {
      total = workspace[this->lane_];
    }
    total = Reducer<T, 1, 1, 1>::allreduce_(total, op);

    return total;
  }

  InterCTASync inter_cta_;

  T *const w0_;
  T *const w1_;
  int bidn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t WARPS_M>
struct Reducer<T, 1, WARPS_M, 1> {
  using Type = T;
  enum { SMEM_BYTES = 0 };
  enum { WORKSPACE_BYTES_PER_GROUP = 0 };

  enum { THREADS_PER_WARP = 32 };

  template <typename Params>
  inline __device__ Reducer(const Params &params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                            uint32_t warp_n, uint32_t lane, void *smem)
      : warp_n_(warp_n), lane_(lane) {}

  template <typename Op>
  static inline __device__ T allreduce_(T data, const Op &op) {
#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      data = op(data, warp_shuffle_xor(data, it));
    }
    return data;
  }

  template <typename Op>
  inline __device__ T allreduce(T data, const Op &op) {
    return allreduce_(data, op);
  }

  template <typename Op>
  inline __device__ T reduce(T data, const Op &op) {
// only lane 0 holds the result!
#pragma unroll
    for (int it = THREADS_PER_WARP / 2; it > 0; it /= 2) {
      data = op(data, warp_shuffle_down(data, it));
    }
    return data;
  }
  int warp_n_;
  int lane_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct Reducer<T, 1, WARPS_M, WARPS_N> : public Reducer<T, 1, WARPS_M, 1> {
  using Base = Reducer<T, 1, WARPS_M, 1>;

  using Type = T;

  enum { SMEM_BYTES = Base::SMEM_BYTES + WARPS_M * WARPS_N * sizeof(T) * 2 };
  enum { WORKSPACE_BYTES_PER_GROUP = 0 };

  enum { THREADS_PER_WARP = 32 };

  template <typename Params>
  inline __device__ Reducer(const Params &params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                            uint32_t warp_n, uint32_t lane, void *smem)
      : Base(params, bidm, bidn, warp_m, warp_n, lane, smem),
        use0_(true),
        smem0_(&(static_cast<T *>(smem)[warp_m * WARPS_N])),
        smem1_(smem0_ + WARPS_M * WARPS_N) {}

  template <typename Op>
  inline __device__ T allreduce(T data, const Op &op) {
    T *const smem = use0_ ? smem0_ : smem1_;
    use0_ = !use0_;
    data = Base::reduce(data, op);
    if (this->lane_ == 0) {
      smem[this->warp_n_] = data;
    }
    __syncthreads();
    T out = Zeros<T>::get();
#pragma unroll
    for (int it = 0; it < WARPS_N; it++) {
      out = op(out, smem[it]);
    }
    return out;
  }

  template <typename Op>
  inline __device__ T reduce(T data, const Op &op) {
    T *const smem = use0_ ? smem0_ : smem1_;
    use0_ = !use0_;
    // only intra-CTA group leader holds the result!
    data = Base::reduce(data, op);
    if (this->lane_ == 0) {
      smem[this->warp_n_] = data;
    }
    __syncthreads();
    T out = Zeros<T>::get();
    if (this->warp_n_ == 0 && this->lane_ == 0) {
#pragma unroll
      for (int it = 0; it < WARPS_N; it++) {
        out = op(out, smem[it]);
      }
    }
    return out;
  }

  T *const smem0_;
  T *const smem1_;
  bool use0_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct DynamicReducer : public Reducer<T, 1, WARPS_M, WARPS_N> {
  using Base = Reducer<T, 1, WARPS_M, WARPS_N>;
  using Type = typename Base::Type;

  template <typename Params>
  inline __device__ DynamicReducer(const Params &params, uint32_t bidm, uint32_t bidn,
                                   uint32_t warp_m, uint32_t warp_n, uint32_t lane, void *smem)
      : Base(params, bidm, bidn, warp_m, warp_n, lane, smem),
        inter_cta_(params.barrier, bidm, params.ctas_per_col, params.ctas_per_row),
        bidn_(bidn)  // CTA id within the group.
        ,
        w0_(static_cast<T *>(params.workspace) + (bidm * WARPS_M + warp_m) * params.ctas_per_row),
        w1_(w0_ + params.ctas_per_col * WARPS_M * params.ctas_per_row) {}

  template <typename Op>
  inline __device__ T allreduce(T data, const Op &op) {
    // Trivial case
    if (inter_cta_.group_size_ == 1) {
      return Base::allreduce(data, op);
    }

    data = Base::reduce(data, op);
    // We switch workspace every iteration.
    T *const workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

    // Warp leaders 0 hold the CTA-local results.
    if (this->warp_n_ == 0 && this->lane_ == 0) {
      workspace[bidn_] = data;
    }
    inter_cta_.sync();
    T total = Zeros<T>::get();
    for (int it = this->lane_; it < inter_cta_.group_size_; it += THREADS_PER_WARP) {
      total = op(total, workspace[it]);
    }
    total = Reducer<T, 1, 1, 1>::allreduce_(total, op);

    return total;
  }

  template <typename Op>
  inline __device__ T reduce(T data, const Op &op) {
    return allreduce(data, op);
  }

  InterCTASync inter_cta_;

  T *const w0_;
  T *const w1_;
  int bidn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
This is an implementation of the parallel Welford algorithm for incrementally computing variance

This algorithm is known as Chan's update formulae (Chat et al '79):
http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

An introduction is provided by Wikipedia here:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance?section=5#Parallel_algorithm

A detailed reference on the exact version implemented (with better numerical stability) is provided here:
https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
*/

template <typename T>
inline __device__ void warp_chan_upd_dynamic(T &m_a, T &m2_a, T &n_a,
                                             int num_active) {  // NOLINT(*)
  // Assume at least leftmost is valid and
  // init: step = next_pow2(num_active) / 2 (might get NaN otherwise)
  int highest_bit_set = (8 * sizeof(num_active)) - __clz(num_active - 1);

#pragma unroll
  for (int step = (1 << (highest_bit_set - 1)); step > 0; step /= 2) {
    // Exchange
    T n_b = warp_shuffle_down(n_a, step);
    T m_b = warp_shuffle_down(m_a, step);
    T m2_b = warp_shuffle_down(m2_a, step);

    // Update
    const T n_ab = n_a + n_b;  // We can handle one of them being 0, not both.
    // Might have different n per thread, otherwise this would simplify :(
    const T rn_ab = 1.f / n_ab;
    const T delta = m_a - m_b;
    const float m2_ab = m2_a + m2_b + delta * delta * n_a * n_b * rn_ab;
    const float m_ab = (n_a * m_a + n_b * m_b) * rn_ab;

    n_a = n_ab;
    m_a = m_ab;
    m2_a = m2_ab;
  }
  // Intra-warp broadcast (only lane 0 has valid stats).
  m_a = __shfl_sync(static_cast<uint32_t>(-1), m_a, 0);
  m2_a = __shfl_sync(static_cast<uint32_t>(-1), m2_a, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t CTAS_PER_ROW, uint32_t WARPS_M, uint32_t WARPS_N>
struct Stats {
  // This could be done generically with the Reducer. But then we
  // would have to exchange 3 instead of 2 fields.

  using BlockStats = Stats<T, 1, WARPS_M, WARPS_N>;
  using stats_t = typename BlockStats::stats_t;

  enum { SMEM_BYTES = BlockStats::SMEM_BYTES };

  template <typename Params>
  inline __device__ Stats(const Params &params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                          uint32_t warp_n, uint32_t lane, void *smem)
      : inter_cta_(params.barrier, bidm, params.ctas_per_col, CTAS_PER_ROW),
        block_stats_(params, bidm, bidn, warp_m, warp_n, lane, smem),
        bidn_(bidn)  // CTA id within the group.
        ,
        w0_(static_cast<stats_t *>(params.workspace) + (bidm * WARPS_M + warp_m) * CTAS_PER_ROW),
        w1_(w0_ + params.ctas_per_col * WARPS_M * CTAS_PER_ROW),
        warp_n_(warp_n),
        lane_(lane) {}

  template <uint32_t N>
  inline __device__ stats_t compute(const T (&elts)[N], const T rn) {
    constexpr T ELTS_PER_ROW_PER_CTA = N * WARPS_N * THREADS_PER_WARP;
    // TODO(ptredak) rn is not really needed here..
    constexpr T block_rn = 1.f / T(ELTS_PER_ROW_PER_CTA);
    stats_t block_stats = block_stats_.compute(elts, block_rn);

    stats_t *const workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

    if (warp_n_ == 0 && lane_ == 0) {
      workspace[bidn_] = block_stats;
    }

    // Wait for all CTAS_PER_ROW CTAS in the group to have written their result.
    inter_cta_.sync();

    T n = Zeros<T>::get();
    T m = Zeros<T>::get();
    T m2 = Zeros<T>::get();

    // Assume CTA group size in N less than 32, such that we can finalize with a single warp.
    static_assert(CTAS_PER_ROW <= 32);

    // Every warp does the final reduction locally.
    if (lane_ < CTAS_PER_ROW) {
      stats_t result = workspace[lane_];
      n = ELTS_PER_ROW_PER_CTA;
      m = transformer_engine::Get<0>::of<stats_t, T>(result);
      m2 = transformer_engine::Get<1>::of<stats_t, T>(result);
    }

    warp_chan_upd_dynamic(m, m2, n, CTAS_PER_ROW);

    return {m, m2};
  }

  InterCTASync inter_cta_;
  BlockStats block_stats_;

  stats_t *const w0_;
  stats_t *const w1_;
  int bidn_;
  int warp_n_;
  int lane_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct Stats<T, 1, WARPS_M, WARPS_N> {
  using WarpStats = Stats<T, 1, WARPS_M, 1>;
  using stats_t = typename WarpStats::stats_t;

  enum { SMEM_BYTES = WARPS_M * WARPS_N * sizeof(stats_t) * 2 };

  template <typename Params>
  inline __device__ Stats(const Params &params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                          uint32_t warp_n, uint32_t lane, void *smem)
      : warp_stats_(params, bidm, bidn, warp_m, warp_n, lane, smem), use0_(true) {
    smem0_ = static_cast<stats_t *>(smem) + warp_m * WARPS_N;
    smem1_ = smem0_ + WARPS_M * WARPS_N;
  }

  template <uint32_t N>
  inline __device__ stats_t compute(const T (&elts)[N], const T rn) {
    stats_t *smem = use0_ ? smem0_ : smem1_;
    use0_ = !use0_;
    // Compute warp local for all WARPS_N
    constexpr T warp_rn = 1.f / T(N * THREADS_PER_WARP);
    stats_t warp_stats = warp_stats_.compute(elts, warp_rn);

    // Each warp warp leader stores its stats
    const auto warp_n = warp_stats_.reducer_.warp_n_;
    const auto lane = warp_stats_.reducer_.lane_;
    if (lane == 0) {
      smem[warp_n] = warp_stats;
    }
    __syncthreads();

    T n = Zeros<T>::get();
    T m = Zeros<T>::get();
    T m2 = Zeros<T>::get();

    // Assume that there are less than 32 warps, such that we can finalize with a single warp
    static_assert(WARPS_N <= 32);
    if (lane < WARPS_N) {
      stats_t result = smem[lane];
      n = N * THREADS_PER_WARP;
      m = transformer_engine::Get<0>::of<stats_t, T>(result);
      m2 = transformer_engine::Get<1>::of<stats_t, T>(result);
    }

    warp_chan_upd_dynamic(m, m2, n, WARPS_N);

    return {m, m2};
  }
  WarpStats warp_stats_;
  stats_t *smem0_;
  stats_t *smem1_;
  bool use0_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32_t WARPS_M>
struct Stats<T, 1, WARPS_M, 1> {
  using stats_t = typename TypeToVec2<T>::Type;
  // The simple Warp reducer.
  using Reducer = Reducer<T, 1, WARPS_M, 1>;

  enum { SMEM_BYTES = 0 };

  template <typename Params>
  inline __device__ Stats(const Params &params, uint32_t bidm, uint32_t bidn, uint32_t warp_m,
                          uint32_t warp_n, uint32_t lane, void *smem)
      : reducer_(params, bidm, bidn, warp_m, warp_n, lane, smem) {}

  template <uint32_t N>
  inline __device__ stats_t compute(const T (&elts)[N], const T rn) {
    auto sum = Sum<T>();

    T m = Zeros<T>::get();
#pragma unroll
    for (int it = 0; it < N; it++) {
      m += elts[it];
    }
    m = reducer_.allreduce(m, sum) * rn;

    T m2 = Zeros<T>::get();
#pragma unroll
    for (int it = 0; it < N; it++) {
      T diff = (elts[it] - m);
      m2 += diff * diff;
    }
    m2 = reducer_.allreduce(m2, sum);

    return {m, m2};
  }

  Reducer reducer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int num_elems>
__device__ __forceinline__ float warp_reduce_max(const float m) {
  float tmp = m;
#pragma unroll
  for (int delta = num_elems / 2; delta > 0; delta /= 2) {
    const float other_m = __shfl_down_sync(0xFFFFFFFF, tmp, delta);
    __builtin_assume(tmp >= 0);
    __builtin_assume(other_m >= 0);
    tmp = fmaxf(tmp, other_m);
  }
  return tmp;
}

__forceinline__ __device__ float warp_reduce_max_broadcast(const float val) {
  float val_tmp = val;
#pragma unroll
  for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    const float val_other = __shfl_down_sync(0xFFFFFFFF, val_tmp, offset);
    __builtin_assume(val_tmp >= 0);
    __builtin_assume(val_other >= 0);
    val_tmp = fmaxf(val_tmp, val_other);
  }
  // Broadcast the amax to other threads of the subwarp from the zero subwarp lane_id
  constexpr int subwarp_lane_zero = 0;
  val_tmp = __shfl_sync(0xFFFFFFFF, val_tmp, subwarp_lane_zero);
  return val_tmp;
}

template <int num_warps, typename compute_t>
__device__ __forceinline__ compute_t reduce_max(const compute_t m, const int warpid) {
  __shared__ float staging[num_warps];
  constexpr int warp_size = 32;
  const float my_max = m;
  const float my_warp_max = warp_reduce_max<warp_size>(my_max);
  if (threadIdx.x % 32 == 0) {
    staging[warpid] = my_warp_max;
  }
  __syncthreads();
  compute_t result = 0.f;
  if (warpid == 0) {
    const float my_max = threadIdx.x < num_warps ? staging[threadIdx.x] : 0;
    result = warp_reduce_max<num_warps>(my_max);
  }
  return result;
}

/**
 * Max reduction in subwarps
 * E.g., if nvec=4, each warp processes 128 elements (32 x 4), that covers four MXFP8 scaling factors.
 * To compute an actual scaling factor for 32 consequentive elements, only 8 threads need to participate,
 * thus splitting the warp into 4x smaller subwarps 8-thread width.
 * 'Butterfly' reduction is used inside subwarps.
 */
template <int subwarp_width>
__forceinline__ __device__ float subwarp_reduce_max_broadcast(const float val) {
  float val_tmp = val;
#pragma unroll
  for (int offset = subwarp_width / 2; offset > 0; offset /= 2) {
    const float val_other = __shfl_down_sync(0xFFFFFFFF, val_tmp, offset, subwarp_width);
    __builtin_assume(val_tmp >= 0);
    __builtin_assume(val_other >= 0);
    val_tmp = fmaxf(val_tmp, val_other);
  }
  // Broadcast the amax to other threads of the subwarp from the zero subwarp lane_id
  constexpr int subwarp_lane_zero = 0;
  val_tmp = __shfl_sync(0xFFFFFFFF, val_tmp, subwarp_lane_zero, subwarp_width);
  return val_tmp;
}

// Works only on positive values
__device__ __forceinline__ void atomicMaxFloat(float *addr, const float value) {
  atomicMax(reinterpret_cast<int *>(addr), __float_as_int(value));
}

// Works only on positive values
__device__ __forceinline__ void atomicMinFloat(float *addr, const float value) {
  atomicMin(reinterpret_cast<int *>(addr), __float_as_int(value));
}

template <typename T>
__device__ __forceinline__ void reciprocal(T *value_inv, const T value) {
  *value_inv = 1 / value;
}

template <>
__device__ __forceinline__ void reciprocal<float>(float *value_inv, const float value) {
  *value_inv = __frcp_rn(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
using e8m0_t = uint8_t;

enum ScalingType { ROWWISE = 0, COLWISE = 1, BIDIMENSIONAL = 2 };

template <typename T>
struct Numeric_Traits;

template <>
struct Numeric_Traits<fp8e4m3> {
  static constexpr int maxUnbiasedExponent = 8;
  static constexpr double maxNorm = 448;
};

template <>
struct Numeric_Traits<fp8e5m2> {
  static constexpr int maxUnbiasedExponent = 15;
  static constexpr double maxNorm = 57344;
};

template <typename T>
struct Quantized_Limits {
  static constexpr int max_unbiased_exponent = Numeric_Traits<T>::maxUnbiasedExponent;
  static constexpr float max_norm = Numeric_Traits<T>::maxNorm;
  static constexpr float max_norm_rcp = 1.0 / max_norm;
  static constexpr float emax = 1 << max_unbiased_exponent;
  static constexpr float emax_rcp = 1.0 / emax;
};

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTILS_CUH_
