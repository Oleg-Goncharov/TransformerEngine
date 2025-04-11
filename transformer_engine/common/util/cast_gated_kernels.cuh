/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_gated_kernels.cuh
 *  \brief CUDA gated activations kernels to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_GATED_KERNELS_CUH_
#define TRANSFORMER_ENGINE_CAST_GATED_KERNELS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include <cfloat>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {

namespace gated_kernels {

constexpr size_t ALIGNMENT_SIZE = 128;
constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 512;
constexpr size_t THREADS_PER_CHUNK_X = CHUNK_DIM_X;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;  // 4 = 512 / 128
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFFER_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                 //   4 = 128 / 32
static_assert(ITERATIONS >= 1);

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_fp8_gated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                          const __grid_constant__ CUtensorMap tensor_map_input_act,
                          const __grid_constant__ CUtensorMap tensor_map_input_gate,
                          const __grid_constant__ CUtensorMap tensor_map_output_act,
                          const __grid_constant__ CUtensorMap tensor_map_output_gate,
                          float *const amax_ptr, float *const scale_inv_ptr,
                          const float *const scale_ptr, const size_t rows, const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  constexpr size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  constexpr size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  constexpr size_t buff_size_aligned_out =
      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

  constexpr size_t grad_mem = IS_DGATED ? buff_size_aligned_in : 0;

  constexpr size_t in_act_mem = buff_size_aligned_in;
  constexpr size_t in_gate_mem = buff_size_aligned_in;
  constexpr size_t in_mem = in_act_mem + in_gate_mem;

  constexpr size_t out_act_mem = buff_size_aligned_out;

  // const size_t in_transaction_size = grad_mem + in_mem;
  constexpr size_t in_transaction_size = buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);
  OType *out_act_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_in_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
  const uint64_t *TMAP_in_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);
  const uint64_t *TMAP_output_act = reinterpret_cast<const uint64_t *>(&tensor_map_output_act);
  const uint64_t *TMAP_output_gate = reinterpret_cast<const uint64_t *>(&tensor_map_output_gate);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  // Prefetch data of the first stage

  if constexpr (IS_DGATED) {
    copy_2d_to_sharedx3(in_grad_sh, TMAP_grad_in, chunk_offset_X, chunk_offset_Y, in_act_sh,
                        TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh, TMAP_in_gate,
                        chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  } else {
    copy_2d_to_sharedx2(in_act_sh, TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh,
                        TMAP_in_gate, chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const int buff = it % BUFFERS_NUM;
    const int next_it = it + 1;
    if (next_it < ITERATIONS) {
      const int next_buff = next_it % BUFFERS_NUM;
      const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      if constexpr (IS_DGATED) {
        copy_2d_to_sharedx3(
            &in_grad_sh[next_buff * buff_elems], TMAP_grad_in, chunk_it_offset_x, chunk_it_offset_y,
            &in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x, chunk_it_offset_y,
            &in_gate_sh[next_buff * buff_elems], TMAP_in_gate, chunk_it_offset_x, chunk_it_offset_y,
            in_transaction_size, &mbar[next_it], is_master_thread);
      } else {
        copy_2d_to_sharedx2(&in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x,
                            chunk_it_offset_y, &in_gate_sh[next_buff * buff_elems], TMAP_in_gate,
                            chunk_it_offset_x, chunk_it_offset_y, in_transaction_size,
                            &mbar[next_it], is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_sh_curr = out_act_sh + buff * buff_elems;
    OType *out_gate_sh_curr = out_gate_sh + buff * buff_elems;

#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);

      if constexpr (IS_DGATED) {
        float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);

        const float x = act_elt;
        float act_x;
        float dact_x;

        if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
          const float s = sigmoidf(x);
          act_x = x * s;
          dact_x = x * s * (1 - s) + s;
        } else {
          act_x = ActOP(x, {});
          dact_x = DActOP(x, {});
        }

        float after_dact = dact_x * grad_elt * gate_elt;
        float after_dgate = act_x * grad_elt;

        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dact);
        out_gate_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dgate);

        amax = fmaxf(amax, fabsf(after_dact));
        amax = fmaxf(amax, fabsf(after_dgate));
      } else {
        const float after_act = ActOP(act_elt, {}) * gate_elt;
        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_act);
        amax = fmaxf(amax, fabsf(after_act));
      }
    }

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      ptx::cp_async_bulk_tensor_2d_shared_to_global(TMAP_output_act, chunk_it_offset_x,
                                                    chunk_it_offset_y,
                                                    reinterpret_cast<uint64_t *>(out_act_sh_curr));

      if constexpr (IS_DGATED) {
        // dGate
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_gate, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_gate_sh_curr));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

namespace mxfp8_kernel {

constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 64;
constexpr size_t THREADS_PER_CHUNK = 64;

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t PACK_SIZE = 4;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

// Number of 1-byte elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4) / 1;  // 128

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM_X;  // 4 = 128 / 32

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr __device__ __forceinline__ bool is_cached_act_op() {
  return (OP == gelu<float, float>) || (OP == dgelu<float, float>) ||
         (OP == sigmoid<float, float>) || (OP == dsigmoid<float, float>) ||
         (OP == qgelu<float, float>) || (OP == dqgelu<float, float>) ||
         (OP == silu<float, float>) || (OP == dsilu<float, float>);
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          bool ROWWISE_SCALING, bool COLWISE_SCALING>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_mxfp8_gated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                            const __grid_constant__ CUtensorMap tensor_map_input_act,
                            const __grid_constant__ CUtensorMap tensor_map_input_gate,
                            const __grid_constant__ CUtensorMap tensor_map_output_act_rowwise,
                            const __grid_constant__ CUtensorMap tensor_map_output_gate_rowwise,
                            const __grid_constant__ CUtensorMap tensor_map_output_act_colwise,
                            const __grid_constant__ CUtensorMap tensor_map_output_gate_colwise,
                            e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                            const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
                            const size_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType2 =
      std::conditional_t<std::is_same_v<IType, float>, float2,
                          std::conditional_t<std::is_same_v<IType, bf16>, ptx::bf16x2, ptx::fp16x2>>;

  using OType2 = std::conditional_t<std::is_same_v<OType, fp8e4m3>, ptx::fp8e4m3x2, ptx::fp8e5m2x2>;
  static_assert(sizeof(OType2) == 2);

  constexpr size_t THREADS_X = CHUNK_DIM_X / SCALE_DIM_X;
  constexpr size_t THREADS_Y = THREADS_PER_CHUNK / THREADS_X;

  constexpr size_t BUFF_DIM_Y = THREADS_Y;
  constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;
  constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
  static_assert(BUFF_DIM_Y == 32);

  constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
  static_assert(STAGES >= 1);

  constexpr bool IS_CACHED_ACT_OP = ROWWISE_SCALING && COLWISE_SCALING && is_cached_act_op<ParamOP, ActOP>();

  const int block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * CHUNK_DIM_X;
  const int scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const int scales_block_offset_X_rowwise = blockIdx.x * CHUNK_DIM_X / SCALE_DIM_X;
  const int scales_block_offset_Y_colwise = blockIdx.y * CHUNK_DIM_Y / SCALE_DIM_Y;
  const int scales_block_offset_X_colwise = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y_rowwise = threadIdx.x / THREADS_X;
  const int tid_X_rowwise = threadIdx.x % THREADS_X;
  const int tid_Y_colwise = 0;
  const int tid_X_colwise = threadIdx.x;

  const int thread_offset_Y_rowwise = tid_Y_rowwise;
  const int thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;
  const int thread_offset_Y_colwise = tid_Y_colwise;
  const int thread_offset_X_colwise = tid_X_colwise;

  const int row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const int col_base_rowwise = block_offset_X + thread_offset_X_rowwise;
  const int row_base_colwise = block_offset_Y + thread_offset_Y_colwise;
  const int col_base_colwise = block_offset_X + thread_offset_X_colwise;

  // const bool col_out_of_bounds_rowwise = (col_base_rowwise >= cols);
  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const int scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const int scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const int scales_offset_Y_colwise = scales_block_offset_Y_colwise + tid_Y_colwise;
  const int scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

  // helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  extern __shared__ __align__(TMA_SHMEM_ALIGNMENT) char dshmem[];

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = (IS_DGATED ? buff_size_aligned_out : 0);
  const size_t out_mem = out_act_mem + out_gate_mem;

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

  OType *out_act_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  OType *out_act_colwise_sh = out_act_rowwise_sh;
  OType *out_gate_colwise_sh = out_gate_rowwise_sh;

  if constexpr (ROWWISE_SCALING && COLWISE_SCALING) {
    out_act_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem);
    out_gate_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem + out_act_mem);
  }

  IType *cached_act_sh = in_act_sh;     // in_act_sh is used as a cache buffer for activations
  IType *cached_gate_sh = in_gate_sh;   // in_gate_sh is used as a cache buffer for gated values

  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  if constexpr (IS_DGATED) {
    copy_2d_to_sharedx3(&in_grad_sh[0], &tensor_map_grad, block_offset_X, block_offset_Y,
                        &in_act_sh[0], &tensor_map_input_act, block_offset_X, block_offset_Y,
                        &in_gate_sh[0], &tensor_map_input_gate, block_offset_X, block_offset_Y,
                        shmem_buff_size, &mbar[0], is_master_thread);
  } else {
    copy_2d_to_sharedx2(&in_act_sh[0], &tensor_map_input_act, block_offset_X, block_offset_Y,
                        &in_gate_sh[0], &tensor_map_input_gate, block_offset_X, block_offset_Y,
                        shmem_buff_size, &mbar[0], is_master_thread);
  }

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const int buff = stage % BUFFS_NUM;
    const int next_stage = stage + 1;
    const int stage_offset_Y = stage * BUFF_DIM_Y;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const int next_buff = next_stage % BUFFS_NUM;
      const int next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const int global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int next_buff_offset = next_buff * BUFF_DIM;
      if constexpr (IS_DGATED) {
        copy_2d_to_sharedx3(&in_grad_sh[next_buff_offset], &tensor_map_grad, global_offset_X, global_offset_Y,
                            &in_act_sh[next_buff_offset], &tensor_map_input_act, global_offset_X, global_offset_Y,
                            &in_gate_sh[next_buff_offset], &tensor_map_input_gate, global_offset_X, global_offset_Y,
                            shmem_buff_size, &mbar[next_stage], is_master_thread);
      } else {
        copy_2d_to_sharedx2(&in_act_sh[next_buff_offset], &tensor_map_input_act, global_offset_X, global_offset_Y,
                            &in_gate_sh[next_buff_offset], &tensor_map_input_gate, global_offset_X, global_offset_Y,
                            shmem_buff_size, &mbar[next_stage], is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], parity);

    if constexpr (COLWISE_SCALING) {
      const int shmem_offset_base_colwise = buff * BUFF_DIM + tid_X_colwise;
      float thread_amax_act = 0.0f;
      float thread_amax_gate = 0.0f;
      float after_act_colwise[BUFF_DIM_Y];
      float after_gate_colwise[BUFF_DIM_Y];

      // 1. Read/Compute elements. Find MXFP8-block AMAX
      #pragma unroll
      for (int i = 0; i < BUFF_DIM_Y; ++i) {
        const int shmem_offset_colwise = shmem_offset_base_colwise + i * BUFF_DIM_X;

        float act_elt = static_cast<float>(in_act_sh[shmem_offset_colwise]);
        float gate_elt = static_cast<float>(in_gate_sh[shmem_offset_colwise]);
        float after_act_elt;
        float after_gate_elt;

        if constexpr (IS_DGATED) {
          float grad_elt = static_cast<float>(in_grad_sh[shmem_offset_colwise]);
          const float x = act_elt;
          float act_x;
          float dact_x;

          if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
            const float s = sigmoidf(x);
            act_x = x * s;
            dact_x = x * s * (1 - s) + s;
          } else {
            act_x = ActOP(x, {});
            dact_x = DActOP(x, {});
          }
          after_act_elt = dact_x * grad_elt * gate_elt;
          after_gate_elt = act_x * grad_elt;
          after_act_colwise[i] = after_act_elt;
          after_gate_colwise[i] = after_gate_elt;
        } else {
          after_act_elt = ActOP(act_elt, {}) * gate_elt;
          after_act_colwise[i] = after_act_elt;
        }

        // Cache computed activations to avoid computing them again in the 2nd pass along another dimension
        if constexpr (IS_CACHED_ACT_OP) {
          cached_act_sh[shmem_offset_colwise] = static_cast<IType>(after_act_elt);
          if constexpr (IS_DGATED) {
            cached_gate_sh[shmem_offset_colwise] = static_cast<IType>(after_gate_elt);
          }
        }

        const bool row_out_of_bounds_colwise = (row_base_colwise + stage_offset_Y + i >= rows);
        const bool out_of_bounds = (col_out_of_bounds_colwise || row_out_of_bounds_colwise);

        if (!out_of_bounds) {
          thread_amax_act = fmaxf(thread_amax_act, fabsf(after_act_elt));
          if constexpr (IS_DGATED) {
            thread_amax_gate = fmaxf(thread_amax_gate, fabsf(after_gate_elt));
          }
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent_act = float_to_e8m0(thread_amax_act * Quantized_Limits<OType>::max_norm_rcp);
      
      const int global_scales_offset_Y = scales_offset_Y_colwise + stage;
      const int global_scales_offset_X = scales_offset_X_colwise;
      const int scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
      scales_colwise[scale_idx] = biased_exponent_act;
      
      float block_scale_inverse_act = exp2f_rcp(biased_exponent_act);
      float block_scale_inverse_gate;
      
      if constexpr (IS_DGATED) {
        const e8m0_t biased_exponent_gate = float_to_e8m0(thread_amax_gate * Quantized_Limits<OType>::max_norm_rcp);
        const int scale_idx_gate = scale_idx + scale_stride_colwise / 2;
        scales_colwise[scale_idx_gate] = biased_exponent_gate;
        block_scale_inverse_gate = exp2f_rcp(biased_exponent_gate);
      }

      // 3. Scale elements
      #pragma unroll
      for (int i = 0; i < SCALE_DIM_Y; ++i) {
        const int shmem_offset_elt = shmem_offset_base_colwise + i * BUFF_DIM_X;
        if constexpr (IS_DGATED) {
          OType2 out_pair;
          float2 in_pair = {after_act_colwise[i], after_gate_colwise[i]};
          const float2 block_scale_inverse_2x_pair = make_float2(block_scale_inverse_act, block_scale_inverse_gate);
          ptx::mul_cvt_2x<OType2, float2>(out_pair, in_pair, block_scale_inverse_2x_pair);
          out_act_colwise_sh[shmem_offset_elt] = out_pair.x;
          out_gate_colwise_sh[shmem_offset_elt] = out_pair.y;
        } else {
          const float scaled_out_act = block_scale_inverse_act * after_act_colwise[i];
          out_act_colwise_sh[shmem_offset_elt] = static_cast<OType>(scaled_out_act);
        }
      }
    }

    if constexpr (ROWWISE_SCALING) {
      const int shmem_offset_base_rowwise = buff * BUFF_DIM + thread_offset_Y_rowwise * BUFF_DIM_X;

      float thread_amax_act = 0.0f;
      float thread_amax_gate = 0.0f;

      Vec<IType, PACK_SIZE> in_cached_act[WAVES];
      Vec<IType, PACK_SIZE> in_cached_gate[WAVES];

      float after_act_rowwise[SCALE_DIM_X];
      float after_gate_rowwise[SCALE_DIM_X];

      // 1. Read/Compute elements. Find MXFP8-block AMAX
      if constexpr (IS_CACHED_ACT_OP) {
        // ensures that all writes to cache made in the section above are visible to all threads
        __syncthreads();
        IType2 thread_amax_2x_act = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
        IType2 thread_amax_2x_gate = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const int shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;

          const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
          const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
          const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);

          // Load cached elements
          in_cached_act[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
          if constexpr (IS_DGATED) {
            in_cached_gate[w].load_from(&cached_gate_sh[shmem_offset_rowwise]);
          }
          // Since TMA requirement for the data alignment is 16B (i.e. cols % 8 == 0, in case of BF16 elements)
          // only single check (w.r.t. column direction) is sufficient to be sure the entire wave is inside the boundaries
          if (!out_of_bounds) {
            if constexpr (std::is_same_v<IType, float>) {
              #pragma unroll
              for (int e = 0; e < PACK_SIZE; ++e) {
                thread_amax_act = fmaxf(thread_amax_act, fabsf(in_cached_act[w].data.elt[e]));
                if constexpr (IS_DGATED) {
                  thread_amax_gate = fmaxf(thread_amax_gate, fabsf(in_cached_gate[w].data.elt[e]));
                }
              }
            } else {
              #pragma unroll
              for (int e = 0; e < PACK_SIZE; e += 2) {
                const IType2 in_cached_2x_act = {in_cached_act[w].data.elt[e],
                                                 in_cached_act[w].data.elt[e + 1]};
                ptx::abs_max_2x<IType2>(thread_amax_2x_act, thread_amax_2x_act, in_cached_2x_act);
                if constexpr (IS_DGATED) {
                  const IType2 in_cached_2x_gate = {in_cached_gate[w].data.elt[e],
                                                    in_cached_gate[w].data.elt[e + 1]};
                  ptx::abs_max_2x<IType2>(thread_amax_2x_gate, thread_amax_2x_gate, in_cached_2x_gate);
                }
              }
            }
          }
        }
        if constexpr (!std::is_same_v<IType, float>) {
          thread_amax_act = static_cast<float>(__hmax(__habs(thread_amax_2x_act.x), __habs(thread_amax_2x_act.y)));
          if constexpr (IS_DGATED) {
            thread_amax_gate = static_cast<float>(__hmax(__habs(thread_amax_2x_gate.x), __habs(thread_amax_2x_gate.y)));
          }
        }
      } else {
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const int shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;

          Vec<IType, PACK_SIZE> in_grad;
          Vec<IType, PACK_SIZE> in_act;
          Vec<IType, PACK_SIZE> in_gate;

          in_act.load_from(&in_act_sh[shmem_offset_rowwise]);
          in_gate.load_from(&in_gate_sh[shmem_offset_rowwise]);
          if constexpr (IS_DGATED) {
            in_grad.load_from(&in_grad_sh[shmem_offset_rowwise]);
          }

          #pragma unroll
          for (int e = 0; e < PACK_SIZE; ++e) {
            const int j = w * PACK_SIZE + e;

            float act_elt = static_cast<float>(in_act.data.elt[e]);
            float gate_elt = static_cast<float>(in_gate.data.elt[e]);
            float after_act_elt;
            float after_gate_elt;

            if constexpr (IS_DGATED) {
              float grad_elt = static_cast<float>(in_grad.data.elt[e]);
              const float x = act_elt;
              float act_x;
              float dact_x;
    
              if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
                const float s = sigmoidf(x);
                act_x = x * s;
                dact_x = x * s * (1 - s) + s;
              } else {
                act_x = ActOP(x, {});
                dact_x = DActOP(x, {});
              }
              after_act_elt = dact_x * grad_elt * gate_elt;
              after_gate_elt = act_x * grad_elt;
              after_act_rowwise[j] = after_act_elt;
              after_gate_rowwise[j] = after_gate_elt;
            } else {
              after_act_elt = ActOP(act_elt, {}) * gate_elt;
              after_act_rowwise[j] = after_act_elt;
            }

            const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
            const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
            const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
            if (!out_of_bounds) {
              thread_amax_act = fmaxf(thread_amax_act, fabsf(after_act_elt));
              if constexpr (IS_DGATED) {
                thread_amax_gate = fmaxf(thread_amax_gate, fabsf(after_gate_elt));
              }
            }
          }
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent_act = float_to_e8m0(thread_amax_act * Quantized_Limits<OType>::max_norm_rcp);
      const int stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
      const int stage_scales_offset_X = scales_offset_X_rowwise;
      const int scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
      scales_rowwise[scale_idx] = biased_exponent_act;
      
      const float block_scale_inverse_act = exp2f_rcp(biased_exponent_act);
      const float2 block_scale_inverse_2x_act = make_float2(block_scale_inverse_act, block_scale_inverse_act);

      float block_scale_inverse_gate;
      float2 block_scale_inverse_2x_gate;
      if constexpr (IS_DGATED) {
        const e8m0_t biased_exponent_gate = float_to_e8m0(thread_amax_gate * Quantized_Limits<OType>::max_norm_rcp);
        const int scale_idx_gate = scale_idx + scale_stride_rowwise / 2;
        scales_rowwise[scale_idx_gate] = biased_exponent_gate;
        block_scale_inverse_gate = exp2f_rcp(biased_exponent_gate);
        block_scale_inverse_2x_gate = make_float2(block_scale_inverse_gate, block_scale_inverse_gate);
      }
      
      // 3. Scale elements
      #pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        Vec<OType2, PACK_SIZE / 2> out_act;
        Vec<OType2, PACK_SIZE / 2> out_gate;
        #pragma unroll
        for (int e = 0; e < PACK_SIZE / 2; ++e) {
          IType2 in_act;
          OType2 &out_act_pair = reinterpret_cast<OType2 &>(out_act.data.elt[e]);

          if constexpr (IS_CACHED_ACT_OP) {
            in_act.x = in_cached_act[w].data.elt[2 * e];
            in_act.y = in_cached_act[w].data.elt[2 * e + 1];
          } else {
            const int j = w * PACK_SIZE + 2 * e;
            in_act.x = after_act_rowwise[j];
            in_act.y = after_act_rowwise[j + 1];
          }
          ptx::mul_cvt_2x<OType2, IType2>(out_act_pair, in_act, block_scale_inverse_2x_act);

          if constexpr (IS_DGATED) {
            IType2 in_gate;
            OType2 &out_gate_pair = reinterpret_cast<OType2 &>(out_gate.data.elt[e]);

            if constexpr (IS_CACHED_ACT_OP) {
              in_gate.x = in_cached_gate[w].data.elt[2 * e];
              in_gate.y = in_cached_gate[w].data.elt[2 * e + 1];
            } else {
              const int j = w * PACK_SIZE + 2 * e;
              in_gate.x = after_gate_rowwise[j];
              in_gate.y = after_gate_rowwise[j + 1];
            }
            ptx::mul_cvt_2x<OType2, IType2>(out_gate_pair, in_gate, block_scale_inverse_2x_gate);
          }
        }
        const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
        const int swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
        const int shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_idx;
        out_act.store_to(&out_act_rowwise_sh[shmem_offset_rowwise]);
        if constexpr (IS_DGATED) {
          out_gate.store_to(&out_gate_rowwise_sh[shmem_offset_rowwise]);
        }
      }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int global_offset_Y = block_offset_Y + stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int buff_offset = buff * BUFF_DIM;

      if constexpr (ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_act_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_act_rowwise_sh[buff_offset]));
        if constexpr (IS_DGATED) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_gate_rowwise_sh[buff_offset]));
        }
      }
      if constexpr (COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_act_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_act_colwise_sh[buff_offset]));
        if constexpr (IS_DGATED) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_gate_colwise_sh[buff_offset]));
        }
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }

  parity ^= 1;
  destroy_barriers<STAGES>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace mxfp8_kernel

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_fp8_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                    cudaStream_t stream) {
  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }

  NVTE_CHECK(!output->has_columnwise_data(), "Only rowwise cast supported in this function.");
  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_input_act{};
          alignas(64) CUtensorMap tensor_map_input_gate{};
          alignas(64) CUtensorMap tensor_map_output_act{};
          alignas(64) CUtensorMap tensor_map_output_gate{};

          if constexpr (IS_DGATED) {
            create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X,
                                 cols, 0, sizeof(IType));
          }

          const uint32_t tensor_stride_elems = output_cols;

          create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, 0, sizeof(IType));
          create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, cols, sizeof(IType));
          create_2D_tensor_map(tensor_map_output_act, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, 0, sizeof(OType));
          create_2D_tensor_map(tensor_map_output_gate, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, cols, sizeof(OType));

          const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
          const size_t buff_size_aligned_in = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE);
          const size_t buff_size_aligned_out = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE);
          const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = buff_size_aligned_out;    // TODO: Check if (IS_DGATED ? buff_size_aligned_out : 0)

          const size_t shmem_size = ALIGNMENT_SIZE + grad_mem + (in_act_mem + in_gate_mem) +
                                    (out_act_mem + out_gate_mem);

          cudaFuncSetAttribute(
              cast_fp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

          cast_fp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType>
          <<<grid_dim, block_dim, shmem_size, stream>>>(
              tensor_map_grad, tensor_map_input_act, tensor_map_input_gate, tensor_map_output_act,
              tensor_map_output_gate, amax_ptr, scale_inv_ptr, scale_ptr, rows,
              cols););  // NOLINT(*)
  );                    // NOLINT(*)
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_mxfp8_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                      cudaStream_t stream) {
  const bool USE_ROWWISE_SCALING = output->has_data();
  const bool USE_COLWISE_SCALING = output->has_columnwise_data();

  if (USE_ROWWISE_SCALING) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (USE_COLWISE_SCALING) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }

  ScalingType scaling_type;
  if (USE_ROWWISE_SCALING && (!USE_COLWISE_SCALING)) {
    scaling_type = ScalingType::ROWWISE;
  } else if ((!USE_ROWWISE_SCALING) && USE_COLWISE_SCALING) {
    scaling_type = ScalingType::COLWISE;
  } else if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
    scaling_type = ScalingType::BIDIMENSIONAL;
  }

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  constexpr size_t THREADS_X = mxfp8_kernel::CHUNK_DIM_X / mxfp8_kernel::SCALE_DIM_X;
  constexpr size_t THREADS_Y = mxfp8_kernel::THREADS_PER_CHUNK / THREADS_X;
  constexpr size_t BUFF_DIM_Y = THREADS_Y;
  constexpr size_t BUFF_DIM_X = mxfp8_kernel::CHUNK_DIM_X;

  const size_t blocks_Y = DIVUP(rows, mxfp8_kernel::CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, mxfp8_kernel::CHUNK_DIM_X);

  const dim3 grid(blocks_X, blocks_Y);
  const dim3 block_size(mxfp8_kernel::THREADS_PER_CHUNK);

  size_t scale_stride_rowwise = USE_ROWWISE_SCALING ? output->scale_inv.shape[1] : 1;
  size_t scale_stride_colwise = USE_COLWISE_SCALING ? output->columnwise_scale_inv.shape[1] : 1;

  e8m0_t *const scales_rowwise_ptr = USE_ROWWISE_SCALING
                                     ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr)
                                     : nullptr;
  e8m0_t *const scales_colwise_ptr = USE_COLWISE_SCALING
                                     ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr)
                                     : nullptr;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_input_act{};
          alignas(64) CUtensorMap tensor_map_input_gate{};
          alignas(64) CUtensorMap tensor_map_output_act_rowwise{};
          alignas(64) CUtensorMap tensor_map_output_gate_rowwise{};
          alignas(64) CUtensorMap tensor_map_output_act_colwise{};
          alignas(64) CUtensorMap tensor_map_output_gate_colwise{};

          if constexpr (IS_DGATED) {
            create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, cols, 0, sizeof(IType));
          }

          const uint32_t tensor_stride_elems = output_cols;
          create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols,
                               BUFF_DIM_Y, BUFF_DIM_X, cols * 2, 0, sizeof(IType));
          create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols,
                               BUFF_DIM_Y, BUFF_DIM_X, cols * 2, cols, sizeof(IType));

          if (USE_ROWWISE_SCALING) {
            create_2D_tensor_map(tensor_map_output_act_rowwise, output->data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, 0, sizeof(OType));
            create_2D_tensor_map(tensor_map_output_gate_rowwise, output->data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, cols, sizeof(OType));
          }

          if (USE_COLWISE_SCALING) {
            create_2D_tensor_map(tensor_map_output_act_colwise, output->columnwise_data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, 0, sizeof(OType));
            create_2D_tensor_map(tensor_map_output_gate_colwise, output->columnwise_data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, tensor_stride_elems, cols, sizeof(OType));
          }

          const size_t buff_elems_total = BUFFERS_NUM * BUFF_DIM_Y * BUFF_DIM_X;
          const size_t buff_size_aligned_in = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE);
          const size_t buff_size_aligned_out = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE);

          const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t in_mem = grad_mem + in_act_mem + in_gate_mem;

          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = (IS_DGATED ? buff_size_aligned_out : 0);
          size_t out_mem = out_act_mem + out_gate_mem;
          if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
            out_mem *= 2;
          }

          const size_t shmem_size = in_mem + out_mem;

          switch (scaling_type) {
            case ScalingType::ROWWISE:
              cudaFuncSetAttribute(
                  mxfp8_kernel::cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType, true, false>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    
              mxfp8_kernel::cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType, true, false>
              <<<grid, block_size, shmem_size, stream>>>(
                  tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                  tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                  tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                  scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                  scale_stride_colwise);
              break;
            case ScalingType::COLWISE:
              cudaFuncSetAttribute(
                  mxfp8_kernel::cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType, false, true>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

              mxfp8_kernel::cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType, false, true>
              <<<grid, block_size, shmem_size, stream>>>(
                  tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                  tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                  tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                  scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                  scale_stride_colwise);
              break;
            case ScalingType::BIDIMENSIONAL:
              cudaFuncSetAttribute(
                  mxfp8_kernel::cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType, true, true>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

              mxfp8_kernel::cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType, true, true>
              <<<grid, block_size, shmem_size, stream>>>(
                  tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                  tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                  tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                  scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                  scale_stride_colwise);
              break;
          }
      );  // NOLINT(*)
  );  // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void cast_gated(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "gated_act_input");
  CheckOutputTensor(*output, "gated_act_output");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape[0] == output->data.shape[0],
             "Input shape[0] must be equal to output shape[0].");
  NVTE_CHECK(input.data.shape[1] == output->data.shape[1] * 2,
             "Input shape[1] must be 2x larger than output shape[1].");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,

          if (!is_fp8_dtype(output->data.dtype) ||
              is_delayed_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            GatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), output->data.shape[0],
                output->data.shape[1], {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_dgated(const Tensor &grad, const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(grad, "dgated_act_grad");
  CheckInputTensor(input, "dgated_act_input");
  CheckOutputTensor(*output, "dgated_act_output");
  NVTE_CHECK(output->flat_first_dim() == grad.flat_first_dim(),
             "Wrong output shape. Expected (after flattening) [", grad.flat_first_dim(),
             ", *], got [", output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(output->flat_last_dim() == grad.flat_last_dim() * 2,
             "Wrong output shape. Expected (after flattening) [*, ", grad.flat_last_dim() * 2,
             "], got [", output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(input.data.shape == output->data.shape,
             "Input and output shapes must match. Input shape: ", input.data.shape,
             ", output shape: ", output->data.shape, ".");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          if (!is_fp8_dtype(output->data.dtype) ||
              is_delayed_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            DGatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP, DActOP>(
                reinterpret_cast<const IType *>(grad.data.dptr),
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), grad.flat_first_dim(),
                grad.flat_last_dim(), {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                    cudaStream_t stream) {
  checkCuDriverContext(stream);
  constexpr bool allow_empty = false;
  CheckInputTensor(gated_input, "gated_input");
  CheckOutputTensor(*output, "output", allow_empty);

  NVTE_CHECK(gated_input.flat_last_dim() % 2 == 0, "Number of columns must be even.");

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  if constexpr (IS_DGATED) {
    CheckInputTensor(grad, "grad");
    NVTE_CHECK(!is_fp8_dtype(grad.data.dtype), "Grad input must be in higher precision.");
    NVTE_CHECK(grad.data.dtype == gated_input.data.dtype, "Types of both inputs must match.");
    NVTE_CHECK(grad.flat_first_dim() == rows, "Wrong dimension of the grad input.");
    NVTE_CHECK(grad.flat_last_dim() == cols, "Wrong dimension of the grad input.");
  }

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  bool is_fp8_rowwise_output = true;
  bool is_fp8_colwise_output = true;
  if (output->has_data()) {
    is_fp8_rowwise_output = is_fp8_dtype(output->data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == output_cols, "Wrong dimension of the output.");
  }
  if (output->has_columnwise_data()) {
    is_fp8_colwise_output = is_fp8_dtype(output->columnwise_data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == output_cols, "Wrong dimension of the output.");
  }

  const bool use_tma_kernels = is_fp8_rowwise_output && is_fp8_colwise_output && cols % 32 == 0;

  if (is_delayed_tensor_scaling(output->scaling_mode)) {
    if (use_tma_kernels) {
      cast_fp8_gated<IS_DGATED, ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
    } else {
      if constexpr (IS_DGATED) {
        cast_dgated<ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
      } else {
        cast_gated<ParamOP, ActOP>(gated_input, output, stream);
      }
    }
  } else if (is_mxfp_scaling(output->scaling_mode)) {
    if (use_tma_kernels) {
      cast_mxfp8_gated<IS_DGATED, ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
    } else {
      NVTE_ERROR("Invalid input shape. Expected the last dimension to be divisible ",
                 "by 32, got input of shape ", gated_input.data.shape);
    }
  } else {
    NVTE_ERROR("Not supported scaling mode");
  }
}
}  // namespace gated_kernels

namespace detail {

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated_helper(const NVTETensor grad, const NVTETensor gated_input, NVTETensor output,
                           cudaStream_t stream) {
  using namespace gated_kernels;
  Tensor grad_empty_tensor;
  const Tensor &grad_tensor =
      IS_DGATED ? *(reinterpret_cast<const Tensor *>(grad)) : grad_empty_tensor;
  const Tensor gated_input_tensor = *reinterpret_cast<const Tensor *>(gated_input);
  Tensor *output_tensor = reinterpret_cast<Tensor *>(output);

  if (is_supported_by_CC_100()) {
    quantize_gated<IS_DGATED, ParamOP, ActOP, DActOP>(grad_tensor, gated_input_tensor,
                                                      output_tensor, stream);
  } else {
    // if (is_delayed_tensor_scaling(output_tensor->scaling_mode)) {
    //   if constexpr (IS_DGATED) {
    //     cast_dgated<ParamOP, ActOP, DActOP>(grad_tensor, gated_input_tensor, output_tensor, stream);
    //   } else {
    //     cast_gated<ParamOP, ActOP>(gated_input_tensor, output_tensor, stream);
    //   }
    // } else {
    //   // MX scaling
    //   NVTE_ERROR("Not supported by the Arch < 10.0");
    // }
  }
}
}  // namespace detail

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_GATED_KERNELS_CUH_
