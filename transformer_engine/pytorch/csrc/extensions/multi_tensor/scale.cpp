/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, float scale) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  int device_id = tensor_lists[0][0].device().index();

  nvte_multi_tensor_scale_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                               num_tensors, scale, device_id, at::cuda::getCurrentCUDAStream());
}
