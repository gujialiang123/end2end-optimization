
# AOT ID: ['9_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/4m/c4meazuvjmsrgrjprkncimzx3wuk6mwl64aslb4ush4sbtuejnbu.py
# Topologically Sorted Source Nodes: [chunk, Bx, transpose, Bx_t], Original ATen: [aten.split, aten.mul, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   Bx => mul
#   Bx_t => clone
#   chunk => split
#   transpose => permute_2
# Graph fragment:
#   %mm : Tensor "bf16[512, 6144][6144, 1]cuda:0" = PlaceHolder[target=mm]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%mm, 2048, -1), kwargs = {})
#   %mul : Tensor "bf16[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_2, %getitem_4), kwargs = {})
#   %permute_2 : Tensor "bf16[2048, 512][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [1, 0]), kwargs = {})
#   %clone : Tensor "bf16[2048, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_mul_split_transpose_0 = async_compile.triton('triton_poi_fused_clone_mul_split_transpose_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 2048}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_split_transpose_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4194304, 'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_split_transpose_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6144*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (4096 + x1 + 6144*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (y0 + 512*x1), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/cq/ccqkljxylknzjy3az767fcqt4eydr22qmqahbjrhsgl5vspvojti.py
# Topologically Sorted Source Nodes: [chunk, mul_1], Original ATen: [aten.split, aten.transpose, aten.mul]
# Source node to ATen node mapping:
#   chunk => split
#   mul_1 => mul_1, permute_4
# Graph fragment:
#   %mm : Tensor "bf16[512, 6144][6144, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf6 : Tensor  = PlaceHolder[target=buf6]
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%mm, 2048, -1), kwargs = {})
#   %permute_4 : Tensor "bf16[512, 2048][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%clone, [1, 0]), kwargs = {})
#   %mul_1 : Tensor "bf16[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_3, %permute_4), kwargs = {})
#   return %mul_1
triton_poi_fused_mul_split_transpose_1 = async_compile.triton('triton_poi_fused_mul_split_transpose_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_split_transpose_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_split_transpose_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + 6144*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + 512*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/f7/cf7rfybsdcx3z3a2hsidropnjp5nixxfjwpfspvzvpnlqcd4tphd.py
# Topologically Sorted Source Nodes: [float_1, scores, scores_for_choice], Original ATen: [aten._to_copy, aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_6
#   scores => sigmoid
#   scores_for_choice => add_1
# Graph fragment:
#   %mm_2 : Tensor "bf16[512, 32][32, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %arg10_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_6 : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %add_1 : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, %arg10_1), kwargs = {})
#   return %add_1
triton_poi_fused__to_copy_add_sigmoid_2 = async_compile.triton('triton_poi_fused__to_copy_add_sigmoid_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_sigmoid_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 163968}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_sigmoid_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/64/c64zxlhgmiqzsjpmqwccgztuprjlnzrqzf5sucjni46bbojdujey.py
# Topologically Sorted Source Nodes: [getitem_9, getitem_10], Original ATen: [aten.select, aten.index]
# Source node to ATen node mapping:
#   getitem_10 => index
#   getitem_9 => select
# Graph fragment:
#   %getitem_12 : Tensor "i64[512, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %arg11_1 : Tensor "bf16[32, 3584, 2048][7340032, 2048, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %select : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_12, 0, 0), kwargs = {})
#   %index : Tensor "bf16[4, 3584, 2048][7340032, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg11_1, [%select]), kwargs = {})
#   return %index
triton_poi_fused_index_select_3 = async_compile.triton('triton_poi_fused_index_select_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29360128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 7340032
    x0 = (xindex % 7340032)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (x0 + 7340032*tmp4), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/55/c55ozntagownyhtywkk4jicg42oweujdkepkufp6unecz3vorjje.py
# Topologically Sorted Source Nodes: [gate_up, chunk_1, silu, act], Original ATen: [aten.bmm, aten.view, aten.permute, aten.split, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   act => mul_3
#   chunk_1 => split_1
#   gate_up => permute_14, unsqueeze_default, view_3, view_4
#   silu => convert_element_type_10, convert_element_type_11, mul_2, sigmoid_1
# Graph fragment:
#   %mm_default : Tensor "bf16[512, 14336][14336, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "bf16[1, 512, 14336][7340032, 14336, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %view_3 : Tensor "bf16[512, 1, 4, 3584][14336, 14336, 3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%unsqueeze_default, [512, 1, 4, 3584]), kwargs = {})
#   %permute_14 : Tensor "bf16[512, 4, 3584, 1][14336, 3584, 1, 14336]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 3, 1]), kwargs = {})
#   %view_4 : Tensor "bf16[512, 4, 3584][14336, 3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_14, [512, 4, 3584]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_4, 1792, -1), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[512, 4, 1792][7168, 1792, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_13, torch.float32), kwargs = {})
#   %sigmoid_1 : Tensor "f32[512, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %mul_2 : Tensor "f32[512, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_10, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[512, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : Tensor "bf16[512, 4, 1792][7168, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, %getitem_14), kwargs = {})
#   return %mul_3
triton_poi_fused_bmm_mul_permute_silu_split_view_4 = async_compile.triton('triton_poi_fused_bmm_mul_permute_silu_split_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_mul_permute_silu_split_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 29360128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_mul_permute_silu_split_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3670016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1792)
    x1 = xindex // 1792
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3584*x1), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (1792 + x0 + 3584*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/c4/cc4imdm3y64pyg4vbtsjoyi56are26pud5hqgykdkkjluyja5iac.py
# Topologically Sorted Source Nodes: [getitem_13, getitem_14], Original ATen: [aten.select, aten.index]
# Source node to ATen node mapping:
#   getitem_13 => select_1
#   getitem_14 => index_1
# Graph fragment:
#   %getitem_12 : Tensor "i64[512, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %arg12_1 : Tensor "bf16[32, 2048, 1792][3670016, 1792, 1]cuda:0" = PlaceHolder[target=arg12_1]
#   %select_1 : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_12, 0, 0), kwargs = {})
#   %index_1 : Tensor "bf16[4, 2048, 1792][3670016, 1792, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg12_1, [%select_1]), kwargs = {})
#   return %index_1
triton_poi_fused_index_select_5 = async_compile.triton('triton_poi_fused_index_select_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14680064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 3670016
    x0 = (xindex % 3670016)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (x0 + 3670016*tmp4), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/jd/cjddy6c2ay4i46me2gtxvejhy5ohk3a444gseomq4yqf7bwvfp5d.py
# Topologically Sorted Source Nodes: [float_1, scores, topk_w, sum_1], Original ATen: [aten._to_copy, aten.sigmoid, aten.gather, aten.sum]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_6
#   scores => sigmoid
#   sum_1 => sum_1
#   topk_w => gather
# Graph fragment:
#   %getitem_12 : Tensor "i64[512, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %mm_2 : Tensor "bf16[512, 32][32, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %convert_element_type_6 : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %gather : Tensor "f32[512, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_12), kwargs = {})
#   %sum_1 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gather, [-1], True), kwargs = {})
#   return %sum_1
triton_poi_fused__to_copy_gather_sigmoid_sum_6 = async_compile.triton('triton_poi_fused__to_copy_gather_sigmoid_sum_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_gather_sigmoid_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_gather_sigmoid_sum_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (tmp4 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.sigmoid(tmp7)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert(((0 <= tmp12) & (tmp12 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp12 < 32")
    tmp14 = tl.load(in_ptr1 + (tmp12 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp8 + tmp16
    tmp19 = tmp18 + tmp1
    tmp20 = tmp18 < 0
    tmp21 = tl.where(tmp20, tmp19, tmp18)
    tl.device_assert(((0 <= tmp21) & (tmp21 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp21 < 32")
    tmp23 = tl.load(in_ptr1 + (tmp21 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp17 + tmp25
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tl.device_assert(((0 <= tmp30) & (tmp30 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp30 < 32")
    tmp32 = tl.load(in_ptr1 + (tmp30 + 32*x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.sigmoid(tmp33)
    tmp35 = tmp26 + tmp34
    tl.store(out_ptr0 + (x0), tmp35, xmask)
''', device_str='cuda')


# kernel path: /home/t-jialianggu/work/EndtoEnd-auto-optimization/results/lfm_fusion/fx/inductor_cache/po/cpouq4jxxai27wdrufkfufr534wqdon4avy54dqpyn7czja3dejp.py
# Topologically Sorted Source Nodes: [float_1, scores, per_expert, topk_w, sum_1, topk_w_1, topk_w_2, unsqueeze, mul_3, out_2, out_3, hidden_states_2], Original ATen: [aten._to_copy, aten.sigmoid, aten.view, aten.permute, aten.gather, aten.sum, aten.div, aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_6
#   hidden_states_2 => add_2
#   mul_3 => mul_4
#   out_2 => sum_2
#   out_3 => mul_5
#   per_expert => permute_19, view_7, view_8
#   scores => sigmoid
#   sum_1 => sum_1
#   topk_w => gather
#   topk_w_1 => div
#   topk_w_2 => convert_element_type_7
#   unsqueeze => unsqueeze_7
# Graph fragment:
#   %bmm_1 : Tensor "bf16[4, 512, 2048][1048576, 2048, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %getitem_12 : Tensor "i64[512, 4][4, 1]cuda:0" = PlaceHolder[target=getitem_12]
#   %mm_2 : Tensor "bf16[512, 32][32, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %sum_1 : Tensor "f32[512, 1][1, 512]cuda:0" = PlaceHolder[target=sum_1]
#   %addmm_default : Tensor "bf16[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=addmm_default]
#   %sum_2 : Tensor "bf16[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %convert_element_type_6 : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[512, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %view_7 : Tensor "bf16[4, 512, 1, 2048][1048576, 2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [4, 512, 1, 2048]), kwargs = {})
#   %permute_19 : Tensor "bf16[512, 4, 2048, 1][2048, 1048576, 1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 3, 2]), kwargs = {})
#   %view_8 : Tensor "bf16[512, 4, 2048][2048, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_19, [512, 4, 2048]), kwargs = {})
#   %gather : Tensor "f32[512, 4][4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_12), kwargs = {})
#   %sum_1 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gather, [-1], True), kwargs = {})
#   %div : Tensor "f32[512, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather, %sum_1), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[512, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[512, 4, 1][4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_7, -1), kwargs = {})
#   %mul_4 : Tensor "bf16[512, 4, 2048][2048, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %unsqueeze_7), kwargs = {})
#   %sum_2 : Tensor "bf16[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1]), kwargs = {})
#   %mul_5 : Tensor "bf16[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 1.0), kwargs = {})
#   %add_2 : Tensor "bf16[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%addmm_default, %mul_5), kwargs = {})
#   return %sum_2,%add_2
triton_poi_fused__to_copy_add_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_7 = async_compile.triton('triton_poi_fused__to_copy_add_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (4*x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1048576 + x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (2097152 + x2), None).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (3145728 + x2), None).to(tl.float32)
    tmp41 = tl.load(in_ptr1 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.full([XBLOCK], 32, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 32), "index out of bounds: 0 <= tmp5 < 32")
    tmp7 = tl.load(in_ptr2 + (tmp5 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.sigmoid(tmp8)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp0 * tmp12
    tmp16 = tmp15 + tmp2
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tl.device_assert((0 <= tmp18) & (tmp18 < 32), "index out of bounds: 0 <= tmp18 < 32")
    tmp20 = tl.load(in_ptr2 + (tmp18 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = (tmp22 / tmp10)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp13 + tmp25
    tmp29 = tmp28 + tmp2
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tl.device_assert((0 <= tmp31) & (tmp31 < 32), "index out of bounds: 0 <= tmp31 < 32")
    tmp33 = tl.load(in_ptr2 + (tmp31 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = (tmp35 / tmp10)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp27 * tmp37
    tmp39 = tmp26 + tmp38
    tmp42 = tmp41 + tmp2
    tmp43 = tmp41 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp41)
    tl.device_assert((0 <= tmp44) & (tmp44 < 32), "index out of bounds: 0 <= tmp44 < 32")
    tmp46 = tl.load(in_ptr2 + (tmp44 + 32*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.sigmoid(tmp47)
    tmp49 = (tmp48 / tmp10)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp40 * tmp50
    tmp52 = tmp39 + tmp51
    tmp54 = 1.0
    tmp55 = tmp52 * tmp54
    tmp56 = tmp53 + tmp55
    tl.store(in_out_ptr0 + (x2), tmp56, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        assert_size_stride(arg0_1, (512, 2048), (2048, 1))
        assert_size_stride(arg1_1, (2048, ), (1, ))
        assert_size_stride(arg2_1, (6144, 2048), (2048, 1))
        assert_size_stride(arg3_1, (2048, 3), (3, 1))
        assert_size_stride(arg4_1, (64, 2048, 3), (6144, 3, 1))
        assert_size_stride(arg5_1, (2, ), (1, ))
        assert_size_stride(arg6_1, (1, ), (1, ))
        assert_size_stride(arg7_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg8_1, (2048, ), (1, ))
        assert_size_stride(arg9_1, (32, 2048), (2048, 1))
        assert_size_stride(arg10_1, (32, ), (1, ))
        assert_size_stride(arg11_1, (32, 3584, 2048), (7340032, 2048, 1))
        assert_size_stride(arg12_1, (32, 2048, 1792), (3670016, 1792, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out, rmsnorm_default], Original ATen: [aten.empty_like, sgl_kernel.rmsnorm]
            torch.ops.sgl_kernel.rmsnorm.default(buf0, arg0_1, arg1_1, 1e-05, True)
            del arg1_1
            buf3 = empty_strided_cuda((512, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out, proj], Original ATen: [aten.empty_like, aten.t, aten.mm]
            extern_kernels.mm(buf0, reinterpret_tensor(arg2_1, (2048, 6144), (1, 2048), 0), out=buf3)
            del arg2_1
            buf4 = reinterpret_tensor(buf0, (2048, 512), (512, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [chunk, Bx, transpose, Bx_t], Original ATen: [aten.split, aten.mul, aten.transpose, aten.clone]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_mul_split_transpose_0.run(buf3, buf4, 512, 2048, stream=stream0)
            # Topologically Sorted Source Nodes: [causal_conv1d_fwd], Original ATen: [sgl_kernel.causal_conv1d_fwd]
            torch.ops.sgl_kernel.causal_conv1d_fwd.default(buf4, arg3_1, None, arg4_1, arg5_1, arg6_1, None, False, -1)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf9 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, mul_1], Original ATen: [aten.split, aten.transpose, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_split_transpose_1.run(buf3, buf4, buf9, 1048576, stream=stream0)
            del buf3
            buf10 = reinterpret_tensor(buf4, (512, 2048), (2048, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [chunk, mul_1, hidden_states], Original ATen: [aten.split, aten.transpose, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg0_1, buf9, reinterpret_tensor(arg7_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf10)
            del arg0_1
            del arg7_1
            buf11 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [out_1, rmsnorm_default_1], Original ATen: [aten.empty_like, sgl_kernel.rmsnorm]
            torch.ops.sgl_kernel.rmsnorm.default(buf11, buf10, arg8_1, 1e-05, True)
            del arg8_1
            buf14 = empty_strided_cuda((512, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_1, router_logits], Original ATen: [aten.empty_like, aten.t, aten.mm]
            extern_kernels.mm(buf11, reinterpret_tensor(arg9_1, (2048, 32), (1, 2048), 0), out=buf14)
            del arg9_1
            buf15 = empty_strided_cuda((512, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, scores, scores_for_choice], Original ATen: [aten._to_copy, aten.sigmoid, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_sigmoid_2.run(buf14, arg10_1, buf15, 16384, stream=stream0)
            del arg10_1
            # Topologically Sorted Source Nodes: [float_1, scores, scores_for_choice, topk], Original ATen: [aten._to_copy, aten.sigmoid, aten.add, aten.topk]
            buf16 = torch.ops.aten.topk.default(buf15, 4)
            del buf15
            buf18 = buf16[1]
            assert_size_stride(buf18, (512, 4), (4, 1), 'torch.ops.aten.topk.default')
            assert_alignment(buf18, 16, 'torch.ops.aten.topk.default')
            del buf16
            buf19 = empty_strided_cuda((4, 3584, 2048), (7340032, 2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_9, getitem_10], Original ATen: [aten.select, aten.index]
            stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_3.run(buf18, arg11_1, buf19, 29360128, stream=stream0)
            del arg11_1
            buf20 = empty_strided_cuda((512, 14336), (14336, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_1, gate_up, getitem_9, getitem_10], Original ATen: [aten.empty_like, aten.unsqueeze, aten.view, aten.bmm, aten.select, aten.index, aten.permute]
            extern_kernels.mm(buf11, reinterpret_tensor(buf19, (2048, 14336), (1, 2048), 0), out=buf20)
            del buf11
            del buf19
            buf21 = empty_strided_cuda((512, 4, 1792), (7168, 1792, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_up, chunk_1, silu, act], Original ATen: [aten.bmm, aten.view, aten.permute, aten.split, aten.silu, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_bmm_mul_permute_silu_split_view_4.run(buf20, buf21, 3670016, stream=stream0)
            del buf20
            buf22 = empty_strided_cuda((4, 2048, 1792), (3670016, 1792, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_13, getitem_14], Original ATen: [aten.select, aten.index]
            stream0 = get_raw_stream(0)
            triton_poi_fused_index_select_5.run(buf18, arg12_1, buf22, 14680064, stream=stream0)
            del arg12_1
            buf23 = empty_strided_cuda((4, 512, 2048), (1048576, 2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [gate_up, chunk_1, silu, act, per_expert, getitem_13, getitem_14], Original ATen: [aten.bmm, aten.view, aten.permute, aten.split, aten.silu, aten.mul, aten.unsqueeze, aten.select, aten.index]
            extern_kernels.bmm(reinterpret_tensor(buf21, (4, 512, 1792), (1792, 7168, 1), 0), reinterpret_tensor(buf22, (4, 1792, 2048), (3670016, 1, 1792), 0), out=buf23)
            del buf21
            del buf22
            buf24 = empty_strided_cuda((512, 1), (1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, scores, topk_w, sum_1], Original ATen: [aten._to_copy, aten.sigmoid, aten.gather, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_gather_sigmoid_sum_6.run(buf18, buf14, buf24, 512, stream=stream0)
            buf26 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [float_1, scores, per_expert, topk_w, sum_1, topk_w_1, topk_w_2, unsqueeze, mul_3, out_2, out_3, hidden_states_2], Original ATen: [aten._to_copy, aten.sigmoid, aten.view, aten.permute, aten.gather, aten.sum, aten.div, aten.unsqueeze, aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_gather_mul_permute_sigmoid_sum_unsqueeze_view_7.run(buf26, buf23, buf18, buf14, buf24, 1048576, stream=stream0)
            del buf14
            del buf18
            del buf23
            del buf24
        return (buf26, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2048, 3), (3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((64, 2048, 3), (6144, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg6_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg7_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, 3584, 2048), (7340032, 2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((32, 2048, 1792), (3670016, 1792, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
