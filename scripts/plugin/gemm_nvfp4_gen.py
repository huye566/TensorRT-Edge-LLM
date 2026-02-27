import os
import torch
import struct
import functools
import numpy as np
from typing import Tuple

RAND_SEED = 2026
DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 64), (128, 128), (256, 64), (256, 128)]
PAD_SHAPES = [
    (90, 64),
    (150, 64),
    (128, 48),
    (128, 80),
    (150, 80),
    (90, 48),
    (90, 128),
    (150, 128),
    (150, 48),
    (90, 80),
]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0, # 0.0?
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
BLOCK_SIZE = 16

def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign

def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        # return 0.0 if x == 0 else 1.0 / x
        return (1 / 1e-8) if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_nvfp4_quant(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // BLOCK_SIZE, BLOCK_SIZE))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    # 从计算逻辑来看，scale 应该不会超出448, scale = FLOAT8_E4M3_MAX * vec_max / vall_max
    # scale = torch.clamp(scale, -FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)  # 超出448了，导致nan
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, k):
    '''
      block-wise FP4 量化的 scale 转换
      1.每个 scale 对应 16 列(CVT_FP4_SF_VEC_SIZE = 16)
      2.每 16 列共享一个 scale
      3.行方向每 128 行(32x4)为一个 tile
      4.列方向每 64 列(16x4)为一个 tile
    '''
    # m_tiles = (m + 128 - 1) // 128
    # f = BLOCK_SIZE * 4
    # k_tiles = (k + f - 1) // f
    # tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    # tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    # out = tmp.reshape(m_tiles * 128, k_tiles * f // BLOCK_SIZE)
    # return out[0:m, 0:k]

    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_k = k // BLOCK_SIZE
    rounded_k = ((scale_k + 4 - 1) // 4) * 4
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_k // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_k)).to(torch.float32)
    return result[:m, :rounded_k]


def convert_swizzled_to_linear(a_sf_swizzled, m, k):
    '''
      block-wise FP4 量化的 scale 转换
      1.每个 scale 对应 16 列(CVT_FP4_SF_VEC_SIZE = 16)
      2.每 16 列共享一个 scale
      3.行方向每 128 行(32x4)为一个 tile
      4.列方向每 64 列(16x4)为一个 tile
    '''
    m_tiles = (m + 128 - 1) // 128
    f = BLOCK_SIZE * 4
    k_tiles = (k + f - 1) // f
    
    # 计算所需的 rounded 尺寸
    rounded_m = m_tiles * 128
    rounded_n = k_tiles * f // BLOCK_SIZE
    
    # 检查输入是否需要padding
    if a_sf_swizzled.shape != (rounded_m, rounded_n):
        # 对输入进行padding到合适的尺寸
        padded_input = torch.zeros((rounded_m, rounded_n), 
                                   dtype=a_sf_swizzled.dtype, 
                                   device=a_sf_swizzled.device)
        # 复制有效数据
        actual_m = min(a_sf_swizzled.shape[0], rounded_m)
        actual_n = min(a_sf_swizzled.shape[1], rounded_n)
        padded_input[:actual_m, :actual_n] = a_sf_swizzled[:actual_m, :actual_n]
        a_sf_swizzled = padded_input
    
    # 按照原来的逻辑进行转换
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // BLOCK_SIZE)
    
    # 输出时裁剪到指定的m,k大小
    return out[0:m, 0:k]

def convert_linear_to_swizzled(a_sf_linear, m, n):
    scale_n = n // BLOCK_SIZE
    rounded_m = ((m + 128 - 1) // 128) * 128
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    
    if a_sf_linear.shape != (rounded_m, rounded_n):
        padded = torch.zeros((rounded_m, rounded_n), dtype=a_sf_linear.dtype, device=a_sf_linear.device)
        padded[:m, :scale_n] = a_sf_linear[:m, :scale_n] if len(a_sf_linear.shape) == 2 else a_sf_linear
    else:
        padded = a_sf_linear
    
    tmp = torch.reshape(padded, (1, rounded_m // 128, 4, 32, rounded_n // 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    
    result = torch.reshape(tmp, (rounded_m, rounded_n))
    return result

def ref_scaled_fp4_quant(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    out_x, scale_linear = ref_nvfp4_quant(x, global_scale)
    scale_swizzled = convert_linear_to_swizzled(scale_linear, m, n)
    return out_x, scale_swizzled.to(torch.float8_e4m3fn)


def test_quantize_to_fp4(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    torch.manual_seed(RAND_SEED)
    torch.set_default_device("cuda:0")

    m, n = shape

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / tensor_amax
    out_ref, scale_linear = ref_nvfp4_quant(x, global_scale)
    print(out_ref.shape, scale_linear.shape)

    scale_swizzle = convert_linear_to_swizzled(scale_linear, m, n)
    scale_linear_check = recover_swizzled_scales(scale_swizzle, m, n)
    scale_linear_pad = convert_swizzled_to_linear(scale_swizzle, m, n)
    print(scale_swizzle.shape, scale_linear_check.shape, scale_linear_pad.shape)
    
    torch.testing.assert_close(scale_linear, scale_linear_check)
    torch.testing.assert_close(scale_linear, scale_linear_pad[:scale_linear.shape[0], :scale_linear.shape[1]])

def cuda_timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()  # 等待所有流完成
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"{func.__name__} 执行耗时: {elapsed_ms:.2f} ms")
        return result
    return wrapper

@cuda_timeit
def break_fp4_bytes(a):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    fH = torch.tensor([E2M1_TO_FLOAT32[x] for x in highHalfByte]).to(a.device)
    fL = torch.tensor([E2M1_TO_FLOAT32[x] for x in lowHalfByte]).to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC], 0xCDAB
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out

@cuda_timeit
def break_fp4_bytes_v2(a):
    assert a.dtype == torch.uint8
    m, n = a.shape

    lut = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=a.device)
    
    # 提取高4位和低4位（注意：a >> 4 已得到高4位数值，无需再 & 0x0F）
    low = a & 0x0F               # 低4位，范围 0-15
    high = a >> 4                # 高4位，范围 0-15

    fL = lut[low.long()]         # [m, n]
    fH = lut[high.long()]        # [m, n]
    
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out

def pack_fp4_bytes_v1(a):
    assert a.dtype == torch.float32
    m, n2 = a.shape
    assert n2 % 2 == 0
    n = n2 // 2
    
    # [m, n*2] -> [m*n, 2]
    a = a.reshape(-1, 2)
    _E2M1_VALUES = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=a.device)
    
    # low 和 high 分别量化
    low = a[:, 0].unsqueeze(-1)   # [m*n, 1]
    high = a[:, 1].unsqueeze(-1)  # [m*n, 1]
    
    low_codes = torch.argmin(torch.abs(low - _E2M1_VALUES), dim=-1).to(torch.uint8)
    high_codes = torch.argmin(torch.abs(high - _E2M1_VALUES), dim=-1).to(torch.uint8)
    
    packed = (high_codes << 4) | low_codes
    return packed.reshape(m, n)

def pack_fp4_bytes(a):
    assert a.dtype == torch.float32
    m, n2 = a.shape
    assert n2 % 2 == 0
    n = n2 // 2
    
    # [m, n*2] -> [m*n, 2]
    a = a.reshape(-1, 2)
    
    low = a[:, 0]  # [m*n]
    high = a[:, 1] # [m*n]
    
    # 预计算所有可能的值
    values_list = E2M1_TO_FLOAT32.copy()
    values_list[8] = -0.0
    
    def quantize_channel(values):
        codes = torch.zeros(values.shape[0], dtype=torch.uint8, device=values.device)
        
        # 检查负零（索引8）
        neg_zero_mask = torch.isclose(values, torch.tensor(0.0, device=values.device)) & \
                        torch.signbit(values)
        codes[neg_zero_mask] = 8
        
        # 检查正零（索引0）
        pos_zero_mask = torch.isclose(values, torch.tensor(0.0, device=values.device)) & \
                       ~torch.signbit(values)
        codes[pos_zero_mask] = 0
        
        other_mask = ~(pos_zero_mask | neg_zero_mask)
        if other_mask.any():
            other_values = values[other_mask]
            for i, target_val in enumerate(values_list):
                if i == 0 or i == 8:  # 跳过两个零
                    continue
                mask_i = torch.isclose(other_values, torch.tensor(target_val, device=values.device))
                if mask_i.any():
                    # 找到这些值在原始数组中的索引
                    orig_indices = torch.nonzero(other_mask, as_tuple=True)[0]
                    indices_to_set = orig_indices[mask_i]
                    codes[indices_to_set] = i
        
        return codes
    
    low_codes = quantize_channel(low)
    high_codes = quantize_channel(high)
    
    packed = (high_codes << 4) | low_codes
    return packed.reshape(m, n)


def test_fp4_pack():
    torch.manual_seed(RAND_SEED)
    torch.set_default_device("cuda:0")
    # test_uint8 = torch.tensor([[0xAB, 0xCD], [0x12, 0x34]], dtype=torch.uint8)
    test_uint8 = torch.randint(0, 256, (4000, 2)).to(torch.uint8)
    print(f"Original uint8:\n{test_uint8}")
    print(f"Shape: {test_uint8.shape}")
    
    # 正向
    broken = break_fp4_bytes_v2(test_uint8)
    print(f"\nBroken float32:\n{broken}")
    print(f"Shape: {broken.shape}")
    
    # 反向
    packed = pack_fp4_bytes(broken)
    print(f"\nPacked back:\n{packed}")
    print(f"Shape: {packed.shape}")
    
    # 验证
    print(f"\nMatch: {torch.equal(test_uint8, packed)}")

    mismatch_mask = test_uint8 != packed
    if mismatch_mask.any():
        print(f"Mismatches at indices: {torch.nonzero(mismatch_mask)}")
        print(f"Original values: {test_uint8[mismatch_mask]}")
        print(f"Packed values: {packed[mismatch_mask]}")


def dequantize_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes_v2(tensor_fp4)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf_linear = convert_swizzled_to_linear(tensor_sf, m, k)
    block_scale = tensor_sf_linear.to(torch.float32) / global_scale

    # scale the tensor
    print(tensor_f32.shape, block_scale.shape)
    out = (tensor_f32 * block_scale.unsqueeze(-1)).reshape(m, k)
    return out


def get_ref_nvfp4_mul_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())

def get_ref_fp16_mul_results(
    a_half,
    b_half):
    _, m_k = a_half.shape
    _, n_k = b_half.shape
    assert m_k == n_k
    return torch.matmul(a_half, b_half.t())


def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int],
    test_mode: bool = False,
) -> None:
    torch.manual_seed(RAND_SEED)
    torch.set_default_device("cuda:0")
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = BLOCK_SIZE
    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(torch.abs(a_dtype.flatten()), dim=-1).to(torch.float32)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(torch.abs(b_dtype.flatten()), dim=-1).to(torch.float32)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    
    if test_mode:
        a_fp4_float, a_scale_linear = ref_nvfp4_quant(a_dtype, a_global_scale)
        a_scale_swizzled = convert_linear_to_swizzled(a_scale_linear, m, k)
        a_scale_interleaved = a_scale_swizzled.to(torch.float8_e4m3fn)
        a_fp4 = pack_fp4_bytes(a_fp4_float)

        b_fp4_float, b_scale_linear = ref_nvfp4_quant(b_dtype, b_global_scale)
        b_scale_swizzled = convert_linear_to_swizzled(b_scale_linear, n, k)
        b_scale_interleaved = b_scale_swizzled.to(torch.float8_e4m3fn)
        b_fp4 = pack_fp4_bytes(b_fp4_float)
        print(a_scale_swizzled.shape, b_scale_swizzled.shape)
        print(a_global_scale, b_global_scale)
        print(a_dtype)
        print(b_dtype)
        # print(a_scale_swizzled)
        # print(b_scale_swizzled)
        print(a_scale_linear)
        print(b_scale_linear)
        print(a_fp4_float)
        print(b_fp4_float)
    else:
        a_fp4_float, a_scale_interleaved = ref_scaled_fp4_quant(a_dtype, a_global_scale)
        b_fp4_float, b_scale_interleaved = ref_scaled_fp4_quant(b_dtype, b_global_scale)
        a_fp4 = pack_fp4_bytes(a_fp4_float)
        b_fp4 = pack_fp4_bytes(b_fp4_float)

    expected_out = get_ref_nvfp4_mul_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        "cuda",
    )
    print(expected_out, expected_out.shape)

    half_out = get_ref_fp16_mul_results(a_dtype, b_dtype)
    print(half_out, half_out.shape)
    # print(expected_out[95,214], half_out[95,214])
    torch.testing.assert_close(half_out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


class NVFP4Data:
    def __init__(self):
        self.M = 0
        self.N = 0
        self.K = 0
        self.A_fp16 = None
        self.B_fp16 = None
        self.A_fp4 = None
        self.B_fp4 = None
        self.A_scales = None
        self.B_scales = None
        self.output = None
        self.output_ref = None
        self.a_global_scale = 0.0
        self.b_global_scale = 0.0
        self.alpha = 0.0
    
    def save(self, filename: str):
        """保存数据到二进制文件"""
        with open(filename, 'wb') as f:
            # 写入维度信息
            f.write(struct.pack('iii', self.M, self.N, self.K))
            
            # 写入全局缩放因子
            f.write(struct.pack('fff', 
                              float(self.a_global_scale), 
                              float(self.b_global_scale), 
                              float(self.alpha)))
            
            # 写入数据大小
            size_A_fp16 = self.A_fp16.numel() if self.A_fp16 is not None else 0
            size_B_fp16 = self.B_fp16.numel() if self.B_fp16 is not None else 0
            size_A_fp4 = self.A_fp4.numel() if self.A_fp4 is not None else 0
            size_B_fp4 = self.B_fp4.numel() if self.B_fp4 is not None else 0
            size_A_scales = self.A_scales.numel() if self.A_scales is not None else 0
            size_B_scales = self.B_scales.numel() if self.B_scales is not None else 0
            size_output = self.output.numel() if self.output is not None else 0
            size_output_ref = self.output_ref.numel() if self.output_ref is not None else 0
            
            f.write(struct.pack('QQQQQQQQ', 
                              size_A_fp16, size_B_fp16,
                              size_A_fp4, size_B_fp4,
                              size_A_scales, size_B_scales,
                              size_output, size_output_ref))
            
            # 写入数据
            if self.A_fp16 is not None:
                f.write(self.A_fp16.cpu().numpy().tobytes())
            if self.B_fp16 is not None:
                f.write(self.B_fp16.cpu().numpy().tobytes())
            if self.A_fp4 is not None:
                f.write(self.A_fp4.cpu().numpy().tobytes())
            if self.B_fp4 is not None:
                f.write(self.B_fp4.cpu().numpy().tobytes())
            if self.A_scales is not None:
                f.write(self.A_scales.cpu().numpy().tobytes())
            if self.B_scales is not None:
                f.write(self.B_scales.cpu().numpy().tobytes())
            if self.output is not None:
                f.write(self.output.cpu().numpy().tobytes())
            if self.output_ref is not None:
                f.write(self.output_ref.cpu().numpy().tobytes())
        
        print(f"Data saved to {filename}")
    
    @classmethod
    def load(cls, filename: str, device="cuda", load_from_cpp=False):
        """从二进制文件加载数据"""
        data = cls()
        
        with open(filename, 'rb') as f:
            # 读取维度信息
            data.M, data.N, data.K = struct.unpack('iii', f.read(12))
            
            # 读取全局缩放因子
            a_gs, b_gs, alpha = struct.unpack('fff', f.read(12))
            data.a_global_scale = torch.tensor(a_gs, dtype=torch.float32)
            data.b_global_scale = torch.tensor(b_gs, dtype=torch.float32)
            data.alpha = torch.tensor(alpha, dtype=torch.float32)
            
            # 读取数据大小
            sizes = struct.unpack('QQQQQQQQ', f.read(64))
            (size_A_fp16, size_B_fp16, 
             size_A_fp4, size_B_fp4,
             size_A_scales, size_B_scales,
             size_output, size_output_ref) = sizes
            print(f"Sizes: {sizes}")
            
            def get_pad_dim(m, n):
                scale_n = n // BLOCK_SIZE
                rounded_m = ((m + 128 - 1) // 128) * 128
                rounded_n = ((scale_n + 4 - 1) // 4) * 4
                return (rounded_m, rounded_n)
            
            # 读取数据
            if size_A_fp16 > 0:
                buffer = f.read(size_A_fp16 * 2)  # FP16是2字节
                arr = np.frombuffer(buffer, dtype=np.float16).reshape(data.M, data.K)
                data.A_fp16 = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float16)
            
            if size_B_fp16 > 0:
                buffer = f.read(size_B_fp16 * 2)
                arr = np.frombuffer(buffer, dtype=np.float16).reshape(data.N, data.K)
                data.B_fp16 = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float16)
            
            if size_A_fp4 > 0:
                buffer = f.read(size_A_fp4 * 8)  # int64_t是8字节
                arr = np.frombuffer(buffer, dtype=np.uint8)
                data.A_fp4 = torch.reshape(torch.from_numpy(arr.copy()).to(device=device, dtype=torch.uint8), (data.M, data.K // 2))
            
            if size_B_fp4 > 0:
                buffer = f.read(size_B_fp4 * 8)
                arr = np.frombuffer(buffer, dtype=np.uint8)
                data.B_fp4 = torch.reshape(torch.from_numpy(arr.copy()).to(device=device, dtype=torch.uint8), (data.N, data.K // 2))
            
            if size_A_scales > 0:
                if not load_from_cpp:
                    buffer = f.read(size_A_scales * 4)
                    arr = np.frombuffer(buffer, dtype=np.int32)
                    data.A_scales = torch.reshape(torch.from_numpy(arr.copy()).to(device=device, dtype=torch.int32).to(torch.float8_e4m3fn), get_pad_dim(data.M, data.K))
                else:
                    buffer = f.read(size_A_scales * 4)
                    arr = np.frombuffer(buffer, dtype=np.uint8)
                    data.A_scales = torch.reshape(torch.from_numpy(arr.copy()).to(device=device, dtype=torch.uint8).view(torch.float8_e4m3fn), get_pad_dim(data.M, data.K))
            if size_B_scales > 0:
                if not load_from_cpp:
                    buffer = f.read(size_B_scales * 4)
                    arr = np.frombuffer(buffer, dtype=np.int32)
                    data.B_scales = torch.reshape(torch.from_numpy(arr.copy()).to(device=device, dtype=torch.int32).to(torch.float8_e4m3fn), get_pad_dim(data.N, data.K))
                else:
                    buffer = f.read(size_B_scales * 4)
                    arr = np.frombuffer(buffer, dtype=np.uint8)
                    data.B_scales = torch.reshape(torch.from_numpy(arr.copy()).to(device=device, dtype=torch.uint8).view(torch.float8_e4m3fn), get_pad_dim(data.N, data.K))
            if size_output > 0:
                buffer = f.read(size_output * 2)
                arr = np.frombuffer(buffer, dtype=np.float16).reshape(data.M, data.N)
                data.output = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float16)

            if size_output_ref > 0:
                buffer = f.read(size_output_ref * 2)
                arr = np.frombuffer(buffer, dtype=np.float16).reshape(data.M, data.N)
                data.output_ref = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float16)

        print(f"Data loaded from {filename}")
        print(f"M={data.M}, N={data.N}, K={data.K}")
        return data


def generate_and_save_test_data(
    M: int, N: int, K: int,
    output_file: str,
    dtype: torch.dtype = torch.float16):
    
    torch.manual_seed(RAND_SEED)
    device = "cuda"
    
    # 生成随机数据
    a_dtype = torch.randn((M, K), dtype=dtype, device=device)
    b_dtype = torch.randn((N, K), dtype=dtype, device=device)
    
    # 计算全局缩放因子
    a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / 
                     torch.amax(torch.abs(a_dtype.flatten()), dim=-1).to(torch.float32)).to(torch.float32)
    b_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / 
                     torch.amax(torch.abs(b_dtype.flatten()), dim=-1).to(torch.float32)).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    
    # 量化
    a_fp4_float, a_scale_interleaved = ref_scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4_float, b_scale_interleaved = ref_scaled_fp4_quant(b_dtype, b_global_scale)
    
    a_fp4 = pack_fp4_bytes(a_fp4_float)
    b_fp4 = pack_fp4_bytes(b_fp4_float)
    
    # 保存数据
    data = NVFP4Data()
    data.M = M
    data.N = N
    data.K = K
    data.A_fp16 = a_dtype
    data.B_fp16 = b_dtype
    data.A_fp4 = torch.from_numpy(np.frombuffer(
        a_fp4.detach().cpu().numpy().tobytes(), dtype=np.int64).copy()).to(device)
    data.B_fp4 = torch.from_numpy(np.frombuffer(
        b_fp4.detach().cpu().numpy().tobytes(), dtype=np.int64).copy()).to(device)
    print("fp4 shape: ", a_fp4.shape, b_fp4.shape)
    print("scale shape: ", a_scale_interleaved.shape, b_scale_interleaved.shape)
    data.A_scales = a_scale_interleaved.to(torch.int32)
    data.B_scales = b_scale_interleaved.to(torch.int32)
    data.a_global_scale = a_global_scale
    data.b_global_scale = b_global_scale
    data.alpha = alpha.detach().clone().to(dtype=torch.float32, device=device)
    
    # 计算参考输出
    data.output = get_ref_nvfp4_mul_results(
        a_fp4, b_fp4,
        a_scale_interleaved, b_scale_interleaved,
        a_global_scale, b_global_scale,
        M, N, dtype, BLOCK_SIZE, device).to(dtype=torch.float16, device=device)
    data.output_half = get_ref_fp16_mul_results(a_dtype, b_dtype)
    # print(a_dtype)
    # print(b_dtype)
    # print(data.output)
    
    data.save(output_file)
    return data

def compare_results(cpp_output_file: str, python_data_file: str, rtol: float = 1e-2, atol: float = 1e-2, device="cuda"):
    cpp_data = NVFP4Data.load(cpp_output_file, device, load_from_cpp=True)
    python_data = NVFP4Data.load(python_data_file, device)

    assert cpp_data.M == python_data.M
    assert cpp_data.N == python_data.N
    assert cpp_data.K == python_data.K

    torch.testing.assert_close(python_data.A_fp4, cpp_data.A_fp4)
    torch.testing.assert_close(python_data.B_fp4, cpp_data.B_fp4)
    torch.testing.assert_close(python_data.A_scales, cpp_data.A_scales)
    torch.testing.assert_close(python_data.B_scales, cpp_data.B_scales)
    torch.testing.assert_close(python_data.a_global_scale, cpp_data.a_global_scale)
    torch.testing.assert_close(python_data.b_global_scale, cpp_data.b_global_scale)
    torch.testing.assert_close(python_data.alpha, cpp_data.alpha)

    if cpp_data.output is not None and python_data.output is not None:
        cpp_output = cpp_data.output
        python_output = python_data.output
        
        abs_diff = torch.abs(cpp_output - python_output)
        rel_diff = abs_diff / (torch.abs(python_output) + 1e-8)
        
        max_abs_error = torch.max(abs_diff).item()
        max_rel_error = torch.max(rel_diff).item()
        mean_abs_error = torch.mean(abs_diff).item()
        mean_rel_error = torch.mean(rel_diff).item()
        
        print("\n=== 结果比较 ===")
        print(f"最大绝对误差: {max_abs_error:.6e}")
        print(f"最大相对误差: {max_rel_error:.6e}")
        print(f"平均绝对误差: {mean_abs_error:.6e}")
        print(f"平均相对误差: {mean_rel_error:.6e}")
        
        # 检查误差是否在容忍范围内
        abs_ok = torch.allclose(cpp_output, python_output, rtol=rtol, atol=atol)
        print(f"绝对误差检查: {'通过' if abs_ok else '失败'}")
        
        return max_abs_error, max_rel_error, abs_ok
    else:
        print("错误：输出数据为空")
        return 0, 0, False
    
def test_nvfp4_gemm_from_data(input_file):
    data = NVFP4Data.load(input_file)
    a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / 
                     torch.amax(torch.abs(data.A_fp16.flatten()), dim=-1).to(torch.float32)).to(torch.float32)
    b_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / 
                     torch.amax(torch.abs(data.B_fp16.flatten()), dim=-1).to(torch.float32)).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4_float, a_scale_interleaved = ref_scaled_fp4_quant(data.A_fp16, data.a_global_scale)
    b_fp4_float, b_scale_interleaved = ref_scaled_fp4_quant(data.B_fp16, data.b_global_scale)
    a_fp4 = pack_fp4_bytes(a_fp4_float)
    b_fp4 = pack_fp4_bytes(b_fp4_float)

    expected_out = get_ref_nvfp4_mul_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        data.a_global_scale,
        data.b_global_scale,
        data.M,
        data.N,
        data.A_fp16.dtype,
        BLOCK_SIZE,
        "cuda",
    )

    torch.testing.assert_close(data.A_fp4, a_fp4)
    torch.testing.assert_close(data.B_fp4, b_fp4)
    torch.testing.assert_close(data.A_scales, a_scale_interleaved)
    torch.testing.assert_close(data.B_scales, b_scale_interleaved)
    torch.testing.assert_close(data.a_global_scale.to("cuda"), a_global_scale)
    torch.testing.assert_close(data.b_global_scale.to("cuda"), b_global_scale)
    torch.testing.assert_close(data.alpha.to("cuda"), alpha)

    print(data.M, data.N, data.K)
    # print(a_global_scale, b_global_scale, alpha)
    # print(data.a_global_scale, data.b_global_scale, data.alpha)
    # print(data.A_fp16)
    # print(data.B_fp16)
    print(expected_out, expected_out.shape)
    print(data.output, data.output.shape)
    torch.testing.assert_close(data.output, expected_out.to(dtype=data.A_fp16.dtype), atol=1e-1, rtol=1e-1)

# test_quantize_to_fp4(torch.float16, (1, 64))
# test_quantize_to_fp4(torch.float16, (6144, 2048))
# test_fp4_pack()
# test_nvfp4_gemm(torch.float16, (1, 1, 64//2), True)
# generate_and_save_test_data(1, 8, 64, "nvfp4_testdata_1_8_64.bin")
# test_nvfp4_gemm_from_data("nvfp4_testdata_1_8_64.bin")


def mopt_nvfp4_quant(x):
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale_mopt =  tensor_amax / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
    weight_scaling_factor, weight_scaling_factor_2 = (
        NVFP4QTensor.get_weights_scaling_factor(x, 16, global_scale_mopt))
    quantized_weight, _, _ = (
        NVFP4QTensor.quantize(x, 16, weight_scaling_factor, weight_scaling_factor_2, try_tensorrt=True))

    return quantized_weight._quantized_data, weight_scaling_factor, 1.0 / weight_scaling_factor_2

def _get_padded_shape(M, K):
    round_up_multiple = lambda x, m: (x + m - 1) // m * m
    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    return M_padded, K_padded

def _prepare_scales(scales: torch.Tensor, scale_ndim: int = 2) -> torch.Tensor:
    B, M, K = scales.shape
    M_padded, K_padded = _get_padded_shape(M, K)
    
    # Create padded tensor
    padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype, device=scales.device)
    padded_scales[:B, :M, :K] = scales
    
    batches, rows, cols = padded_scales.shape
    assert rows % 128 == 0
    assert cols % 4 == 0
    
    # Reshape and permute
    padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
    padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
    padded_scales = padded_scales.contiguous()
    
    # Final reshape
    if scale_ndim == 2:
        padded_scales = padded_scales.reshape(M_padded, K_padded)
    else:
        padded_scales = padded_scales.reshape(B, M_padded, K_padded)
    
    return padded_scales

def test_mopt_nvfp4_quant(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    torch.manual_seed(RAND_SEED)
    torch.set_default_device("cuda:0")

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / tensor_amax
    out_ref, scale_linear = ref_nvfp4_quant(x, global_scale)
    packed_ref = pack_fp4_bytes(out_ref)
    scale_linear_ref = scale_linear.to(torch.float8_e4m3fn)
    print(packed_ref.shape, scale_linear_ref.shape)
    scale_swizzle = convert_linear_to_swizzled(scale_linear_ref, m, n)

    quantized_weight, weight_scaling_factor, weight_scaling_factor_2 = mopt_nvfp4_quant(x)
    qscales = _prepare_scales(weight_scaling_factor.unsqueeze(0), scale_ndim=3).squeeze(0)
    print(scale_swizzle.shape, qscales.shape)
    print(quantized_weight.shape, weight_scaling_factor.shape)
    print(packed_ref)
    print(quantized_weight)
    print(scale_linear_ref, weight_scaling_factor)
    print(global_scale, weight_scaling_factor_2)
    print(packed_ref[69, 246], quantized_weight[69, 246])
    print(packed_ref[2429, 950], quantized_weight[2429, 950])

    torch.testing.assert_close(global_scale, weight_scaling_factor_2)
    torch.testing.assert_close(scale_linear_ref, weight_scaling_factor)
    torch.testing.assert_close(scale_swizzle, qscales)
    torch.testing.assert_close(packed_ref, quantized_weight)

# test_mopt_nvfp4_quant(torch.float16, (1, 64))
# test_mopt_nvfp4_quant(torch.float16, (6144, 2048))