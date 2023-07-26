import torch
from torch.backends.cuda import sdp_kernel

from torch.nn.functional import scaled_dot_product_attention

torch.cuda.manual_seed(2020)
torch.manual_seed(2020)
torch.backends.cuda.enable_flash_sdp(enabled=False)
torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
torch.backends.cuda.enable_math_sdp(enabled=False)


def main():
    query = torch.rand(1, 16, 1, 64, dtype=torch.bfloat16).to("cuda")
    key = torch.rand(1, 16, 1, 64, dtype=torch.bfloat16).to("cuda")
    value = torch.rand(1, 16, 1, 64, dtype=torch.bfloat16).to("cuda")
    mask = torch.rand(1, 1, 1, 1, dtype=torch.bfloat16).to("cuda")
    print(f"{mask.stride()=}")
    mask = mask.as_strided(mask.size(), (1, 16, 1, 1), 16)
    print(f"{mask.stride()=}")
    # with sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
    res = scaled_dot_product_attention(query, key, value, mask, scale=1)
    print(res)


if __name__ == "__main__":
    main()
