#!/usr/bin/env python3

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"xpu_available={torch.xpu.is_available()}")
    print(f"xpu_count={torch.xpu.device_count()}")
    for idx in range(torch.xpu.device_count()):
        print(f"xpu_{idx}={torch.xpu.get_device_name(idx)}")

    x = torch.arange(16, device="xpu")
    print(f"tensor_device={x.device}")
    print(f"tensor_sum={x.sum().item()}")


if __name__ == "__main__":
    main()
