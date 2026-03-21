#!/usr/bin/env python3
"""Pad a GGUF MoE model's expert count to be divisible by N GPUs.

Binary-level approach: copies the GGUF file, modifying only expert tensors
and the expert_count metadata. Non-expert tensors are copied byte-for-byte.

Usage:
    python3 pad-experts-gguf.py INPUT.gguf OUTPUT.gguf [--n-gpus 3]
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np


# GGUF format constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# GGML quant block sizes: (block_elements, block_bytes)
QUANT_SIZES = {
    0:  (1, 4),        # F32
    1:  (1, 2),        # F16
    2:  (32, 18),      # Q4_0
    3:  (32, 20),      # Q4_1
    6:  (32, 22),      # Q5_0
    7:  (32, 24),      # Q5_1
    8:  (32, 34),      # Q8_0
    9:  (32, 36),      # Q8_1
    10: (256, 84),     # Q2_K
    11: (256, 110),    # Q3_K
    12: (256, 144),    # Q4_K
    13: (256, 176),    # Q5_K
    14: (256, 210),    # Q6_K
    15: (256, 292),    # Q8_K
    16: (256, 50),     # IQ2_XXS
    17: (256, 66),     # IQ2_XS
    18: (256, 82),     # IQ3_XXS
    19: (256, 34),     # IQ1_S (QK_K=256, sizeof=2+32+16=50? check)
    20: (32, 18),      # IQ4_NL (QK4_NL=32, sizeof=2+16=18)
    21: (256, 110),    # IQ3_S
    22: (256, 82),     # IQ2_S
    23: (32, 20),      # IQ4_XS
    28: (1, 2),        # BF16
}

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def write_string(f, s):
    data = s.encode('utf-8')
    f.write(struct.pack('<Q', len(data)))
    f.write(data)

def read_value(f, vtype):
    if vtype == GGUF_TYPE_STRING:
        return read_string(f)
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack('<?', f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack('<q', f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack('<d', f.read(8))[0]
    elif vtype == GGUF_TYPE_UINT8:
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack('<b', f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack('<H', f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack('<h', f.read(2))[0]
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, elem_type) for _ in range(count)]
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")

def write_value(f, vtype, val):
    if vtype == GGUF_TYPE_STRING:
        write_string(f, val)
    elif vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack('<I', val))
    elif vtype == GGUF_TYPE_INT32:
        f.write(struct.pack('<i', val))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack('<f', val))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(struct.pack('<?', val))
    elif vtype == GGUF_TYPE_UINT64:
        f.write(struct.pack('<Q', val))
    elif vtype == GGUF_TYPE_INT64:
        f.write(struct.pack('<q', val))
    elif vtype == GGUF_TYPE_FLOAT64:
        f.write(struct.pack('<d', val))
    elif vtype == GGUF_TYPE_UINT8:
        f.write(struct.pack('<B', val))
    elif vtype == GGUF_TYPE_INT8:
        f.write(struct.pack('<b', val))
    elif vtype == GGUF_TYPE_UINT16:
        f.write(struct.pack('<H', val))
    elif vtype == GGUF_TYPE_INT16:
        f.write(struct.pack('<h', val))
    elif vtype == GGUF_TYPE_ARRAY:
        raise ValueError("Use write_array for arrays")

def write_array(f, elem_type, arr):
    f.write(struct.pack('<I', elem_type))
    f.write(struct.pack('<Q', len(arr)))
    for v in arr:
        write_value(f, elem_type, v)


def compute_tensor_bytes(shape, tensor_type):
    """Compute total bytes for a tensor given its logical shape and quant type."""
    block_elems, block_bytes = QUANT_SIZES[tensor_type]
    n_elements = 1
    for d in shape:
        n_elements *= d
    if tensor_type in (0, 1, 28):  # F32, F16, BF16
        return n_elements * block_bytes
    else:
        # Quantized: row-major, quant blocks along first dimension
        row_size = shape[0]
        n_rows = n_elements // row_size
        blocks_per_row = (row_size + block_elems - 1) // block_elems
        return n_rows * blocks_per_row * block_bytes


def expert_slice_bytes(shape, tensor_type):
    """Bytes for one expert slice (shape without last dim)."""
    inner_shape = list(shape[:-1])
    return compute_tensor_bytes(inner_shape, tensor_type)


def main():
    parser = argparse.ArgumentParser(description="Pad GGUF MoE experts for GPU divisibility")
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", help="Output GGUF file")
    parser.add_argument("--n-gpus", type=int, default=3, help="Number of GPUs (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be done")
    args = parser.parse_args()

    with open(args.input, 'rb') as fin:
        # Read header
        magic = fin.read(4)
        assert magic == GGUF_MAGIC, f"Not a GGUF file: {magic}"
        version = struct.unpack('<I', fin.read(4))[0]
        assert version == GGUF_VERSION, f"Unsupported GGUF version: {version}"
        n_tensors = struct.unpack('<Q', fin.read(8))[0]
        n_kv = struct.unpack('<Q', fin.read(8))[0]
        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} KV pairs")

        # Read KV pairs
        kv_pairs = []  # (key, vtype, value)
        expert_count = None
        expert_count_key = None
        for _ in range(n_kv):
            key = read_string(fin)
            vtype = struct.unpack('<I', fin.read(4))[0]
            if vtype == GGUF_TYPE_ARRAY:
                elem_type = struct.unpack('<I', fin.read(4))[0]
                count = struct.unpack('<Q', fin.read(8))[0]
                arr = [read_value(fin, elem_type) for _ in range(count)]
                kv_pairs.append((key, vtype, (elem_type, arr)))
            else:
                val = read_value(fin, vtype)
                kv_pairs.append((key, vtype, val))
                if key.endswith('.expert_count'):
                    expert_count = val
                    expert_count_key = key

        print(f"Expert count: {expert_count}")
        if expert_count is None:
            print("ERROR: No expert_count found in metadata")
            sys.exit(1)

        if expert_count % args.n_gpus == 0:
            print(f"Already divisible by {args.n_gpus}!")
            return

        padded_count = math.ceil(expert_count / args.n_gpus) * args.n_gpus
        n_fake = padded_count - expert_count
        print(f"Padding: {expert_count} → {padded_count} (+{n_fake} fake)")
        print(f"  {padded_count} / {args.n_gpus} = {padded_count // args.n_gpus} per GPU")

        # Read tensor infos
        tensor_infos = []  # (name, n_dims, shape, tensor_type, offset)
        for _ in range(n_tensors):
            name = read_string(fin)
            n_dims = struct.unpack('<I', fin.read(4))[0]
            shape = [struct.unpack('<Q', fin.read(8))[0] for _ in range(n_dims)]
            tensor_type = struct.unpack('<I', fin.read(4))[0]
            offset = struct.unpack('<Q', fin.read(8))[0]
            tensor_infos.append((name, n_dims, shape, tensor_type, offset))

        # Find data start (aligned to 32 bytes after header)
        header_end = fin.tell()
        alignment = 32
        data_start = (header_end + alignment - 1) // alignment * alignment

        # Categorize tensors
        n_expert_tensors = 0
        n_gate_tensors = 0
        gate_layer_indices = []  # block indices that have gate_inp (for bias tensors)
        total_pad_bytes = 0
        for name, n_dims, shape, ttype, offset in tensor_infos:
            if any(x in name for x in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')):
                n_expert_tensors += 1
                total_pad_bytes += expert_slice_bytes(shape, ttype) * n_fake
            elif name.endswith('ffn_gate_inp.weight') and 'shexp' not in name:
                n_gate_tensors += 1
                total_pad_bytes += shape[0] * 4 * n_fake  # F32 rows
                # Extract block index for bias tensor creation
                # name like "blk.0.ffn_gate_inp.weight"
                parts = name.split('.')
                for j, p in enumerate(parts):
                    if p == 'blk' and j + 1 < len(parts):
                        gate_layer_indices.append(int(parts[j + 1]))
                        break
                # Also account for the new bias tensor bytes
                total_pad_bytes += padded_count * 4  # F32 bias [n_expert_padded]

        print(f"\n  {n_expert_tensors} expert FFN tensors")
        print(f"  {n_gate_tensors} router gate tensors")
        print(f"  {n_gate_tensors} new gate bias tensors (to suppress fake experts)")
        print(f"  Total padding: {total_pad_bytes / 1024 / 1024:.1f} MB")

        if args.dry_run:
            print("\n--dry-run: stopping here")
            return

        # Build bias tensor infos for each MoE gate layer
        # These suppress fake experts: real experts get bias=0, fake experts get bias=-1e30
        # This is added AFTER the dot product, so logit = dot(zero_row, hidden) + (-1e30) = -1e30
        # regardless of hidden_state sign. Combined with zero gate weights, this guarantees
        # fake experts are never selected.
        bias_tensors = []  # (name, shape, data_bytes)
        for blk_idx in sorted(gate_layer_indices):
            bias_name = f"blk.{blk_idx}.ffn_gate_inp.bias"
            bias_shape = [padded_count]
            bias_data = np.zeros(padded_count, dtype=np.float32)
            bias_data[expert_count:] = -1e30  # fake experts get -1e30
            bias_tensors.append((bias_name, bias_shape, bias_data.tobytes()))

        n_tensors_out = n_tensors + len(bias_tensors)

        # Write output
        print(f"\nWriting {args.output}...")
        print(f"  Tensors: {n_tensors} → {n_tensors_out} (+{len(bias_tensors)} gate bias)")
        with open(args.output, 'wb') as fout:
            # Header
            fout.write(GGUF_MAGIC)
            fout.write(struct.pack('<I', version))
            fout.write(struct.pack('<Q', n_tensors_out))
            fout.write(struct.pack('<Q', n_kv))

            # KV pairs (update expert_count)
            for key, vtype, val in kv_pairs:
                write_string(fout, key)
                fout.write(struct.pack('<I', vtype))
                if key == expert_count_key:
                    print(f"  {key}: {expert_count} → {padded_count}")
                    write_value(fout, vtype, padded_count)
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type, arr = val
                    write_array(fout, elem_type, arr)
                else:
                    write_value(fout, vtype, val)

            # Compute new shapes
            new_shapes = []
            for name, n_dims, shape, ttype, old_offset in tensor_infos:
                new_shape = list(shape)
                if any(x in name for x in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')):
                    new_shape[-1] = padded_count
                elif name.endswith('ffn_gate_inp.weight') and 'shexp' not in name:
                    new_shape[-1] = padded_count
                new_shapes.append(new_shape)

            # Write tensor infos with placeholder offsets (we'll patch them later)
            ti_offset_positions = []  # file positions where we wrote offset values
            for i, (name, n_dims, shape, ttype, old_offset) in enumerate(tensor_infos):
                write_string(fout, name)
                fout.write(struct.pack('<I', n_dims))
                for d in new_shapes[i]:
                    fout.write(struct.pack('<Q', d))
                fout.write(struct.pack('<I', ttype))
                ti_offset_positions.append(fout.tell())
                fout.write(struct.pack('<Q', 0))  # placeholder

            # Write bias tensor infos (F32, 1D)
            bias_offset_positions = []
            for bias_name, bias_shape, bias_bytes in bias_tensors:
                write_string(fout, bias_name)
                fout.write(struct.pack('<I', len(bias_shape)))  # n_dims = 1
                for d in bias_shape:
                    fout.write(struct.pack('<Q', d))
                fout.write(struct.pack('<I', 0))  # type = F32
                bias_offset_positions.append(fout.tell())
                fout.write(struct.pack('<Q', 0))  # placeholder

            # Align to data start
            pos = fout.tell()
            new_data_start = (pos + alignment - 1) // alignment * alignment
            fout.write(b'\x00' * (new_data_start - pos))

            # Read alignment from KV pairs (default 32)
            file_alignment = 32
            for key, vtype, val in kv_pairs:
                if key == 'general.alignment':
                    file_alignment = val if not isinstance(val, tuple) else 32
                    break

            def GGML_PAD(x, n):
                return ((x + n - 1) // n) * n

            # Compute offsets first (the way the GGUF loader does):
            # offset[0] = 0, offset[i] = offset[i-1] + GGML_PAD(nbytes[i-1], alignment)
            computed_offsets = []
            running = 0
            for i, (name, n_dims, old_shape, ttype, old_offset) in enumerate(tensor_infos):
                computed_offsets.append(running)
                new_shape = list(old_shape)
                if any(x in name for x in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')) or \
                   (name.endswith('ffn_gate_inp.weight') and 'shexp' not in name):
                    new_shape[-1] = padded_count
                nb = compute_tensor_bytes(new_shape, ttype)
                running += GGML_PAD(nb, file_alignment)

            # Compute offsets for bias tensors (appended after original tensors)
            bias_offsets = []
            for bias_name, bias_shape, bias_bytes in bias_tensors:
                bias_offsets.append(running)
                nb = len(bias_bytes)  # F32, simple
                running += GGML_PAD(nb, file_alignment)

            # Pre-allocate the full data section with zeros
            fout.write(b'\x00' * running)

            # Write tensor data at computed offsets
            for i, (name, n_dims, old_shape, ttype, old_offset) in enumerate(tensor_infos):
                old_bytes = compute_tensor_bytes(old_shape, ttype)

                # Seek to the correct position
                fout.seek(new_data_start + computed_offsets[i])

                # Read original data
                fin.seek(data_start + old_offset)
                data = fin.read(old_bytes)

                if any(x in name for x in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')):
                    # Expert tensor: write original + zero padding for fake experts
                    fout.write(data)
                    pad_bytes = expert_slice_bytes(old_shape, ttype) * n_fake
                    fout.write(b'\x00' * pad_bytes)

                elif name.endswith('ffn_gate_inp.weight') and 'shexp' not in name:
                    # Router gate: write original + ZERO rows for fake experts
                    # WHY zeros (not -1e9): The gate weight dot product is logit = dot(W_row, hidden_state).
                    # With -1e9 weights, if sum(hidden_state) < 0, the logit becomes POSITIVE,
                    # causing fake experts to be selected ~50% of the time. Zero weights always
                    # produce logit = 0.0 regardless of input. The companion bias tensor then
                    # adds -1e30 to fake expert logits, guaranteeing they're never selected.
                    fout.write(data)
                    hidden_size = old_shape[0]
                    zero_row = np.zeros(hidden_size, dtype=np.float32).tobytes()
                    for _ in range(n_fake):
                        fout.write(zero_row)
                else:
                    # Copy as-is
                    fout.write(data)

            # Write bias tensor data
            for j, (bias_name, bias_shape, bias_bytes) in enumerate(bias_tensors):
                fout.seek(new_data_start + bias_offsets[j])
                fout.write(bias_bytes)

            # Write computed offsets into header (original tensors)
            for i, offset_pos in enumerate(ti_offset_positions):
                fout.seek(offset_pos)
                fout.write(struct.pack('<Q', computed_offsets[i]))

            # Write computed offsets into header (bias tensors)
            for j, offset_pos in enumerate(bias_offset_positions):
                fout.seek(offset_pos)
                fout.write(struct.pack('<Q', bias_offsets[j]))

        in_size = Path(args.input).stat().st_size
        out_size = Path(args.output).stat().st_size
        print(f"\nDone! {in_size/1024/1024/1024:.2f} GB → {out_size/1024/1024/1024:.2f} GB (+{(out_size-in_size)/1024/1024:.1f} MB)")
        print(f"Fake experts suppressed via: zero gate weights + bias=-1e30 (sign-invariant)")


if __name__ == "__main__":
    main()
