"""Reversible scalefactor embedding with QMDCT compensation."""

import os
import math
import hashlib
import numpy as np

from mp3_decoder.decoder.decoder import Decoder
from mp3_decoder.decoder.FrameHeader import ChannelMode
from mp3_decoder.decoder.tables import slen as slen_table
from mp3_encoder import (
    BitWriter, pack_mp3,
    encode_scale_factors, encode_huffman_samples,
    select_optimal_tables_for_granule,
)


MAX_QUANTIZED_VALUE = 8191          # 2^13 - 1, max magnitude in MPEG-1 Layer III
HISTOGRAM_SHIFT_THRESHOLD = 0       # T = 0 for histogram shifting
DEFAULT_PRUNE_EXACT_CANDIDATES = 32  # exact net-capacity candidates per step

# Pre-compute the two forward multipliers once
FORWARD_MULT_K2 = 2.0 ** (3 * 2 / 16.0)   # k=2 → 2^(6/16) = 2^(3/8)
FORWARD_MULT_K4 = 2.0 ** (3 * 4 / 16.0)   # k=4 → 2^(12/16) = 2^(3/4)

# Maximum *original* magnitude whose forward-expanded result ≤ 8191
MAX_SAFE_K2 = int(math.floor(MAX_QUANTIZED_VALUE / FORWARD_MULT_K2))
MAX_SAFE_K4 = int(math.floor(MAX_QUANTIZED_VALUE / FORWARD_MULT_K4))


def _build_canonical_magnitude_table(mult):
    """Collapse every forward-compensation orbit to its smallest value."""
    inverse = {}
    for old in range(MAX_QUANTIZED_VALUE + 1):
        new = round(old * mult)
        if new > MAX_QUANTIZED_VALUE:
            break
        inverse[new] = old
    roots = []
    for value in range(MAX_QUANTIZED_VALUE + 1):
        current = value
        while current in inverse and inverse[current] < current:
            current = inverse[current]
        roots.append(current)
    return np.asarray(roots, dtype=np.int32)


CANONICAL_MAGNITUDE_K2 = _build_canonical_magnitude_table(FORWARD_MULT_K2)
CANONICAL_MAGNITUDE_K4 = _build_canonical_magnitude_table(FORWARD_MULT_K4)


def print_binary(label, bits):
    """Print a bit list as binary string and hex to the console."""


def forward_compensate_band(samples, band_start, band_end, k, big_value_end=576):
    """Expand magnitudes in place within the big_value region, preserving signs."""
    mult = FORWARD_MULT_K2 if k == 2 else FORWARD_MULT_K4
    actual_end = min(band_end, big_value_end, len(samples))
    for i in range(band_start, actual_end):
        val = int(samples[i])
        if val == 0:
            continue
        sign = 1 if val > 0 else -1
        samples[i] = sign * round(abs(val) * mult)


def get_k(scalefac_scale_flag):
    """Return k=2 if scalefac_scale==0, else k=4."""
    return 2 if int(scalefac_scale_flag) == 0 else 4


def get_slen_limits(side_info, gr, ch):
    s1 = int(side_info.slen1[gr][ch])
    s2 = int(side_info.slen2[gr][ch])
    lim1 = (1 << s1) - 1 if s1 > 0 else 0
    lim2 = (1 << s2) - 1 if s2 > 0 else 0
    return s1, s2, lim1, lim2


def get_embeddable_sfbs(side_info, gr, ch):
    s1 = int(side_info.slen1[gr][ch])
    s2 = int(side_info.slen2[gr][ch])
    lim1 = (1 << s1) - 1 if s1 > 0 else 0
    lim2 = (1 << s2) - 1 if s2 > 0 else 0

    if gr == 0:
        for sfb in range(11):
            if s1 > 0:
                yield sfb, s1, lim1
        for sfb in range(11, 21):
            if s2 > 0:
                yield sfb, s2, lim2
    else:
        SB = [6, 11, 16, 21]
        PREV_SB = [0, 6, 11, 16]
        for i in range(2):
            if not side_info.scfsi[ch][i]:
                for sfb in range(PREV_SB[i], SB[i]):
                    if s1 > 0:
                        yield sfb, s1, lim1
        for i in range(2, 4):
            if not side_info.scfsi[ch][i]:
                for sfb in range(PREV_SB[i], SB[i]):
                    if s2 > 0:
                        yield sfb, s2, lim2


def get_long_band_boundaries(header):
    bw = header.band_index.long_win
    boundaries = []
    for sfb in range(21):
        s = int(bw[sfb])
        e = int(bw[sfb + 1]) if sfb + 1 < len(bw) else 576
        boundaries.append((s, e))
    return boundaries


def _rice_encode(value, k):
    """Golomb-Rice encode value >= 1 with parameter k."""
    q = (value - 1) >> k
    r = (value - 1) & ((1 << k) - 1)
    bits = [0] * q + [1]          # unary quotient + stop bit
    for shift in range(k - 1, -1, -1):
        bits.append((r >> shift) & 1)
    return bits


def _rice_decode(bits, pos, k):
    q = 0
    while pos < len(bits) and bits[pos] == 0:
        q += 1
        pos += 1
    if pos >= len(bits):
        raise ValueError("Truncated Golomb-Rice code.")
    pos += 1  # skip stop bit
    r = 0
    for _ in range(k):
        if pos >= len(bits):
            raise ValueError("Truncated Golomb-Rice remainder.")
        r = (r << 1) | bits[pos]
        pos += 1
    return (q << k) + r + 1, pos


def _gamma_encode_nonnegative(value):
    """Elias-gamma encode a non-negative integer as value + 1."""
    binary = [int(bit) for bit in bin(value + 1)[2:]]
    return [0] * (len(binary) - 1) + binary


def _gamma_decode_nonnegative(bits, pos=0):
    """Decode one Elias-gamma-coded non-negative integer."""
    zeros = 0
    while pos < len(bits) and bits[pos] == 0:
        zeros += 1
        pos += 1
    if pos >= len(bits):
        raise ValueError("Truncated Elias-gamma integer.")
    value = 1
    pos += 1
    for _ in range(zeros):
        if pos >= len(bits):
            raise ValueError("Truncated Elias-gamma payload.")
        value = (value << 1) | bits[pos]
        pos += 1
    return value - 1, pos


def pack_secret_bits(secret_bits):
    """Prefix a binary secret with a self-delimiting bit-length code."""
    secret_bits = list(secret_bits)
    prefix = _gamma_encode_nonnegative(len(secret_bits))
    return prefix + secret_bits


def unpack_secret_bits(bits):
    """Decode a length-prefixed secret and return (secret, consumed_bits)."""
    length, prefix_end = _gamma_decode_nonnegative(bits)
    secret_end = prefix_end + length
    if secret_end > len(bits):
        raise ValueError("Truncated self-delimiting secret packet.")
    return list(bits[prefix_end:secret_end]), secret_end


def max_secret_bits_for_packet_capacity(capacity):
    """Largest user payload whose gamma length prefix also fits."""
    capacity = max(0, int(capacity))
    length = capacity
    while length + len(_gamma_encode_nonnegative(length)) > capacity:
        length -= 1
    return length


def _delta_rice_compress(bitmap):
    positions = [i for i, b in enumerate(bitmap) if b == 0]
    n_safe = len(positions)
    if n_safe == 0:
        return _gamma_encode_nonnegative(0) + [0, 0, 0]

    deltas = []
    prev = -1
    for p in positions:
        deltas.append(p - prev)
        prev = p

    best_k = 0
    best_coded = None
    best_len = float('inf')
    for k in range(7):
        coded = []
        for d in deltas:
            coded.extend(_rice_encode(d, k))
        if len(coded) < best_len:
            best_len = len(coded)
            best_k = k
            best_coded = coded

    out = _gamma_encode_nonnegative(n_safe)
    for shift in range(2, -1, -1):
        out.append((best_k >> shift) & 1)
    out.extend(best_coded)
    return out


def _delta_rice_decompress(bits, expected_length, return_consumed=False):
    """Decompress delta-Rice encoded bitmap."""
    n_safe, pos_bits = _gamma_decode_nonnegative(bits)
    if n_safe > expected_length:
        raise ValueError("Location map contains too many SAFE groups.")
    if pos_bits + 3 > len(bits):
        raise ValueError("Truncated delta-Rice location map header.")
    k = 0
    for i in range(3):
        k = (k << 1) | bits[pos_bits + i]

    bitmap = [1] * expected_length
    pos_bits += 3
    prev = -1
    for _ in range(n_safe):
        delta, pos_bits = _rice_decode(bits, pos_bits, k)
        prev += delta
        if prev < expected_length:
            bitmap[prev] = 0

    return (bitmap, pos_bits) if return_consumed else bitmap


def _rle_compress(bitmap):
    if not bitmap:
        return []
    
    runs = []
    current_bit = bitmap[0]
    count = 1
    
    for i in range(1, len(bitmap)):
        if bitmap[i] == current_bit:
            count += 1
        else:
            runs.append(count)
            current_bit = bitmap[i]
            count = 1
    runs.append(count)
    
    # If bitmap starts with 0, prepend a 0-length run of 1s
    if bitmap[0] == 0:
        runs.insert(0, 0)
    
    # Encode each run length as 8 bits
    bits = []
    for run in runs:
        # Split runs > 255 into multiple runs with alternating 0-length runs
        while run > 255:
            for shift in range(7, -1, -1):
                bits.append((255 >> shift) & 1)
            for shift in range(7, -1, -1):
                bits.append(0)  # 0-length run of opposite bit
            run -= 255
        for shift in range(7, -1, -1):
            bits.append((run >> shift) & 1)
    
    return bits


def _rle_decompress(bits, expected_length, return_consumed=False):
    """Decompress RLE encoded bitmap."""
    if not bits:
        result = []
        return (result, 0) if return_consumed else result
    
    bitmap = []
    current_bit = 1
    pos = 0
    
    while pos < len(bits) and len(bitmap) < expected_length:
        # Read 8-bit run length
        if pos + 8 > len(bits):
            raise ValueError("Truncated RLE location map run.")
        run_length = 0
        for i in range(8):
            run_length = (run_length << 1) | bits[pos + i]
        pos += 8
        
        # Add run_length of current_bit
        for _ in range(min(run_length, expected_length - len(bitmap))):
            bitmap.append(current_bit)
        
        # Alternate bit
        current_bit = 1 - current_bit
    
    # Pad if necessary
    while len(bitmap) < expected_length:
        bitmap.append(1)
    
    result = bitmap[:expected_length]
    return (result, pos) if return_consumed else result


def compress_location_map(bitmap):
    if not bitmap:
        return []

    dr_bits = _delta_rice_compress(bitmap)
    rle_bits = _rle_compress(bitmap)

    dr_cost = 1 + len(dr_bits)
    rle_cost = 1 + len(rle_bits)

    if dr_cost <= rle_cost:
        return [1] + dr_bits           # flag=1: delta-Rice
    else:
        return [0] + rle_bits          # flag=0: RLE


def decompress_location_map(bits, expected_length, return_consumed=False):
    if not bits or len(bits) < 1:
        result = []
        return (result, 0) if return_consumed else result

    flag = bits[0]

    if flag == 1:          # delta-Rice
        result = _delta_rice_decompress(
            bits[1:], expected_length, return_consumed=return_consumed)
    else:                  # RLE
        result = _rle_decompress(
            bits[1:], expected_length, return_consumed=return_consumed)

    if not return_consumed:
        return result
    bitmap, consumed = result
    return bitmap, consumed + 1


def pack_displaced_lsbs(bits):
    """Self-terminate displaced LSBs without ever expanding by >1 bit."""
    if not bits:
        return []
    compressed = compress_location_map(bits)
    if len(compressed) < len(bits):
        return [1] + compressed
    return [0] + list(bits)


def unpack_displaced_lsbs(bits, expected_length):
    """Return displaced LSBs and the number of Super-Payload bits used."""
    if expected_length == 0:
        return [], 0
    if not bits:
        raise ValueError("Missing displaced Header LSB mode bit.")
    if bits[0] == 0:
        end = 1 + expected_length
        if len(bits) < end:
            raise ValueError("Truncated raw displaced Header LSBs.")
        return list(bits[1:end]), end
    decoded, consumed = decompress_location_map(
        bits[1:], expected_length, return_consumed=True)
    return decoded, consumed + 1


def collect_group_keys(frame_data):
    """Return audio (frame, granule, channel) keys in LM order."""
    info_indices = get_info_frame_indices(frame_data)
    return [
        (fi, gr, ch)
        for fi, frame in enumerate(frame_data)
        if fi not in info_indices
        for gr in range(2)
        for ch in range(frame.get_header().channels)
    ]


def is_short_group(frame_data, key):
    fi, gr, ch = key
    si = frame_data[fi].side_info
    return bool(si.window_switching[gr][ch]) and int(si.block_type[gr][ch]) == 2


def stored_location_map(frame_data, location_map, group_keys):
    """Drop short-block flags because the extractor observes them directly."""
    return [
        bit for bit, key in zip(location_map, group_keys)
        if not is_short_group(frame_data, key)
    ]


def restore_location_map(frame_data, stored_map):
    """Reinsert deterministic short-block SKIP flags."""
    restored = []
    pos = 0
    for key in collect_group_keys(frame_data):
        if is_short_group(frame_data, key):
            restored.append(1)
            continue
        if pos >= len(stored_map):
            raise ValueError("Stored location map ended before all long groups.")
        restored.append(stored_map[pos])
        pos += 1
    if pos != len(stored_map):
        raise ValueError("Stored location map has trailing group flags.")
    return restored


def collect_sf_positions(frame_data):
    for frame_idx, frame in enumerate(frame_data):
        header = frame.get_header()
        si = frame.side_info

        # Skip Info/Xing frames
        if all(int(si.part2_3_length[g][c]) == 0
               for g in range(2) for c in range(header.channels)):
            continue

        for gr in range(2):
            for ch in range(header.channels):
                bt = int(si.block_type[gr][ch])
                ws = bool(si.window_switching[gr][ch])

                # Only long blocks
                if ws and bt == 2:
                    continue

                s1 = int(si.slen1[gr][ch])
                s2 = int(si.slen2[gr][ch])
                lim1 = (1 << s1) - 1 if s1 > 0 else 0
                lim2 = (1 << s2) - 1 if s2 > 0 else 0

                if gr == 0:
                    # Granule 0: all 21 bands always encoded
                    for sfb in range(11):
                        if s1 > 0:
                            yield (frame_idx, gr, ch, sfb, s1, lim1)
                    for sfb in range(11, 21):
                        if s2 > 0:
                            yield (frame_idx, gr, ch, sfb, s2, lim2)
                else:
                    # Granule 1: respect scfsi reuse flags
                    SB = [6, 11, 16, 21]
                    PREV_SB = [0, 6, 11, 16]
                    for i in range(2):
                        if not si.scfsi[ch][i]:
                            for sfb in range(PREV_SB[i], SB[i]):
                                if s1 > 0:
                                    yield (frame_idx, gr, ch, sfb, s1, lim1)
                    for i in range(2, 4):
                        if not si.scfsi[ch][i]:
                            for sfb in range(PREV_SB[i], SB[i]):
                                if s2 > 0:
                                    yield (frame_idx, gr, ch, sfb, s2, lim2)


def sf_position_consumers(frame_data, position):
    """Return granules whose decoded audio uses one encoded sf field."""
    fi, gr, ch, sfb, _, _ = position
    consumers = [(gr, ch)]
    if gr != 0:
        return consumers

    if sfb < 6:
        scfsi_band = 0
    elif sfb < 11:
        scfsi_band = 1
    elif sfb < 16:
        scfsi_band = 2
    else:
        scfsi_band = 3
    if bool(frame_data[fi].side_info.scfsi[ch][scfsi_band]):
        consumers.append((1, ch))
    return consumers


def sf_position_is_zero_band(frame_data, position):
    """True when every granule consuming this sf has an all-zero band."""
    return sf_position_nonzero_count(frame_data, position) == 0


def sf_position_nonzero_count(frame_data, position):
    """Count spectral nonzeros affected by one encoded sf field."""
    fi, _, _, sfb, _, _ = position
    frame = frame_data[fi]
    start, end = get_long_band_boundaries(frame.get_header())[sfb]
    return sum(
        int(np.count_nonzero(
            frame.raw_quantized_samples[gr][ch][start:end]))
        for gr, ch in sf_position_consumers(frame_data, position)
    )


def sf_position_distortion_score(frame_data, position):
    fi, _, _, sfb, _, _ = position
    frame = frame_data[fi]
    side_info = frame.side_info
    start, end = get_long_band_boundaries(frame.get_header())[sfb]
    weighted_energy = 0
    nonzero_count = 0
    for gr, ch in sf_position_consumers(frame_data, position):
        k = get_k(side_info.scale_fac_scale[gr][ch])
        roots = (CANONICAL_MAGNITUDE_K2 if k == 2
                 else CANONICAL_MAGNITUDE_K4)
        magnitudes = np.abs(np.asarray(
            frame.raw_quantized_samples[gr][ch][start:end],
            dtype=np.int64))
        canonical = roots[magnitudes]
        band_energy = sum(int(value) ** 4 for value in canonical)
        global_gain = int(side_info.global_gain[gr][ch])
        half_step = 46341 if global_gain & 1 else 32768
        weighted_energy += (
            band_energy * half_step << (global_gain // 2))
        nonzero_count += int(np.count_nonzero(magnitudes))
    return weighted_energy, nonzero_count


def collect_header_sf_positions(frame_data, order="distortion"):
    """Return Header carriers in file or distortion-prioritized order."""
    positions = list(collect_sf_positions(frame_data))
    if order == "file":
        return positions
    if order == "nonzero":
        return sorted(positions,
                      key=lambda position: (
                          sf_position_nonzero_count(frame_data, position),
                          position))
    if order not in ("distortion", "energy"):
        raise ValueError(f"Unknown Header carrier order: {order}")
    return sorted(positions,
                  key=lambda position: (
                      *sf_position_distortion_score(frame_data, position),
                      position))


def get_info_frame_indices(frame_data):
    info_indices = set()
    for fi, fr in enumerate(frame_data):
        hdr = fr.get_header()
        si = fr.side_info
        if all(int(si.part2_3_length[g][c]) == 0
               for g in range(2) for c in range(hdr.channels)):
            info_indices.add(fi)
    return info_indices


def frame_payload_capacity(frame):
    """Return the number of payload bits available in a single frame."""
    header = frame.get_header()
    header_len = 4
    crc_len = 2 if header.crc == 0 else 0
    if header.channel_mode == ChannelMode.Mono:
        si_len = 17 if header.mpeg_version == 1 else 9
    else:
        si_len = 32 if header.mpeg_version == 1 else 17
    payload_bytes = frame.frame_size - header_len - crc_len - si_len
    return payload_bytes * 8


def dry_run_build_location_map(frame_data, compensate_payload=True):
    info_indices = get_info_frame_indices(frame_data)
    location_map = []
    group_keys = []
    skip_short = 0
    skip_sf_overflow = 0
    skip_sample_overflow = 0

    for frame_idx, frame in enumerate(frame_data):
        if frame_idx in info_indices:
            continue

        header = frame.get_header()
        si = frame.side_info
        band_bounds = get_long_band_boundaries(header)

        for gr in range(2):
            for ch in range(header.channels):
                bt = int(si.block_type[gr][ch])
                ws = bool(si.window_switching[gr][ch])

                group_keys.append((frame_idx, gr, ch))

                # Condition 1: Short block interception
                if ws and bt == 2:
                    location_map.append(1)
                    skip_short += 1
                    continue

                k = get_k(si.scale_fac_scale[gr][ch])
                max_safe = MAX_SAFE_K2 if k == 2 else MAX_SAFE_K4
                s1, s2, lim1, lim2 = get_slen_limits(si, gr, ch)

                scalefac_l = si.scale_fac_l[gr][ch]
                samples = frame.raw_quantized_samples[gr][ch]

                skip = False
                skip_reason = None

                # Use scfsi-aware band iterator
                embeddable = list(get_embeddable_sfbs(si, gr, ch))

                # Condition 2 & 3: sf overflow & sample overflow
                for sfb, slen_bits, limit in embeddable:
                    sf_val = int(scalefac_l[sfb])

                    if sf_val >= limit:
                        skip = True
                        skip_reason = 'sf'
                        break

                    if not compensate_payload:
                        continue
                    bstart, bend = band_bounds[sfb]
                    band_max = 0
                    for idx in range(bstart, bend):
                        v = abs(int(samples[idx]))
                        if v > band_max:
                            band_max = v
                    if band_max > max_safe:
                        skip = True
                        skip_reason = 'sample'
                        break

                if skip:
                    location_map.append(1)
                    if skip_reason == 'sf':
                        skip_sf_overflow += 1
                    else:
                        skip_sample_overflow += 1
                    continue

                # All checks passed: safe group
                location_map.append(0)

    skip_counts = {'short_block': skip_short,
                   'sf_overflow': skip_sf_overflow,
                   'sample_overflow': skip_sample_overflow}
    return location_map, group_keys, skip_counts


def _simulate_reservoir(ordered_fi, frame_payload_bytes, frame_enc_bytes):
    reservoir = 0
    worst = 0
    for fi in ordered_fi:
        reservoir = min(511, reservoir + frame_payload_bytes[fi]
                        - frame_enc_bytes[fi])
        if reservoir < worst:
            worst = reservoir
    return worst


def reservoir_prune(frame_data, location_map, group_keys,
                    header_order="distortion", required_secret_bits=0,
                    compensate_payload=True, reservoir_pruning=True,
                    capacity_target_pruning=True,
                    prune_exact_candidates=DEFAULT_PRUNE_EXACT_CANDIDATES):
    """Prune carrier groups using exact coding costs and net-capacity checks."""
    info_indices = get_info_frame_indices(frame_data)

    # Ordered audio frame indices (excluding Info/Xing)
    ordered_fi = [fi for fi in range(len(frame_data))
                  if fi not in info_indices]

    # Map: frame_index → [group_indices …] for ALL audio groups
    frame_groups = {}
    for gi, (fi, gr, ch) in enumerate(group_keys):
        frame_groups.setdefault(fi, []).append(gi)

    safe_group_set = {gi for gi in range(len(group_keys))
                      if location_map[gi] == 0}
    if not safe_group_set:
        return {
            "pruned_groups": 0,
            "feasibility_pruned_groups": 0,
            "target_pruned_groups": 0,
            "reservoir_feasible": True,
            "gross_capacity_before_pruning": 0,
            "packet_capacity_before_pruning": 0,
            "net_secret_capacity_before_pruning": 0,
            "reservoir_feasible_gross_capacity": 0,
            "reservoir_feasible_packet_capacity": 0,
            "reservoir_feasible_net_secret_capacity": 0,
            "retained_gross_capacity": 0,
            "retained_packet_capacity": 0,
            "retained_net_secret_capacity": 0,
        }

    # Frame payload capacity in bytes
    frame_payload_bytes = {}
    for fi in ordered_fi:
        frame_payload_bytes[fi] = frame_payload_capacity(frame_data[fi]) // 8

    group_orig_bits = {}

    for fi in ordered_fi:
        frame = frame_data[fi]
        header = frame.get_header()
        si = frame.side_info
        for gi in frame_groups.get(fi, []):
            _, gr, ch = group_keys[gi]
            samples = frame.raw_quantized_samples[gr][ch]
            # Ensure Huffman tables match the *original* samples
            select_optimal_tables_for_granule(samples, si, header, gr, ch)
            tw = BitWriter()
            bits = encode_scale_factors(tw, si, header, gr, ch)
            bits += encode_huffman_samples(tw, samples, si, header, gr, ch)
            group_orig_bits[gi] = bits

    group_mod_bits = {}
    group_cost = {}
    group_benefit = {}
    group_ratio = {}
    group_distortion = {}

    for gi in safe_group_set:
        fi, gr, ch = group_keys[gi]
        frame = frame_data[fi]
        header = frame.get_header()
        si = frame.side_info
        bb = get_long_band_boundaries(header)

        k = get_k(si.scale_fac_scale[gr][ch])
        scalefac_l = si.scale_fac_l[gr][ch]
        samples = frame.raw_quantized_samples[gr][ch]
        big_value_end = int(si.big_value[gr][ch]) * 2

        # snapshot original state
        orig_sf = scalefac_l.copy()
        orig_samp = samples.copy()
        orig_tables = [int(si.table_select[gr][ch][r]) for r in range(3)]

        # count Benefit and simulate the configured payload transform
        benefit = 0
        for sfb, slen_bits, limit in get_embeddable_sfbs(si, gr, ch):
            sf_val = int(scalefac_l[sfb])
            if sf_val == HISTOGRAM_SHIFT_THRESHOLD:
                benefit += 1
            if sf_val >= HISTOGRAM_SHIFT_THRESHOLD and sf_val < limit:
                scalefac_l[sfb] = sf_val + 1
                if compensate_payload:
                    bstart, bend = bb[sfb]
                    forward_compensate_band(
                        samples, bstart, bend, k, big_value_end)

        # Re-select optimal Huffman tables for *modified* samples
        select_optimal_tables_for_granule(samples, si, header, gr, ch)

        tw = BitWriter()
        mod_bits = encode_scale_factors(tw, si, header, gr, ch)
        mod_bits += encode_huffman_samples(tw, samples, si, header, gr, ch)

        attenuation = 2.0 ** (-k / 4.0)
        distortion = 0.0
        for sfb, _, _ in get_embeddable_sfbs(si, gr, ch):
            bstart, bend = bb[sfb]
            for index in range(bstart, bend):
                original_magnitude = abs(int(orig_samp[index]))
                modified_magnitude = abs(int(samples[index]))
                if original_magnitude == 0 and modified_magnitude == 0:
                    continue
                original_value = original_magnitude ** (4.0 / 3.0)
                modified_value = (
                    modified_magnitude ** (4.0 / 3.0) * attenuation)
                distortion += (modified_value - original_value) ** 2

        # restore original state (no permanent side-effects)
        scalefac_l[:] = orig_sf
        samples[:] = orig_samp
        for r in range(3):
            si.table_select[gr][ch][r] = orig_tables[r]

        # store metrics
        group_mod_bits[gi] = mod_bits
        group_benefit[gi] = benefit
        group_distortion[gi] = distortion
        cost = mod_bits - group_orig_bits[gi]
        group_cost[gi] = cost

        if benefit == 0 and cost > 0:
            group_ratio[gi] = float('inf')   # No capacity gain; prune first
        elif cost <= 0:
            group_ratio[gi] = float('-inf')  # No coding expansion; keep
        else:
            group_ratio[gi] = cost / benefit

    active_bits = {}
    for gi in range(len(group_keys)):
        if gi in safe_group_set:
            active_bits[gi] = group_mod_bits[gi]
        else:
            active_bits[gi] = group_orig_bits[gi]

    # Precompute per-frame encoded bytes from group bits
    frame_enc_bytes = {}
    for fi in ordered_fi:
        total = sum(active_bits[gi] for gi in frame_groups.get(fi, []))
        frame_enc_bytes[fi] = (total + 7) // 8

    pruned_groups = []
    active_safe = set(safe_group_set)
    current_compressed_map = compress_location_map(
        stored_location_map(frame_data, location_map, group_keys))
    current_map_bits = len(current_compressed_map)
    current_gross_capacity = sum(group_benefit[gi] for gi in active_safe)
    header_positions = collect_header_sf_positions(frame_data, header_order)
    group_index = {key: gi for gi, key in enumerate(group_keys)}
    displaced_length_cache = {}

    def _net_secret_capacity(compressed_map, gross_capacity):
        header_size = len(compressed_map)
        if header_size > len(header_positions):
            return float('-inf')
        if header_size not in displaced_length_cache:
            displaced_length_cache[header_size] = len(pack_displaced_lsbs([
                int(frame_data[fi].side_info.scale_fac_l[gr][ch][sfb]) & 1
                for fi, gr, ch, sfb, _, _
                in header_positions[:header_size]
            ]))
        header_peaks = sum(
            location_map[group_index[(fi, gr, ch)]] == 0
            and int(frame_data[fi].side_info.scale_fac_l[gr][ch][sfb])
            == HISTOGRAM_SHIFT_THRESHOLD
            for fi, gr, ch, sfb, _, _ in header_positions[:header_size]
        )
        return (gross_capacity - header_peaks
                - displaced_length_cache[header_size])

    current_net_capacity = _net_secret_capacity(
        current_compressed_map, current_gross_capacity)
    gross_capacity_before_pruning = current_gross_capacity
    net_capacity_before_pruning = current_net_capacity
    if len(active_safe) > prune_exact_candidates:
        print(f"    Exact net-capacity shortlist: "
              f"{prune_exact_candidates}/{len(active_safe)} groups per step")

    while (reservoir_pruning
           and _simulate_reservoir(ordered_fi, frame_payload_bytes,
                                   frame_enc_bytes) < 0):
        best = None
        eligible = [gi for gi in active_safe if group_cost[gi] > 0]
        candidates = sorted(
            eligible,
            key=lambda gi: (
                group_benefit[gi] == 0,
                group_cost[gi] / max(group_benefit[gi], 1),
                group_cost[gi],
            ),
            reverse=True,
        )[:prune_exact_candidates]
        for gi in candidates:
            saved_bits = group_cost[gi]

            location_map[gi] = 1
            trial_compressed_map = compress_location_map(
                stored_location_map(frame_data, location_map, group_keys))
            trial_map_bits = len(trial_compressed_map)
            trial_gross_capacity = (
                current_gross_capacity - group_benefit[gi])
            trial_net_capacity = _net_secret_capacity(
                trial_compressed_map, trial_gross_capacity)
            location_map[gi] = 0

            net_loss = current_net_capacity - trial_net_capacity
            score = (1, saved_bits) if net_loss <= 0 else (
                0, saved_bits / net_loss)
            candidate = (score, saved_bits, -gi, gi, trial_map_bits,
                         trial_gross_capacity, trial_net_capacity)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            print("    WARNING: no positive-cost safe group remains but "
                  "the reservoir still overflows")
            break

        gi = best[3]

        # Prune: revert this group to its original (unmodified) cost
        location_map[gi] = 1
        active_bits[gi] = group_orig_bits[gi]
        active_safe.remove(gi)
        pruned_groups.append(gi)
        current_map_bits = best[4]
        current_gross_capacity = best[5]
        current_net_capacity = best[6]

        # Recompute only the affected frame's encoded bytes
        fi = group_keys[gi][0]
        total = sum(active_bits[g] for g in frame_groups.get(fi, []))
        frame_enc_bytes[fi] = (total + 7) // 8

    feasibility_pruned_groups = len(pruned_groups)
    reservoir_feasible = _simulate_reservoir(
        ordered_fi, frame_payload_bytes, frame_enc_bytes) >= 0
    reservoir_feasible_gross_capacity = (
        current_gross_capacity if reservoir_feasible else None)
    reservoir_feasible_net_capacity = (
        current_net_capacity if reservoir_feasible else None)

    surplus_pruned = 0
    while (capacity_target_pruning
           and reservoir_feasible
           and current_net_capacity >= required_secret_bits
           and active_safe):
        margin = current_net_capacity - required_secret_bits
        eligible = [
            gi for gi in active_safe
            if group_benefit[gi] <= margin + 1
            or group_benefit[gi] == 0
        ]
        candidates = sorted(
            eligible,
            key=lambda gi: (
                group_distortion[gi] / max(group_benefit[gi], 1),
                group_distortion[gi], -group_benefit[gi], group_cost[gi],
            ),
            reverse=True,
        )[:prune_exact_candidates]
        best = None
        for gi in candidates:
            fi = group_keys[gi][0]
            old_frame_bytes = frame_enc_bytes[fi]
            trial_total = sum(
                group_orig_bits[group]
                if group == gi else active_bits[group]
                for group in frame_groups.get(fi, []))
            frame_enc_bytes[fi] = (trial_total + 7) // 8
            reservoir_ok = _simulate_reservoir(
                ordered_fi, frame_payload_bytes, frame_enc_bytes) >= 0
            frame_enc_bytes[fi] = old_frame_bytes
            if not reservoir_ok:
                continue

            location_map[gi] = 1
            trial_compressed_map = compress_location_map(
                stored_location_map(frame_data, location_map, group_keys))
            trial_gross_capacity = (
                current_gross_capacity - group_benefit[gi])
            trial_net_capacity = _net_secret_capacity(
                trial_compressed_map, trial_gross_capacity)
            location_map[gi] = 0
            if trial_net_capacity < required_secret_bits:
                continue

            net_loss = current_net_capacity - trial_net_capacity
            score = (
                group_distortion[gi] / max(net_loss, 1),
                group_distortion[gi], -net_loss, group_cost[gi], -gi,
            )
            candidate = (score, gi, len(trial_compressed_map),
                         trial_gross_capacity, trial_net_capacity)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break

        gi = best[1]
        location_map[gi] = 1
        active_bits[gi] = group_orig_bits[gi]
        active_safe.remove(gi)
        pruned_groups.append(gi)
        surplus_pruned += 1
        current_map_bits = best[2]
        current_gross_capacity = best[3]
        current_net_capacity = best[4]
        fi = group_keys[gi][0]
        total = sum(active_bits[group]
                    for group in frame_groups.get(fi, []))
        frame_enc_bytes[fi] = (total + 7) // 8

    total_payload = sum(frame_payload_bytes[f] for f in ordered_fi)
    total_orig_bytes = sum(
        (sum(group_orig_bits[gi]
             for gi in frame_groups.get(fi, [])) + 7) // 8
        for fi in ordered_fi)
    remaining_cost_bits = sum(
        group_cost[gi] for gi in range(len(group_keys))
        if location_map[gi] == 0 and gi in group_cost)

    # Count prune reasons
    n_vampire = 0   # benefit==0, cost>0
    n_negative = 0  # cost<=0 (shouldn't be pruned, but count)
    n_normal = 0    # cost>0, benefit>0
    for gi in pruned_groups:
        r = group_ratio[gi]
        if r == float('inf'):
            n_vampire += 1
        elif r == float('-inf'):
            n_negative += 1
        else:
            n_normal += 1

    print(f"    Reservoir slack: {total_payload - total_orig_bytes} B, "
          f"compensation cost: {remaining_cost_bits} bits")
    print(f"    Pruned {len(pruned_groups)} groups "
          f"(vampire={n_vampire}, normal={n_normal}, free={n_negative})")
    print(f"    Capacity-target pruning: {surplus_pruned} groups")
    print(f"    Stored LM after pruning: {current_map_bits} bits")
    print(f"    Self-contained net capacity: {current_net_capacity} bits")

    return {
        "pruned_groups": len(pruned_groups),
        "feasibility_pruned_groups": feasibility_pruned_groups,
        "target_pruned_groups": surplus_pruned,
        "reservoir_feasible": reservoir_feasible,
        "gross_capacity_before_pruning": gross_capacity_before_pruning,
        "packet_capacity_before_pruning": net_capacity_before_pruning,
        "net_secret_capacity_before_pruning":
            max_secret_bits_for_packet_capacity(net_capacity_before_pruning),
        "reservoir_feasible_gross_capacity":
            reservoir_feasible_gross_capacity,
        "reservoir_feasible_packet_capacity":
            reservoir_feasible_net_capacity,
        "reservoir_feasible_net_secret_capacity": (
            max_secret_bits_for_packet_capacity(
                reservoir_feasible_net_capacity)
            if reservoir_feasible_net_capacity is not None else None),
        "retained_gross_capacity": current_gross_capacity,
        "retained_packet_capacity": current_net_capacity,
        "retained_net_secret_capacity":
            max_secret_bits_for_packet_capacity(current_net_capacity),
    }


def embed_payload(frame_data, location_map, group_keys, secret_bits,
                  header_order="distortion", compensate_header=True,
                  compensate_payload=True):
    """Embed the location map and payload with spectral compensation."""
    stored_lm = stored_location_map(frame_data, location_map, group_keys)
    compressed_lm = compress_location_map(stored_lm)
    M = len(compressed_lm)
    print(f"  Location Map: {len(location_map)} groups, "
          f"skipped={sum(location_map)}, "
          f"stored={len(stored_lm)}, compressed to {M} bits.")
    print_binary("Location Map (raw)", location_map)
    print_binary("Location Map (compressed)", compressed_lm)

    sf_positions = collect_header_sf_positions(frame_data, header_order)
    header_zone_size = M

    if len(sf_positions) < header_zone_size:
        raise RuntimeError(
            f"Not enough valid sf positions ({len(sf_positions)}) for "
            f"Header Zone ({header_zone_size}).")

    original_lsbs = []
    for i in range(header_zone_size):
        fi, gr, ch, sfb, slen_bits, _ = sf_positions[i]
        sf_val = int(frame_data[fi].side_info.scale_fac_l[gr][ch][sfb])
        original_lsbs.append(sf_val & 1)

    packed_original_lsbs = pack_displaced_lsbs(original_lsbs)
    secret_packet = pack_secret_bits(secret_bits)
    length_prefix_bits = len(secret_packet) - len(secret_bits)
    super_payload = packed_original_lsbs + secret_packet

    print_binary("original_lsbs", original_lsbs)
    print_binary("original_lsbs (packed)", packed_original_lsbs)
    print_binary("secret_bits", list(secret_bits))
    print(f"  Self-delimiting secret: {length_prefix_bits}-bit length prefix "
          f"+ {len(secret_bits)} data bits")
    print_binary("Super_Payload", super_payload)

    # Write the self-terminating LM and compensate deterministic +1s
    group_index = {key: i for i, key in enumerate(group_keys)}
    header_lsb_changes = 0
    header_plus1_total = 0
    header_plus1_compensated = 0
    header_nonzero_plus1_compensated = 0
    for i, bit in enumerate(compressed_lm):
        fi, gr, ch, sfb, _, _ = sf_positions[i]
        frame = frame_data[fi]
        si = frame.side_info
        sf_val = int(si.scale_fac_l[gr][ch][sfb])
        new_sf = (sf_val & ~1) | bit
        si.scale_fac_l[gr][ch][sfb] = new_sf
        header_lsb_changes += int(new_sf != sf_val)
        header_plus1_total += int(new_sf == sf_val + 1)

        key = (fi, gr, ch)
        if (not compensate_header or new_sf != sf_val + 1
                or location_map[group_index[key]] != 0
                or len(sf_position_consumers(frame_data, sf_positions[i])) != 1):
            continue
        start, end = get_long_band_boundaries(frame.get_header())[sfb]
        forward_compensate_band(
            frame.raw_quantized_samples[gr][ch], start, end,
            get_k(si.scale_fac_scale[gr][ch]),
            int(si.big_value[gr][ch]) * 2)
        header_plus1_compensated += 1
        header_nonzero_plus1_compensated += int(
            not sf_position_is_zero_band(frame_data, sf_positions[i]))

    header_zone_set = set()
    for i in range(header_zone_size):
        fi, gr, ch, sfb, _, _ = sf_positions[i]
        header_zone_set.add((fi, gr, ch, sfb))

    zero_header = sum(
        sf_position_is_zero_band(frame_data, sf_positions[i])
        for i in range(header_zone_size))
    print(f"  Header carriers: zero-band={zero_header}/{header_zone_size}, "
          f"LSB changes={header_lsb_changes}, +1 compensated="
          f"{header_plus1_compensated}/{header_plus1_total}")

    info_indices = get_info_frame_indices(frame_data)
    payload_idx = 0          # pointer into super_payload
    total_embedded = 0
    group_iter = 0           # index into group_keys / location_map

    for frame_idx, frame in enumerate(frame_data):
        if frame_idx in info_indices:
            continue

        header = frame.get_header()
        si = frame.side_info
        band_bounds = get_long_band_boundaries(header)

        for gr in range(2):
            for ch in range(header.channels):
                if group_iter >= len(location_map):
                    break
                lm_val = location_map[group_iter]
                group_iter += 1

                bt = int(si.block_type[gr][ch])
                ws = bool(si.window_switching[gr][ch])

                if ws and bt == 2:
                    continue
                if lm_val == 1:
                    continue

                # Safe group: shift + compensate
                k = get_k(si.scale_fac_scale[gr][ch])
                scalefac_l = si.scale_fac_l[gr][ch]
                samples = frame.raw_quantized_samples[gr][ch]
                big_value_end = int(si.big_value[gr][ch]) * 2

                for sfb, slen_bits, limit in get_embeddable_sfbs(si, gr, ch):
                    if (frame_idx, gr, ch, sfb) in header_zone_set:
                        continue

                    sf_val = int(scalefac_l[sfb])
                    T = HISTOGRAM_SHIFT_THRESHOLD
                    did_shift = False

                    if sf_val > T and sf_val < limit:
                        # Shift: sf += 1
                        scalefac_l[sfb] = sf_val + 1
                        did_shift = True

                    elif sf_val == T:
                        # Embed one bit
                        if payload_idx < len(super_payload):
                            bit = super_payload[payload_idx]
                            payload_idx += 1
                            if bit == 1:
                                scalefac_l[sfb] = T + 1
                                did_shift = True
                            total_embedded += 1

                    if did_shift and compensate_payload:
                        bstart, bend = band_bounds[sfb]
                        forward_compensate_band(
                            samples, bstart, bend, k, big_value_end)
                select_optimal_tables_for_granule(samples, si, header, gr, ch)

    if payload_idx < len(super_payload):
        print(f"  WARNING: Only embedded {payload_idx}/{len(super_payload)} "
              f"Super_Payload bits (capacity exhausted).")
    else:
        print(f"  Embedded full Super_Payload: {len(super_payload)} bits "
              f"(secret_packet={len(secret_packet)}, "
              f"displaced_lsbs={len(packed_original_lsbs)}/"
              f"{len(original_lsbs)}).")

    return {
        "total_embedded": total_embedded,
        "header_lsb_changes": header_lsb_changes,
        "header_plus1_total": header_plus1_total,
        "header_plus1_compensated": header_plus1_compensated,
        "header_nonzero_plus1_compensated":
            header_nonzero_plus1_compensated,
        "displaced_lsb_bits": len(packed_original_lsbs),
        "secret_packet_bits": len(secret_packet),
        "length_prefix_bits": length_prefix_bits,
    }


def file_hash(path):
    """Return the SHA-256 hex-digest of a file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def str_to_bits(s):
    """Convert a UTF-8 string to a list of bits."""
    bits = []
    for byte in s.encode('utf-8'):
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_str(bits):
    """Convert complete bytes from a bit list back to UTF-8."""
    values = []
    for start in range(0, len(bits) - 7, 8):
        value = 0
        for bit in bits[start:start + 8]:
            value = (value << 1) | bit
        values.append(value)
    return bytes(values).decode('utf-8', errors='replace')


def embed(input_file, T_value, output_folder,
          secret_message="Hello, data hidding.", header_order="distortion",
          compensate_header=True, compensate_payload=True,
          reservoir_pruning=True, capacity_target_pruning=True,
          prune_exact_candidates=DEFAULT_PRUNE_EXACT_CANDIDATES,
          original_data=None, parser=None, frame_data=None,
          id3_offset=None):
    """Embed the secret; supplied frame_data is modified in place."""
    global HISTOGRAM_SHIFT_THRESHOLD
    HISTOGRAM_SHIFT_THRESHOLD = T_value

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "out_rdh.mp3")

    # Secret message to embed
    secret_bits = str_to_bits(secret_message)
    print(f"Secret message : \"{secret_message}\"")
    print(f"Secret length  : {len(secret_bits)} bits")
    print(f"T value        : {T_value}")

    if frame_data is not None and original_data is not None \
            and parser is not None and id3_offset is not None:
        num_frames = len(frame_data)
        print(f"Reusing pre-decoded frames ({num_frames} frames, "
              f"audio offset = {id3_offset} bytes).")
    else:
        with open(input_file, 'rb') as f:
            original_data = bytearray(f.read())

        decoder = Decoder(input_file, "temp.wav")
        parser, frame_data, num_frames = decoder.decode(quiet=True)

        id3_offset = parser.audio_offset

        print(f"Decoded {num_frames} frames (audio offset = {id3_offset} bytes).")

    print("\n--- Phase 1a: Dry Run (Conditions 1-3) ---")
    location_map, group_keys, skip_counts = dry_run_build_location_map(
        frame_data, compensate_payload=True)
    safe_count = location_map.count(0)
    print(f"  Groups: {len(location_map)} total, {safe_count} safe, "
          f"{len(location_map) - safe_count} skipped "
          f"(short={skip_counts['short_block']}, "
          f"sf_ovf={skip_counts['sf_overflow']}, "
          f"samp_ovf={skip_counts['sample_overflow']})")

    print("\n--- Phase 1b: Bit Reservoir Simulation (Condition 4) ---")
    secret_packet_preview = pack_secret_bits(secret_bits)
    pruning_stats = reservoir_prune(
        frame_data, location_map, group_keys, header_order,
        required_secret_bits=len(secret_packet_preview),
        compensate_payload=True,
        reservoir_pruning=reservoir_pruning,
        capacity_target_pruning=capacity_target_pruning,
        prune_exact_candidates=prune_exact_candidates)
    safe_count = location_map.count(0)
    print(f"  Final: {safe_count} safe groups (all 100% compensatable)")

    stored_lm_preview = stored_location_map(frame_data, location_map, group_keys)
    compressed_lm_preview = compress_location_map(stored_lm_preview)
    M = len(compressed_lm_preview)
    header_zone_size = M
    sf_positions = collect_header_sf_positions(frame_data, header_order)
    if len(sf_positions) < header_zone_size:
        print(f"\n  ERROR: Header needs {header_zone_size} sf positions, "
              f"but only {len(sf_positions)} are available. Aborting.")
        return {
            "success": False,
            "reason": "header_capacity_insufficient",
            "header_zone_size": header_zone_size,
            "sf_positions": len(sf_positions),
            "safe_groups": safe_count,
            "all_groups": len(location_map),
            "skip_counts": skip_counts,
            "pruned_groups": pruning_stats["pruned_groups"],
            "pruning_stats": pruning_stats,
            "compensate_header": compensate_header,
            "compensate_payload": compensate_payload,
            "selection_models_payload_compensation": True,
            "reservoir_pruning": reservoir_pruning,
            "capacity_target_pruning": capacity_target_pruning,
            "prune_exact_candidates": prune_exact_candidates,
        }
    header_zone_set = {
        (fi, gr, ch, sfb)
        for fi, gr, ch, sfb, _, _ in sf_positions[:header_zone_size]
    }

    # Exact Payload-Zone capacity (# of sf==T outside Header carriers).
    info_indices = get_info_frame_indices(frame_data)
    capacity_bits = 0
    gi = 0
    for frame_idx, frame in enumerate(frame_data):
        if frame_idx in info_indices:
            continue
        header = frame.get_header()
        si = frame.side_info
        for gr in range(2):
            for ch in range(header.channels):
                if gi >= len(location_map):
                    break
                if location_map[gi] == 0:
                    bt = int(si.block_type[gr][ch])
                    ws = bool(si.window_switching[gr][ch])
                    if not (ws and bt == 2):
                        for sfb, sb, lim in get_embeddable_sfbs(si, gr, ch):
                            if ((frame_idx, gr, ch, sfb) not in header_zone_set
                                    and int(si.scale_fac_l[gr][ch][sfb])
                                    == HISTOGRAM_SHIFT_THRESHOLD):
                                capacity_bits += 1
                gi += 1

    zero_header = sum(
        sf_position_is_zero_band(frame_data, position)
        for position in sf_positions[:header_zone_size])
    original_lsbs_preview = [
        int(frame_data[fi].side_info.scale_fac_l[gr][ch][sfb]) & 1
        for fi, gr, ch, sfb, _, _ in sf_positions[:header_zone_size]
    ]
    packed_original_lsbs_preview = pack_displaced_lsbs(
        original_lsbs_preview)
    displaced_lsb_bits = len(packed_original_lsbs_preview)
    print(f"  Stored LM: {len(stored_lm_preview)}/{len(location_map)} flags, "
          f"compressed to {M} bits")
    print(f"  Header Zone: {header_zone_size} sf LSBs "
          f"(zero-band={zero_header}, available={len(sf_positions)})")

    length_prefix_bits = len(secret_packet_preview) - len(secret_bits)
    super_payload_len = displaced_lsb_bits + len(secret_packet_preview)
    print(f"  Capacity: {capacity_bits} bits, "
          f"Super_Payload: {super_payload_len} bits")
    print(f"  Secret packet: {len(secret_bits)} data + "
          f"{length_prefix_bits} self-delimiting length bits")
    print(f"  Displaced Header LSBs: {header_zone_size} raw -> "
          f"{displaced_lsb_bits} packed bits")

    if capacity_bits < super_payload_len:
        print(f"\n  ERROR: Capacity ({capacity_bits}) < "
              f"Super_Payload ({super_payload_len}). Aborting.")
        return {
            "success": False,
            "reason": "capacity_insufficient",
            "capacity": capacity_bits,
            "super_payload": super_payload_len,
            "embedded_bits": len(secret_bits),
            "secret_packet_bits": len(secret_packet_preview),
            "length_prefix_bits": length_prefix_bits,
            "header_zone_size": header_zone_size,
            "displaced_lsb_bits": displaced_lsb_bits,
            "packet_capacity": capacity_bits - displaced_lsb_bits,
            "net_secret_capacity": max_secret_bits_for_packet_capacity(
                capacity_bits - displaced_lsb_bits),
            "stored_groups": len(stored_lm_preview),
            "header_zero_bands": zero_header,
            "header_fallback_bands": header_zone_size - zero_header,
            "safe_groups": safe_count,
            "all_groups": len(location_map),
            "skip_counts": skip_counts,
            "pruned_groups": pruning_stats["pruned_groups"],
            "pruning_stats": pruning_stats,
            "compensate_header": compensate_header,
            "compensate_payload": compensate_payload,
            "selection_models_payload_compensation": True,
            "reservoir_pruning": reservoir_pruning,
            "capacity_target_pruning": capacity_target_pruning,
            "prune_exact_candidates": prune_exact_candidates,
        }

    print("\n--- Phase 2: Embedding (sf shift + joint compensation) ---")
    embed_stats = embed_payload(
        frame_data, location_map, group_keys, secret_bits,
        header_order, compensate_header, compensate_payload)

    print("\n--- Phase 3: Encoding ---")
    pack_mp3(original_data, frame_data, id3_offset, output_file)

    orig_hash = file_hash(input_file)
    out_hash = file_hash(output_file)

    print()
    print("=" * 60)
    print("Embedding Complete")
    print("=" * 60)
    print(f"  Original file  : {orig_hash}  ({input_file})")
    print(f"  Stego output   : {out_hash}  ({output_file})")
    print(f"  Secret message : \"{secret_message}\"")
    print(f"  Bits embedded  : {len(secret_bits)}")
    print(f"  Payload comp.  : "
          f"{'enabled' if compensate_payload else 'disabled'}")
    orig_size = os.path.getsize(input_file)
    out_size = os.path.getsize(output_file)
    print(f"  File sizes     : {orig_size} → {out_size} bytes "
          f"(delta = {out_size - orig_size})")

    return {
        "success": True,
        "output_file": output_file,
        "output_folder": output_folder,
        "capacity": capacity_bits,
        "retained_gross_capacity":
            pruning_stats["retained_gross_capacity"],
        "reservoir_feasible_gross_capacity":
            pruning_stats["reservoir_feasible_gross_capacity"],
        "reservoir_feasible_net_capacity":
            pruning_stats["reservoir_feasible_net_secret_capacity"],
        "packet_capacity": capacity_bits - displaced_lsb_bits,
        "net_secret_capacity": max_secret_bits_for_packet_capacity(
            capacity_bits - displaced_lsb_bits),
        "super_payload": super_payload_len,
        "secret_packet_bits": embed_stats["secret_packet_bits"],
        "length_prefix_bits": embed_stats["length_prefix_bits"],
        "header_zone_size": header_zone_size,
        "stored_groups": len(stored_lm_preview),
        "header_zero_bands": zero_header,
        "header_fallback_bands": header_zone_size - zero_header,
        "displaced_lsb_bits": embed_stats["displaced_lsb_bits"],
        "header_lsb_changes": embed_stats["header_lsb_changes"],
        "header_plus1_total": embed_stats["header_plus1_total"],
        "header_plus1_compensated":
            embed_stats["header_plus1_compensated"],
        "header_nonzero_plus1_compensated":
            embed_stats["header_nonzero_plus1_compensated"],
        "embedded_bits": len(secret_bits),
        "secret_message": secret_message,
        "safe_groups": safe_count,
        "all_groups": len(location_map),
        "skip_counts": skip_counts,
        "pruned_groups": pruning_stats["pruned_groups"],
        "pruning_stats": pruning_stats,
        "compensate_header": compensate_header,
        "compensate_payload": compensate_payload,
        "selection_models_payload_compensation": True,
        "reservoir_pruning": reservoir_pruning,
        "capacity_target_pruning": capacity_target_pruning,
        "prune_exact_candidates": prune_exact_candidates,
        "delta_bytes": out_size - orig_size,
    }


def main():
    result = embed(
        input_file="input/test3.mp3",
        T_value=0,
        output_folder="output",
        secret_message="Hello, data hidding.",
    )
    return result


if __name__ == "__main__":
    main()
