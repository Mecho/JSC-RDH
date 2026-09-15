"""Blind extraction and canonical-cover recovery."""

import os
import math
import hashlib
import numpy as np

from mp3_decoder.decoder.decoder import Decoder
from mp3_decoder.decoder.FrameHeader import ChannelMode
from mp3_encoder import pack_mp3

# Re-use shared helpers from the embedder
from embed_rdh import (
    MAX_QUANTIZED_VALUE, HISTOGRAM_SHIFT_THRESHOLD,
    FORWARD_MULT_K2, FORWARD_MULT_K4,
    get_k, get_long_band_boundaries, get_embeddable_sfbs,
    decompress_location_map, collect_header_sf_positions,
    collect_group_keys, is_short_group, restore_location_map,
    sf_position_consumers, unpack_displaced_lsbs, unpack_secret_bits,
    get_info_frame_indices, bits_to_str, print_binary,
)


def _build_inverse_lut(mult):
    lut = {}
    for old in range(MAX_QUANTIZED_VALUE + 1):
        new = round(old * mult)
        if new > MAX_QUANTIZED_VALUE:
            break                       # no need to go further
        if new not in lut:
            lut[new] = old              # first (smallest) old wins
    return lut


# Module-level construction (executed once at import time)
INVERSE_LUT_K2 = _build_inverse_lut(FORWARD_MULT_K2)
INVERSE_LUT_K4 = _build_inverse_lut(FORWARD_MULT_K4)


def inverse_compensate_band(samples, band_start, band_end, k,
                            big_value_end=576,
                            frame_idx=-1, gr=-1, ch=-1, sfb=-1):
    """Restore big_value magnitudes in place using the inverse lookup table."""
    lut = INVERSE_LUT_K2 if k == 2 else INVERSE_LUT_K4
    actual_end = min(band_end, big_value_end, len(samples))
    for i in range(band_start, actual_end):
        val = int(samples[i])
        if val == 0:
            continue
        sign = 1 if val > 0 else -1
        abs_val = abs(val)
        if abs_val in lut:
            samples[i] = sign * lut[abs_val]
        else:
            print(f"  WARNING: inverse LUT miss |s|={abs_val} k={k} "
                  f"idx={i} frame={frame_idx} gr={gr} ch={ch} sfb={sfb}")


def read_header_zone(frame_data, header_order="distortion"):
    """Decode the location map and restore implicit short-block flags."""
    group_keys = collect_group_keys(frame_data)
    stored_groups = sum(not is_short_group(frame_data, key)
                        for key in group_keys)
    if stored_groups == 0:
        return 0, [], 0, [1] * len(group_keys)

    sf_positions = collect_header_sf_positions(frame_data, header_order)
    candidate_bits = [
        int(frame_data[fi].side_info.scale_fac_l[gr][ch][sfb]) & 1
        for fi, gr, ch, sfb, _, _ in sf_positions
    ]
    stored_lm, consumed = decompress_location_map(
        candidate_bits, stored_groups, return_consumed=True)
    if consumed > len(sf_positions):
        raise RuntimeError("Self-terminating Header exceeds sf candidates.")
    return consumed, candidate_bits[:consumed], consumed, restore_location_map(
        frame_data, stored_lm)


def extract_and_recover(frame_data, location_map, header_zone_size,
                        super_payload_length, header_order="distortion",
                        compensate_payload=True):
    """Extract payload bits and reverse scalefactor shifts and compensation."""
    info_indices = get_info_frame_indices(frame_data)
    sf_positions = collect_header_sf_positions(frame_data, header_order)

    # Build set of Header Zone (frame_idx, gr, ch, sfb) to skip
    header_zone_set = set()
    for i in range(header_zone_size):
        fi, gr, ch, sfb, _, _ = sf_positions[i]
        header_zone_set.add((fi, gr, ch, sfb))

    super_payload = []
    group_iter = 0

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

                # Safe group: ALL are inverse-compensated
                k = get_k(si.scale_fac_scale[gr][ch])
                scalefac_l = si.scale_fac_l[gr][ch]
                samples = frame.raw_quantized_samples[gr][ch]
                big_value_end = int(si.big_value[gr][ch]) * 2

                T = HISTOGRAM_SHIFT_THRESHOLD

                for sfb, slen_bits, limit in get_embeddable_sfbs(si, gr, ch):
                    # Header Zone bands are recovered separately.
                    if (frame_idx, gr, ch, sfb) in header_zone_set:
                        continue

                    sf_val = int(scalefac_l[sfb])
                    bstart, bend = band_bounds[sfb]

                    if sf_val == T + 1:
                        if len(super_payload) < super_payload_length:
                            # Extract bit '1', restore sf to T
                            super_payload.append(1)
                            scalefac_l[sfb] = T
                            if compensate_payload:
                                inverse_compensate_band(
                                    samples, bstart, bend, k, big_value_end,
                                    frame_idx, gr, ch, sfb)
                        else:
                            # Beyond payload: shifted sf, restore
                            scalefac_l[sfb] = sf_val - 1
                            if compensate_payload:
                                inverse_compensate_band(
                                    samples, bstart, bend, k, big_value_end,
                                    frame_idx, gr, ch, sfb)

                    elif sf_val == T:
                        if len(super_payload) < super_payload_length:
                            # Extract bit '0', sf stays at T
                            super_payload.append(0)

                    elif sf_val > T + 1:
                        # Shifted sf: restore sf -= 1
                        scalefac_l[sfb] = sf_val - 1
                        if compensate_payload:
                            inverse_compensate_band(
                                samples, bstart, bend, k, big_value_end,
                                frame_idx, gr, ch, sfb)

    return super_payload


def recover_header_zone(frame_data, original_lsbs, header_zone_size,
                        location_map, header_order="distortion",
                        compensate_header=True):
    sf_positions = collect_header_sf_positions(frame_data, header_order)
    group_index = {
        key: i for i, key in enumerate(collect_group_keys(frame_data))
    }
    inverse_count = 0

    for i in range(header_zone_size):
        fi, gr, ch, sfb, _, _ = sf_positions[i]
        frame = frame_data[fi]
        si = frame.side_info
        sf_val = int(si.scale_fac_l[gr][ch][sfb])
        if (compensate_header and original_lsbs[i] == 0
                and (sf_val & 1) == 1
                and location_map[group_index[(fi, gr, ch)]] == 0
                and len(sf_position_consumers(frame_data,
                                              sf_positions[i])) == 1):
            start, end = get_long_band_boundaries(frame.get_header())[sfb]
            inverse_compensate_band(
                frame.raw_quantized_samples[gr][ch], start, end,
                get_k(si.scale_fac_scale[gr][ch]),
                int(si.big_value[gr][ch]) * 2, fi, gr, ch, sfb)
            inverse_count += 1
        # Restore original LSB
        sf_val = (sf_val & ~1) | original_lsbs[i]
        si.scale_fac_l[gr][ch][sfb] = sf_val
    return inverse_count


def file_hash(path):
    """Return the SHA-256 hex-digest of a file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract(stego_file, original_file, output_folder, reencoded_ref=None,
            header_order="distortion", compensate_header=True,
            compensate_payload=True):
    """Extract the secret and recover the canonical cover."""
    # Sync T from embed_rdh (caller should set embed_rdh.HISTOGRAM_SHIFT_THRESHOLD)
    import embed_rdh
    global HISTOGRAM_SHIFT_THRESHOLD
    HISTOGRAM_SHIFT_THRESHOLD = embed_rdh.HISTOGRAM_SHIFT_THRESHOLD

    os.makedirs(output_folder, exist_ok=True)
    recovered_file = os.path.join(output_folder, "out_recovered.mp3")

    if not os.path.exists(stego_file):
        print(f"ERROR: Stego file '{stego_file}' not found. "
              f"Run embed_rdh.py first.")
        return None

    with open(stego_file, 'rb') as f:
        stego_data = bytearray(f.read())

    decoder = Decoder(stego_file, "temp_extract.wav")
    parser, frame_data, num_frames = decoder.decode(quiet=True)

    id3_offset = parser.audio_offset

    print(f"Decoded {num_frames} frames from stego file.")
    print(f"  Inverse LUT k=2 entries: {len(INVERSE_LUT_K2)}")
    print(f"  Inverse LUT k=4 entries: {len(INVERSE_LUT_K4)}")

    print("\n--- Step 1: Read Header Zone & Location Map ---")
    M, compressed_lm, header_zone_size, location_map = read_header_zone(
        frame_data, header_order)
    print(f"  M = {M} bits (compressed LM length)")
    print(f"  Header Zone size = {header_zone_size} sf LSBs")

    safe_count = location_map.count(0)
    skip_count = location_map.count(1)
    print(f"  Location Map: {len(location_map)} groups, "
          f"{safe_count} safe, {skip_count} skipped.")
    print(f"  Payload compensation: "
          f"{'enabled' if compensate_payload else 'disabled'}")
    print_binary("Location Map (compressed)", compressed_lm)
    print_binary("Location Map (decompressed)", location_map)

    print("\n--- Step 2: Extract & Recover Payload Zone ---")

    # Pass a large number so extract_and_recover collects all bits
    max_bits = 999999
    super_payload = extract_and_recover(frame_data, location_map,
                                        header_zone_size,
                                        max_bits, header_order,
                                        compensate_payload)
    print(f"  Extracted bits: {len(super_payload)}")
    print_binary("Super_Payload (extracted)", super_payload)

    print("\n--- Step 3: Closed-loop Recovery ---")

    original_lsbs, displaced_bits = unpack_displaced_lsbs(
        super_payload, header_zone_size)
    secret_bits, secret_packet_bits = unpack_secret_bits(
        super_payload[displaced_bits:])
    length_prefix_bits = secret_packet_bits - len(secret_bits)
    trailing_bits = len(super_payload) - displaced_bits - secret_packet_bits

    print_binary("original_lsbs", original_lsbs)
    print(f"  Compressed displaced LSBs: {displaced_bits} bits")
    print(f"  Self-delimiting length prefix: {length_prefix_bits} bits")
    print_binary("secret_bits", secret_bits)
    print(f"  Secret data   : {len(secret_bits)} bits")
    print(f"  Ignored tail  : {trailing_bits} recovery-only bits")
    print(f"  Original LSBs : {len(original_lsbs)} bits")

    # Decode secret message
    secret_message = None
    if len(secret_bits) >= 8:
        secret_message = bits_to_str(secret_bits)
        print(f"  Extracted secret message: \"{secret_message}\"")
    else:
        print(f"  Secret bits (raw): {secret_bits}")

    # Restore Header Zone LSBs for perfect file recovery
    header_inverse = recover_header_zone(
        frame_data, original_lsbs, header_zone_size, location_map,
        header_order, compensate_header)
    print(f"  Header Zone restored (+1 inverse-compensated: "
          f"{header_inverse}).")

    print("\n--- Encoding recovered MP3 ---")
    pack_mp3(stego_data, frame_data, id3_offset, recovered_file)

    stego_hash = file_hash(stego_file)
    recovered_hash = file_hash(recovered_file)

    print()
    print("=" * 60)
    print("Extraction & Recovery Complete")
    print("=" * 60)
    print(f"  Stego file     : {stego_hash}  ({stego_file})")
    print(f"  Recovered file : {recovered_hash}  ({recovered_file})")

    recovery_match = None
    if os.path.exists(original_file):
        orig_hash = file_hash(original_file)
        print(f"  Original file  : {orig_hash}  ({original_file})")

        # For a true bit-exact check, compare against the re-encoded baseline
        if reencoded_ref and os.path.exists(reencoded_ref):
            ref_hash = file_hash(reencoded_ref)
            recovery_match = (recovered_hash == ref_hash)
            match_str = "MATCH" if recovery_match else "MISMATCH"
            print(f"  Re-encoded ref : {ref_hash}  ({reencoded_ref})")
            print(f"  Recovery check : {match_str}")
        else:
            print(f"  (No re-encoded reference provided for bit-exact check.)")

    orig_size = os.path.getsize(stego_file)
    rec_size = os.path.getsize(recovered_file)
    print(f"  File sizes     : stego={orig_size}, recovered={rec_size} bytes")

    return {
        "recovered_file": recovered_file,
        "secret_message": secret_message,
        "secret_bits": secret_bits,
        "secret_bit_length": len(secret_bits),
        "secret_packet_bits": secret_packet_bits,
        "length_prefix_bits": length_prefix_bits,
        "trailing_recovery_bits": trailing_bits,
        "recovery_match": recovery_match,
        "recovered_hash": recovered_hash,
    }


def main():
    result = extract(
        stego_file="output/out_rdh.mp3",
        original_file="input/test.mp3",
        output_folder="output",
        reencoded_ref="output/out_reencoded.mp3",
    )
    return result


if __name__ == "__main__":
    main()
