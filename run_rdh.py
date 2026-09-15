"""Embed, extract, and verify canonical-cover recovery."""

import os
import sys
import argparse
import hashlib
import subprocess

from mp3_encoder import MP3PackingError, reconstruct_mp3, identity_func


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_carrier(input_file, output_file, bitrate_kbps=256):
    """Create the deterministic, reservoir-free MP3 carrier profile."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_file), "-vn", "-map_metadata", "-1",
        "-ac", "2", "-ar", "44100", "-codec:a", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k", "-reservoir", "0", "-write_xing", "0",
        str(output_file),
    ]
    subprocess.run(command, check=True)
    return command


def run(input_file, T_value, output_folder,
        secret_message="Hello, data hidding.", header_order="distortion",
        compensate_header=True, allow_canonical_fallback=False,
        compensate_payload=True, reservoir_pruning=True,
        capacity_target_pruning=True, prune_exact_candidates=32):
    """Embed, extract, and return recovery verification results."""
    import embed_rdh
    import extract_recover_rdh

    os.makedirs(output_folder, exist_ok=True)
    source_input = input_file
    carrier_input = input_file
    canonicalization_command = None
    canonicalization_bitrate_kbps = None

    def use_canonical_carrier(reason, bitrate_kbps):
        nonlocal carrier_input, canonicalization_command
        nonlocal canonicalization_bitrate_kbps
        carrier_input = os.path.join(output_folder, "canonical_carrier.mp3")
        print(f"  Canonical carrier fallback ({bitrate_kbps} kb/s): {reason}")
        canonicalization_command = canonicalize_carrier(
            source_input, carrier_input, bitrate_kbps)
        canonicalization_bitrate_kbps = bitrate_kbps

    reencoded_ref = os.path.join(output_folder, "out_reencoded.mp3")
    print("=" * 60)
    print(f"Step 0: Re-encode baseline → {reencoded_ref}")
    print("=" * 60)
    try:
        decoded = reconstruct_mp3(
            carrier_input, reencoded_ref, modify_func=identity_func,
            return_decoded=True)
    except MP3PackingError as error:
        if not allow_canonical_fallback:
            print(f"  Strict carrier rejection: {error}")
            return {
                "success": False,
                "reason": "carrier_layout_unsupported",
                "error": str(error),
                "source_input_file": source_input,
                "carrier_input_file": carrier_input,
                "carrier_canonicalized": False,
                "canonicalization_command": None,
                "canonicalization_bitrate_kbps": None,
            }
        use_canonical_carrier(str(error), 256)
        decoded = reconstruct_mp3(
            carrier_input, reencoded_ref, modify_func=identity_func,
            return_decoded=True)
    original_data, parser, frame_data, id3_offset = decoded
    print(f"  Baseline hash: {file_hash(reencoded_ref)}")

    print("\n" + "=" * 60)
    print(f"Step 1: Embed (T={T_value})")
    print("=" * 60)
    embed_result = embed_rdh.embed(
        input_file=carrier_input,
        T_value=T_value,
        output_folder=output_folder,
        secret_message=secret_message,
        header_order=header_order,
        compensate_header=compensate_header,
        compensate_payload=compensate_payload,
        reservoir_pruning=reservoir_pruning,
        capacity_target_pruning=capacity_target_pruning,
        prune_exact_candidates=prune_exact_candidates,
        original_data=original_data, parser=parser,
        frame_data=frame_data, id3_offset=id3_offset,
    )
    retry_reasons = {"capacity_insufficient", "header_capacity_insufficient"}
    while (embed_result is not None
           and not embed_result.get("success", False)
           and embed_result.get("reason") in retry_reasons
           and allow_canonical_fallback
           and canonicalization_bitrate_kbps != 320):
        next_bitrate = 256 if canonicalization_bitrate_kbps is None else 320
        use_canonical_carrier(embed_result["reason"], next_bitrate)
        original_data, parser, frame_data, id3_offset = reconstruct_mp3(
            carrier_input, reencoded_ref, modify_func=identity_func,
            return_decoded=True)
        embed_result = embed_rdh.embed(
            input_file=carrier_input,
            T_value=T_value,
            output_folder=output_folder,
            secret_message=secret_message,
            header_order=header_order,
            compensate_header=compensate_header,
            compensate_payload=compensate_payload,
            reservoir_pruning=reservoir_pruning,
            capacity_target_pruning=capacity_target_pruning,
            prune_exact_candidates=prune_exact_candidates,
            original_data=original_data, parser=parser,
            frame_data=frame_data, id3_offset=id3_offset,
        )
    if embed_result is None or not embed_result.get("success", False):
        print("\n*** EMBED FAILED ***")
        if isinstance(embed_result, dict):
            embed_result["source_input_file"] = source_input
            embed_result["carrier_input_file"] = carrier_input
            embed_result["carrier_canonicalized"] = (
                canonicalization_command is not None)
            embed_result["canonicalization_command"] = (
                canonicalization_command)
            embed_result["canonicalization_bitrate_kbps"] = (
                canonicalization_bitrate_kbps)
        return embed_result if embed_result else False

    stego_file = embed_result["output_file"]

    print("\n" + "=" * 60)
    print(f"Step 2: Extract & Recover")
    print("=" * 60)
    extract_result = extract_recover_rdh.extract(
        stego_file=stego_file,
        original_file=carrier_input,
        output_folder=output_folder,
        reencoded_ref=reencoded_ref,
        header_order=header_order,
        compensate_header=compensate_header,
        compensate_payload=compensate_payload,
    )
    if extract_result is None:
        print("\n*** EXTRACT FAILED ***")
        return False

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Source input   : {source_input}")
    print(f"  Carrier input  : {carrier_input}")
    print(f"  T value        : {T_value}")
    print(f"  Output folder  : {output_folder}")
    print(f"  Stego file     : {stego_file}")
    print(f"  Recovered file : {extract_result['recovered_file']}")
    print(f"  Embedded msg   : \"{embed_result['secret_message']}\"")
    print(f"  Extracted msg  : \"{extract_result['secret_message']}\"")
    print(f"  Capacity       : {embed_result['capacity']} bits")
    print(f"  Embedded bits  : {embed_result['embedded_bits']}")
    print(f"  Safe groups    : {embed_result['safe_groups']}")
    print(f"  Size delta     : {embed_result['delta_bytes']} bytes")

    extracted_msg = extract_result['secret_message'] or ""
    msg_match = (embed_result['secret_message'] == extracted_msg)
    rec_match = extract_result['recovery_match']

    print()
    print(f"  Message match  : {'PASS' if msg_match else 'FAIL'}")
    print(f"  Recovery match : {'PASS' if rec_match else 'FAIL' if rec_match is False else 'N/A'}")

    ok = msg_match and (rec_match is True)
    print()
    print(f"  *** {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} ***")

    embed_result["ok"] = ok
    embed_result["msg_match"] = msg_match
    embed_result["recovery_match"] = rec_match
    embed_result["source_input_file"] = source_input
    embed_result["carrier_input_file"] = carrier_input
    embed_result["carrier_canonicalized"] = (
        canonicalization_command is not None)
    embed_result["canonicalization_command"] = canonicalization_command
    embed_result["canonicalization_bitrate_kbps"] = (
        canonicalization_bitrate_kbps)
    return embed_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDH embed + extract + verify")
    parser.add_argument("--input", default="input/2.mp3", help="Input MP3 file (default: input/test.mp3)")
    parser.add_argument("--T", type=int, default=0, help="Histogram shift threshold T (default: 0)")
    parser.add_argument("--output_folder", default="output_test", help="Output folder (default: output_test)")
    parser.add_argument("--secret", default="Hello, data hidding. Hello, Monash SOIT. Hi World.", help="Secret message")
    parser.add_argument("--header-order",
                        choices=("distortion", "energy", "nonzero", "file"),
                        default="distortion")
    parser.add_argument("--no-header-compensation", action="store_true")
    parser.add_argument("--no-payload-compensation", action="store_true")
    parser.add_argument("--no-reservoir-pruning", action="store_true")
    parser.add_argument("--no-capacity-target-pruning", action="store_true")
    parser.add_argument("--prune-exact-candidates", type=int, default=32)
    parser.add_argument(
        "--allow-canonical-fallback", action="store_true",
        help=("explicitly permit 256/320 kb/s reservoir-free transcoding "
              "when the input carrier is unsupported or too small"))
    args = parser.parse_args()

    result = run(args.input, args.T, args.output_folder, args.secret,
                 args.header_order, not args.no_header_compensation,
                 args.allow_canonical_fallback,
                 not args.no_payload_compensation,
                 not args.no_reservoir_pruning,
                 not args.no_capacity_target_pruning,
                 args.prune_exact_candidates)
    ok = result.get("ok", False) if isinstance(result, dict) else bool(result)
    sys.exit(0 if ok else 1)
