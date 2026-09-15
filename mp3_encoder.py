"""Huffman encoding and fixed-frame MP3 repacking."""

import os
import numpy as np
from mp3_decoder.decoder.decoder import Decoder
from mp3_decoder.decoder.FrameHeader import ChannelMode
from mp3_decoder.decoder.tables import (
    big_value_table, big_value_max, big_value_linbit, quad_table_1, slen
)


class MP3PackingError(RuntimeError):
    """Raised when re-encoded main data cannot fit the fixed frame layout."""


def validate_main_data_layout(payload_lengths, frame_bit_ranges,
                              info_frame_indices=()):
    """Raise when byte-aligned main_data cannot fit the fixed MP3 frames."""
    info_frame_indices = set(info_frame_indices)
    audio_start = 0
    for frame_idx, payload_len in enumerate(payload_lengths):
        if frame_idx not in info_frame_indices:
            break
        audio_start += payload_len

    write_byte_pos = audio_start
    cumulative_payload = 0
    for frame_idx, payload_len in enumerate(payload_lengths):
        if frame_idx in info_frame_indices:
            cumulative_payload += payload_len
            continue
        if cumulative_payload - write_byte_pos > 511:
            write_byte_pos = cumulative_payload - 511
        if write_byte_pos > cumulative_payload:
            deficit = write_byte_pos - cumulative_payload
            raise MP3PackingError(
                f"frame {frame_idx}: fixed layout is short by {deficit} "
                "main_data bytes")
        start_bit, end_bit = frame_bit_ranges[frame_idx]
        write_byte_pos += (end_bit - start_bit + 7) // 8
        cumulative_payload += payload_len
    if write_byte_pos > cumulative_payload:
        raise MP3PackingError(
            f"end of stream: fixed layout is short by "
            f"{write_byte_pos - cumulative_payload} main_data bytes")


class BitWriter:
    """Bit-level writer for constructing MP3 bitstreams."""
    
    def __init__(self):
        self.data = bytearray()
        self.current_byte = 0
        self.bits_in_byte = 0

    def write_bits(self, value, num_bits):
        """Write bits from MSB to LSB."""
        if num_bits == 0:
            return
        # Ensure value fits in num_bits
        value = value & ((1 << num_bits) - 1)
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.current_byte = (self.current_byte << 1) | bit
            self.bits_in_byte += 1
            if self.bits_in_byte == 8:
                self.data.append(self.current_byte)
                self.current_byte = 0
                self.bits_in_byte = 0

    def flush(self):
        """Pad with zeros to complete the current byte."""
        if self.bits_in_byte > 0:
            self.current_byte = self.current_byte << (8 - self.bits_in_byte)
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bits_in_byte = 0
            
    def get_bytes(self):
        return bytes(self.data)

    def bit_count(self):
        return len(self.data) * 8 + self.bits_in_byte
    
    def byte_align(self):
        """Align to byte boundary by padding with zeros."""
        if self.bits_in_byte > 0:
            self.flush()


# Build Huffman encoding lookup tables
def build_huffman_encode_table(table_num):
    table = big_value_table[table_num]
    max_val = big_value_max[table_num]
    
    if max_val == 0:
        return {}
    
    encode_map = {}
    for row in range(max_val):
        for col in range(max_val):
            idx = 2 * (row * max_val + col)
            if idx + 1 < len(table):
                code_left = table[idx]      # Left-aligned 32-bit
                length = table[idx + 1]
                if length > 0:
                    # Right-align the code
                    code = code_left >> (32 - length)
                    encode_map[(row, col)] = (code, length)
    return encode_map


def build_quad_encode_table():
    """Build encoding table for count1 quadruples."""
    qt = quad_table_1
    encode_map = {}
    for i in range(len(qt.value)):
        vals = tuple(qt.value[i])
        code_left = qt.h_cod[i]
        length = qt.h_len[i]
        if length > 0:
            code = code_left >> (32 - length)
            encode_map[vals] = (code, length)
    return encode_map


# Pre-build all encoding tables
HUFF_ENCODE_TABLES = {i: build_huffman_encode_table(i) for i in range(32)}
QUAD_ENCODE_TABLE = build_quad_encode_table()

VALID_BIG_VALUE_TABLES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


def calculate_table_bits(samples, start_idx, end_idx, table_num):
    if table_num == 0:
        # Table 0 can only encode zeros
        for i in range(start_idx, end_idx, 2):
            x = abs(int(samples[i]))
            y = abs(int(samples[i + 1])) if i + 1 < len(samples) else 0
            if x != 0 or y != 0:
                return float('inf'), False
        return 0, True
    
    encode_table = HUFF_ENCODE_TABLES.get(table_num, {})
    if not encode_table:
        return float('inf'), False
    
    table_max = big_value_max[table_num]
    linbits = big_value_linbit[table_num]
    
    if table_max == 0:
        return float('inf'), False
    
    total_bits = 0
    
    for i in range(start_idx, end_idx, 2):
        x = int(samples[i])
        y = int(samples[i + 1]) if i + 1 < len(samples) else 0
        abs_x, abs_y = abs(x), abs(y)
        
        # Check if values can be encoded
        if linbits == 0:
            # No linbits, values must fit in table
            if abs_x >= table_max or abs_y >= table_max:
                return float('inf'), False
            x_code, y_code = abs_x, abs_y
        else:
            # With linbits
            limit = table_max - 1
            x_code = min(abs_x, limit)
            y_code = min(abs_y, limit)
            
            # Add linbits
            if abs_x >= limit:
                total_bits += linbits
            if abs_y >= limit:
                total_bits += linbits
        
        # Get Huffman code length
        if (x_code, y_code) not in encode_table:
            return float('inf'), False
        
        _, length = encode_table[(x_code, y_code)]
        total_bits += length
        
        # Add sign bits
        if x != 0:
            total_bits += 1
        if y != 0:
            total_bits += 1
    
    return total_bits, True


def select_optimal_huffman_table(samples, start_idx, end_idx):
    if start_idx >= end_idx:
        return 0
    
    # Find max absolute value in region
    max_val = 0
    all_zero = True
    for i in range(start_idx, end_idx):
        if i < len(samples):
            val = abs(int(samples[i]))
            if val > max_val:
                max_val = val
            if val != 0:
                all_zero = False
    
    if all_zero:
        return 0
    
    best_table = 0
    best_bits = float('inf')
    
    for table_num in VALID_BIG_VALUE_TABLES:
        if table_num == 0:
            continue
        
        table_max = big_value_max[table_num]
        linbits = big_value_linbit[table_num]
        
        # Quick check: can this table handle the max value?
        if linbits == 0 and max_val >= table_max:
            continue
        if linbits > 0 and max_val > (table_max - 1) + (1 << linbits) - 1:
            continue
        
        bits, can_encode = calculate_table_bits(samples, start_idx, end_idx, table_num)
        
        if can_encode and bits < best_bits:
            best_bits = bits
            best_table = table_num
    
    return best_table


def select_optimal_tables_for_granule(samples, side_info, header, gr, ch):
    big_value = int(side_info.big_value[gr][ch])
    
    if big_value == 0:
        return [0, 0, 0]
    
    # Get region boundaries
    if side_info.window_switching[gr][ch] and side_info.block_type[gr][ch] == 2:
        region0_end = 36
        region1_end = 576
    else:
        region0_count = int(side_info.region0_count[gr][ch])
        region1_count = int(side_info.region1_count[gr][ch])
        
        region0_end = header.band_index.long_win[region0_count + 1] if region0_count + 1 < len(header.band_index.long_win) else 576
        idx = region0_count + 1 + region1_count + 1
        region1_end = header.band_index.long_win[idx] if idx < len(header.band_index.long_win) else 576
    
    big_value_end = big_value * 2
    
    # Select optimal table for each region
    region0_end = min(region0_end, big_value_end)
    region1_end = min(region1_end, big_value_end)
    
    table0 = select_optimal_huffman_table(samples, 0, region0_end)
    table1 = select_optimal_huffman_table(samples, region0_end, region1_end)
    table2 = select_optimal_huffman_table(samples, region1_end, big_value_end)
    
    # Update side_info
    side_info.table_select[gr][ch][0] = table0
    side_info.table_select[gr][ch][1] = table1
    side_info.table_select[gr][ch][2] = table2
    
    return [table0, table1, table2]


def encode_side_info(writer, side_info, header, new_part2_3_lengths=None):
    start_bits = writer.bit_count()
    
    # main_data_begin (9 bits)
    writer.write_bits(int(side_info.main_data_begin), 9)
    
    # private_bits (5 bits for mono, 3 bits for stereo)
    if header.channel_mode == ChannelMode.Mono:
        writer.write_bits(0, 5)  # private bits
    else:
        writer.write_bits(0, 3)  # private bits
    
    # scfsi (scale factor selection info) - 4 bits per channel
    for ch in range(header.channels):
        for scfsi_band in range(4):
            writer.write_bits(int(side_info.scfsi[ch][scfsi_band]), 1)
    
    # Granule/channel specific info
    for gr in range(2):
        for ch in range(header.channels):
            # part2_3_length (12 bits) - use new value if provided
            if new_part2_3_lengths and (gr, ch) in new_part2_3_lengths:
                p23_len = new_part2_3_lengths[(gr, ch)]
            else:
                p23_len = int(side_info.part2_3_length[gr][ch])
            writer.write_bits(p23_len, 12)
            
            # big_value (9 bits)
            writer.write_bits(int(side_info.big_value[gr][ch]), 9)
            
            # global_gain (8 bits)
            writer.write_bits(int(side_info.global_gain[gr][ch]), 8)
            
            # scalefac_compress (4 bits)
            writer.write_bits(int(side_info.scale_fac_compress[gr][ch]), 4)
            
            # window_switching_flag (1 bit)
            window_switching = int(side_info.window_switching[gr][ch])
            writer.write_bits(window_switching, 1)
            
            if window_switching:
                # block_type (2 bits)
                writer.write_bits(int(side_info.block_type[gr][ch]), 2)
                
                # mixed_block_flag (1 bit)
                writer.write_bits(int(side_info.mixed_block_flag[gr][ch]), 1)
                
                # table_select for 2 regions (5 bits each)
                for region in range(2):
                    writer.write_bits(int(side_info.table_select[gr][ch][region]), 5)
                
                # subblock_gain for 3 windows (3 bits each)
                for window in range(3):
                    writer.write_bits(int(side_info.sub_block_gain[gr][ch][window]), 3)
            else:
                # table_select for 3 regions (5 bits each)
                for region in range(3):
                    writer.write_bits(int(side_info.table_select[gr][ch][region]), 5)
                
                # region0_count (4 bits)
                writer.write_bits(int(side_info.region0_count[gr][ch]), 4)
                
                # region1_count (3 bits)
                writer.write_bits(int(side_info.region1_count[gr][ch]), 3)
            
            # preflag (1 bit)
            writer.write_bits(int(side_info.pre_flag[gr][ch]), 1)
            
            # scalefac_scale (1 bit)
            writer.write_bits(int(side_info.scale_fac_scale[gr][ch]), 1)
            
            # count1table_select (1 bit)
            writer.write_bits(int(side_info.count1table_select[gr][ch]), 1)
    
    return writer.bit_count() - start_bits


def encode_scale_factors(writer, side_info, header, gr, ch):
    start_bits = writer.bit_count()
    
    slen1 = int(side_info.slen1[gr][ch])
    slen2 = int(side_info.slen2[gr][ch])
    
    block_type = int(side_info.block_type[gr][ch])
    window_switching = side_info.window_switching[gr][ch]
    mixed_block = side_info.mixed_block_flag[gr][ch]
    
    if window_switching and block_type == 2:
        # Short or mixed blocks
        if mixed_block:
            # Mixed blocks: 8 long bands + short bands
            for sfb in range(8):
                val = int(side_info.scale_fac_l[gr][ch][sfb])
                writer.write_bits(val, slen1)
            for sfb in range(3, 6):
                for window in range(3):
                    val = int(side_info.scale_fac_s[gr][ch][window][sfb])
                    writer.write_bits(val, slen1)
            for sfb in range(6, 12):
                for window in range(3):
                    val = int(side_info.scale_fac_s[gr][ch][window][sfb])
                    writer.write_bits(val, slen2)
        else:
            # Pure short blocks
            for sfb in range(6):
                for window in range(3):
                    val = int(side_info.scale_fac_s[gr][ch][window][sfb])
                    writer.write_bits(val, slen1)
            for sfb in range(6, 12):
                for window in range(3):
                    val = int(side_info.scale_fac_s[gr][ch][window][sfb])
                    writer.write_bits(val, slen2)
    else:
        # Long blocks
        if gr == 0:
            # Granule 0: always write all scale factors
            for sfb in range(11):
                val = int(side_info.scale_fac_l[gr][ch][sfb])
                writer.write_bits(val, slen1)
            for sfb in range(11, 21):
                val = int(side_info.scale_fac_l[gr][ch][sfb])
                writer.write_bits(val, slen2)
        else:
            # Granule 1: check scfsi for reuse
            SB = [6, 11, 16, 21]
            PREV_SB = [0, 6, 11, 16]
            
            # Bands 0-5, 6-10 use slen1
            for i in range(2):
                if not side_info.scfsi[ch][i]:
                    for sfb in range(PREV_SB[i], SB[i]):
                        val = int(side_info.scale_fac_l[gr][ch][sfb])
                        writer.write_bits(val, slen1)
            
            # Bands 11-15, 16-20 use slen2
            for i in range(2, 4):
                if not side_info.scfsi[ch][i]:
                    for sfb in range(PREV_SB[i], SB[i]):
                        val = int(side_info.scale_fac_l[gr][ch][sfb])
                        writer.write_bits(val, slen2)
    
    return writer.bit_count() - start_bits


def encode_huffman_samples(writer, samples, side_info, header, gr, ch):
    start_bits = writer.bit_count()
    
    big_value = int(side_info.big_value[gr][ch])
    
    # Get region boundaries
    if side_info.window_switching[gr][ch] and side_info.block_type[gr][ch] == 2:
        region0 = 36
        region1 = 576
    else:
        region0 = header.band_index.long_win[int(side_info.region0_count[gr][ch]) + 1]
        idx = int(side_info.region0_count[gr][ch]) + 1 + int(side_info.region1_count[gr][ch]) + 1
        if idx < len(header.band_index.long_win):
            region1 = header.band_index.long_win[idx]
        else:
            region1 = 576
    
    # Encode big_value region (pairs)
    sample_idx = 0
    while sample_idx < big_value * 2:
        # Select table based on region
        if sample_idx < region0:
            table_num = int(side_info.table_select[gr][ch][0])
        elif sample_idx < region1:
            table_num = int(side_info.table_select[gr][ch][1])
        else:
            table_num = int(side_info.table_select[gr][ch][2])
        
        x = int(samples[sample_idx])
        y = int(samples[sample_idx + 1]) if sample_idx + 1 < 576 else 0
        
        if table_num == 0:
            # Table 0: no encoding, samples must be 0
            sample_idx += 2
            continue
        
        abs_x, abs_y = abs(x), abs(y)
        linbits = big_value_linbit[table_num]
        table_max = big_value_max[table_num]
        
        # Determine code values and linbits
        x_code, y_code = abs_x, abs_y
        x_lin, y_lin = 0, 0
        
        if linbits > 0:
            limit = table_max - 1
            if abs_x >= limit:
                x_code = limit
                x_lin = abs_x - limit
            if abs_y >= limit:
                y_code = limit
                y_lin = abs_y - limit
        else:
            # Clamp to table max
            x_code = min(x_code, table_max - 1)
            y_code = min(y_code, table_max - 1)
        
        # Write Huffman code
        encode_table = HUFF_ENCODE_TABLES[table_num]
        if (x_code, y_code) in encode_table:
            code, length = encode_table[(x_code, y_code)]
            writer.write_bits(code, length)
        else:
            # This shouldn't happen with valid data
            print(f"Warning: No code for ({x_code}, {y_code}) in table {table_num}")
        
        # Write x linbits and sign (interleaved order: x_lin, x_sign, y_lin, y_sign)
        if linbits > 0 and x_code == table_max - 1:
            writer.write_bits(x_lin, linbits)
        if x != 0:
            writer.write_bits(1 if x < 0 else 0, 1)
        
        # Write y linbits and sign
        if linbits > 0 and y_code == table_max - 1:
            writer.write_bits(y_lin, linbits)
        if y != 0:
            writer.write_bits(1 if y < 0 else 0, 1)
        
        sample_idx += 2
    
    last_nz = 575
    while last_nz >= sample_idx and samples[last_nz] == 0:
        last_nz -= 1
    
    while sample_idx <= last_nz and sample_idx + 3 < 576:
        v = [int(samples[sample_idx + i]) for i in range(4)]
        abs_v = [abs(x) for x in v]
        
        if side_info.count1table_select[gr][ch] == 1:
            # Table B: 4 bits, 1=zero, 0=non-zero
            val_bits = 0
            for i in range(4):
                bit = 1 if abs_v[i] == 0 else 0
                val_bits |= (bit << (3 - i))
            writer.write_bits(val_bits, 4)
        else:
            # Table A: Huffman coded
            key = tuple(abs_v)
            if key in QUAD_ENCODE_TABLE:
                code, length = QUAD_ENCODE_TABLE[key]
                writer.write_bits(code, length)
            else:
                print(f"Warning: No quad code for {key}")
        
        # Write sign bits for non-zero values
        for i in range(4):
            if v[i] != 0:
                writer.write_bits(1 if v[i] < 0 else 0, 1)
        
        sample_idx += 4
    
    return writer.bit_count() - start_bits


def encode_granule_channel(writer, samples, side_info, header, gr, ch):
    start_bits = writer.bit_count()
    
    # Encode scale factors
    encode_scale_factors(writer, side_info, header, gr, ch)
    
    # Encode Huffman samples
    encode_huffman_samples(writer, samples, side_info, header, gr, ch)
    
    return writer.bit_count() - start_bits


def pack_mp3(original_data, frame_data, id3_offset, output_file):
    """Repack decoded frames within their original physical budgets."""
    info_frame_indices = set()
    for fi, fr in enumerate(frame_data):
        hdr = fr.get_header()
        si = fr.side_info
        if all(int(si.part2_3_length[g][c]) == 0
               for g in range(2) for c in range(hdr.channels)):
            info_frame_indices.add(fi)

    frame_new_lengths = []        # List of dicts {(gr, ch): new_length}
    frame_modified_samples = []   # {(gr, ch): np.ndarray} per frame

    for frame_idx, frame in enumerate(frame_data):
        if frame_idx in info_frame_indices:
            frame_new_lengths.append({})
            frame_modified_samples.append({})
            continue

        header = frame.get_header()
        side_info = frame.side_info

        new_lengths = {}
        modified_samples_frame = {}

        for gr in range(2):
            for ch in range(header.channels):
                samples = frame.raw_quantized_samples[gr][ch].copy()
                modified_samples_frame[(gr, ch)] = samples

                # Select optimal Huffman tables for the (possibly modified) samples
                select_optimal_tables_for_granule(samples, side_info, header, gr, ch)

                # Calculate actual encoding length (without padding)
                test_writer = BitWriter()
                sf_bits = encode_scale_factors(test_writer, side_info, header, gr, ch)
                huff_bits = encode_huffman_samples(test_writer, samples, side_info, header, gr, ch)
                actual_length = sf_bits + huff_bits

                new_lengths[(gr, ch)] = actual_length

        frame_new_lengths.append(new_lengths)
        frame_modified_samples.append(modified_samples_frame)

    all_main_data = BitWriter()
    frame_main_data_info = []  # (start_bit, end_bit) for each frame

    current_offset = id3_offset

    for frame_idx, frame in enumerate(frame_data):
        if frame_idx in info_frame_indices:
            frame_main_data_info.append((all_main_data.bit_count(), all_main_data.bit_count()))
            current_offset += frame.frame_size
            continue

        header = frame.get_header()
        side_info = frame.side_info
        modified_samples = frame_modified_samples[frame_idx]

        frame_start_bit = all_main_data.bit_count()

        for gr in range(2):
            for ch in range(header.channels):
                samples = modified_samples[(gr, ch)]
                encode_granule_channel(
                    all_main_data, samples, side_info, header, gr, ch
                )

        frame_end_bit = all_main_data.bit_count()
        frame_main_data_info.append((frame_start_bit, frame_end_bit))

        current_offset += frame.frame_size

    # Byte-align the final stream
    all_main_data.byte_align()
    main_data_bytes = all_main_data.get_bytes()

    payload_lengths = []
    for frame in frame_data:
        header = frame.get_header()
        header_len = 4
        crc_len = 2 if header.crc == 0 else 0
        si_len = (17 if header.mpeg_version == 1 else 9) \
            if header.channel_mode == ChannelMode.Mono \
            else (32 if header.mpeg_version == 1 else 17)
        payload_lengths.append(frame.frame_size - header_len - crc_len - si_len)

    validate_main_data_layout(
        payload_lengths, frame_main_data_info, info_frame_indices)

    with open(output_file, 'wb') as f:
        # Copy ID3v2 tag
        if id3_offset > 0:
            f.write(bytes(original_data[:id3_offset]))

        # Collect payload geometry for every frame
        payload_info = []
        temp_offset = id3_offset

        for frame_idx, frame in enumerate(frame_data):
            header = frame.get_header()
            frame_size = frame.frame_size

            header_len = 4
            crc_len = 2 if header.crc == 0 else 0

            if header.channel_mode == ChannelMode.Mono:
                si_len = 17 if header.mpeg_version == 1 else 9
            else:
                si_len = 32 if header.mpeg_version == 1 else 17

            payload_start = header_len + crc_len + si_len
            payload_len = frame_size - payload_start

            payload_info.append({
                'file_offset': temp_offset,
                'frame_size': frame_size,
                'payload_start': payload_start,
                'payload_len': payload_len,
                'main_data_begin': int(frame.side_info.main_data_begin),
                'new_part2_3_lengths': frame_new_lengths[frame_idx]
            })

            temp_offset += frame_size

        # Create a continuous buffer for all payloads
        total_payload_size = sum(p['payload_len'] for p in payload_info)
        payload_buffer = bytearray(total_payload_size)


        # Pass 1: copy Info/Xing payloads into their fixed positions
        cumulative_payload = 0
        for frame_idx in range(len(frame_data)):
            info = payload_info[frame_idx]
            if frame_idx in info_frame_indices:
                orig_payload = original_data[
                    info['file_offset'] + info['payload_start']:
                    info['file_offset'] + info['frame_size']
                ]
                payload_buffer[cumulative_payload:cumulative_payload + info['payload_len']] = orig_payload
            cumulative_payload += info['payload_len']

        audio_payload_byte_start = 0
        for fi in range(len(frame_data)):
            if fi not in info_frame_indices:
                break
            audio_payload_byte_start += payload_info[fi]['payload_len']

        write_byte_pos = audio_payload_byte_start  # next write position (byte)
        cumulative_payload = 0

        for frame_idx, frame in enumerate(frame_data):
            info = payload_info[frame_idx]

            if frame_idx in info_frame_indices:
                cumulative_payload += info['payload_len']
                continue

            frame_start_bit, frame_end_bit = frame_main_data_info[frame_idx]
            frame_bits = frame_end_bit - frame_start_bit

            if cumulative_payload - write_byte_pos > 511:
                write_byte_pos = cumulative_payload - 511
            if write_byte_pos > cumulative_payload:
                raise MP3PackingError(
                    "preflight invariant failed during reservoir packing")

            new_mdb = cumulative_payload - write_byte_pos
            frame.side_info.main_data_begin = new_mdb

            frame_byte_count = (frame_bits + 7) // 8
            src_byte_start = frame_start_bit // 8
            src_bit_offset = frame_start_bit % 8
            dst_start = write_byte_pos

            if src_bit_offset == 0:
                # Byte-aligned source: fast slice copy
                copy_len = min(frame_byte_count,
                               len(main_data_bytes) - src_byte_start,
                               len(payload_buffer) - dst_start)
                if copy_len > 0:
                    payload_buffer[dst_start:dst_start + copy_len] = \
                        main_data_bytes[src_byte_start:src_byte_start + copy_len]
            else:
                # Non-aligned source: shift-combine two adjacent bytes
                for i in range(frame_byte_count):
                    dst_idx = dst_start + i
                    if dst_idx >= len(payload_buffer):
                        break
                    si = src_byte_start + i
                    hi = main_data_bytes[si] if si < len(main_data_bytes) else 0
                    lo = main_data_bytes[si + 1] if si + 1 < len(main_data_bytes) else 0
                    payload_buffer[dst_idx] = ((hi << src_bit_offset) | (lo >> (8 - src_bit_offset))) & 0xFF

            # Advance write position to the next byte boundary
            write_byte_pos += (frame_bits + 7) // 8
            cumulative_payload += info['payload_len']

        # Write frames with the filled payload buffer
        payload_buffer_offset = 0

        for frame_idx, frame in enumerate(frame_data):
            info = payload_info[frame_idx]
            header = frame.get_header()
            side_info = frame.side_info

            # Info/Xing frame: copy byte-for-byte from original
            if frame_idx in info_frame_indices:
                frame_bytes = original_data[info['file_offset']:info['file_offset'] + info['frame_size']]
                f.write(bytes(frame_bytes))
                payload_buffer_offset += info['payload_len']
                continue

            # Build the frame from scratch
            frame_writer = BitWriter()

            # 1. Copy header (4 bytes) from original: structure unchanged
            header_bytes = original_data[info['file_offset']:info['file_offset'] + 4]
            for b in header_bytes:
                frame_writer.write_bits(b, 8)

            # 2. Copy CRC if present (2 bytes)
            crc_len = 2 if header.crc == 0 else 0
            if crc_len > 0:
                crc_bytes = original_data[info['file_offset'] + 4:info['file_offset'] + 6]
                for b in crc_bytes:
                    frame_writer.write_bits(b, 8)

            # 3. Encode side info (use new part2_3_length if computed)
            new_part2_3_lengths = info.get('new_part2_3_lengths', None)
            encode_side_info(frame_writer, side_info, header, new_part2_3_lengths)

            # 4. Write payload (main_data)
            payload_data = payload_buffer[payload_buffer_offset:payload_buffer_offset + info['payload_len']]
            for b in payload_data:
                frame_writer.write_bits(b, 8)

            frame_writer.flush()
            f.write(frame_writer.get_bytes())

            payload_buffer_offset += info['payload_len']

        # Copy any trailing data after the last frame (e.g., ID3v1 tag)
        last_frame_end = payload_info[-1]['file_offset'] + payload_info[-1]['frame_size']
        if last_frame_end < len(original_data):
            f.write(bytes(original_data[last_frame_end:]))

    print(f"Wrote {len(frame_data)} frames to {output_file}")
    print(f"Output size: {os.path.getsize(output_file)} bytes")


def reconstruct_mp3(input_file, output_file, modify_func=None,
                     return_decoded=False):
    """Decode, apply optional modifications, and repack the file."""
    # Read original file
    with open(input_file, 'rb') as f:
        original_data = bytearray(f.read())

    # Decode
    decoder = Decoder(input_file, "temp.wav")
    parser, frame_data, num_frames = decoder.decode(quiet=True)

    print(f"Decoded {num_frames} frames.")

    id3_offset = parser.audio_offset

    # If no modification, just copy
    if modify_func is None:
        with open(output_file, 'wb') as f:
            f.write(original_data)
        print(f"Direct copy to {output_file}")
        if return_decoded:
            return original_data, parser, frame_data, id3_offset
        return

    # Apply modify_func to each granule/channel in-place
    for frame_idx, frame in enumerate(frame_data):
        header = frame.get_header()
        side_info = frame.side_info

        # Skip Info/Xing frames (all part2_3_length == 0)
        if all(int(side_info.part2_3_length[g][c]) == 0
               for g in range(2) for c in range(header.channels)):
            continue

        for gr in range(2):
            for ch in range(header.channels):
                samples = frame.raw_quantized_samples[gr][ch].copy()
                scalefac_l = side_info.scale_fac_l[gr][ch].copy()
                scalefac_s = side_info.scale_fac_s[gr][ch].copy()

                block_type = int(side_info.block_type[gr][ch])
                window_switching = bool(side_info.window_switching[gr][ch])
                mixed_block_flag = bool(side_info.mixed_block_flag[gr][ch])

                result = modify_func(
                    frame_idx, gr, ch,
                    samples, scalefac_l, scalefac_s,
                    block_type, window_switching, mixed_block_flag
                )
                mod_samples, mod_scalefac_l, mod_scalefac_s = result

                if mod_samples is not None:
                    frame.raw_quantized_samples[gr][ch][:] = mod_samples
                if mod_scalefac_l is not None:
                    side_info.scale_fac_l[gr][ch][:] = mod_scalefac_l
                if mod_scalefac_s is not None:
                    side_info.scale_fac_s[gr][ch][:] = mod_scalefac_s

    # Delegate to the pure packer
    pack_mp3(original_data, frame_data, id3_offset, output_file)

    if return_decoded:
        return original_data, parser, frame_data, id3_offset


def identity_func(frame_idx, gr, ch, samples, scalefac_l, scalefac_s,
                  block_type, window_switching, mixed_block_flag):
    """No modification - for testing / baseline re-encode."""
    return samples, None, None


def example_data_hiding_func(frame_idx, gr, ch, samples, scalefac_l, scalefac_s,
                             block_type, window_switching, mixed_block_flag):
    """Embed demo bits in sample and scalefactor LSBs."""
    # demo payload: bits of the ASCII string "HIDE"
    payload_bytes = b"HIDE"
    payload_bits = []
    for byte in payload_bytes:
        for shift in range(7, -1, -1):
            payload_bits.append((byte >> shift) & 1)

    def get_bit(global_bit_idx):
        """Cyclic payload for demo purposes."""
        return payload_bits[global_bit_idx % len(payload_bits)]

    # Deterministic bit offset per granule/channel
    bit_offset = (frame_idx * 4 + gr * 2 + ch) * 32

    # embed into quantized samples (LSB of non-zero values)
    mod_samples = samples.copy()
    bit_idx = bit_offset
    for i in range(len(mod_samples)):
        if mod_samples[i] != 0:
            val = int(mod_samples[i])
            sign = -1 if val < 0 else 1
            abs_val = abs(val)
            abs_val = (abs_val & ~1) | get_bit(bit_idx)
            mod_samples[i] = sign * abs_val
            bit_idx += 1

    sf_bit_idx = bit_offset + 576  # separate offset to avoid collision with samples

    # Determine which scalefactor arrays are actually encoded for this block
    is_short = window_switching and block_type == 2
    is_mixed = is_short and mixed_block_flag
    uses_long = (not is_short) or is_mixed
    uses_short = is_short

    # embed into long block scalefactors
    mod_scalefac_l = None
    if uses_long:
        mod_scalefac_l = scalefac_l.copy()
        for sfb in range(len(mod_scalefac_l)):
            if mod_scalefac_l[sfb] != 0:
                mod_scalefac_l[sfb] = (int(mod_scalefac_l[sfb]) & ~1) | get_bit(sf_bit_idx)
                sf_bit_idx += 1

    # embed into short block scalefactors
    mod_scalefac_s = None
    if uses_short:
        mod_scalefac_s = scalefac_s.copy()
        for window in range(3):
            for sfb in range(mod_scalefac_s.shape[1]):
                if mod_scalefac_s[window][sfb] != 0:
                    mod_scalefac_s[window][sfb] = (int(mod_scalefac_s[window][sfb]) & ~1) | get_bit(sf_bit_idx)
                    sf_bit_idx += 1

    return mod_samples, mod_scalefac_l, mod_scalefac_s


if __name__ == "__main__":
    import hashlib

    input_file = "input/test_long.mp3"

    def file_hash(path):
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    print("=" * 60)
    print("Test 1: Direct copy (no modification)")
    print("=" * 60)
    reconstruct_mp3(input_file, "output/out_direct.mp3", modify_func=None)

    print("\n" + "=" * 60)
    print("Test 2: Re-encode without modification (identity)")
    print("=" * 60)
    reconstruct_mp3(input_file, "output/out_reencoded.mp3", modify_func=identity_func)

    print("\n" + "=" * 60)
    print("Test 3: Re-encode with data hiding (samples + scalefactors)")
    print("=" * 60)
    reconstruct_mp3(input_file, "output/out_hidden.mp3", modify_func=example_data_hiding_func)

    print("\n" + "=" * 60)
    print("File Hashes:")
    print("=" * 60)
    print(f"Original:    {file_hash(input_file)}")
    print(f"Direct copy: {file_hash('output/out_direct.mp3')}")
    print(f"Re-encoded:  {file_hash('output/out_reencoded.mp3')}")
    print(f"Data hidden: {file_hash('output/out_hidden.mp3')}")
