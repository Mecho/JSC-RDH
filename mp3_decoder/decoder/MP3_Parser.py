from ast import List
from scipy.io.wavfile import write
from tqdm import tqdm
import copy

from mp3_decoder.decoder.Frame import *

HEADER_SIZE = 4


class MP3Parser:

    def __init__(self, file_data: list, offset: int, wav_file_path: str):
        # Declarations
        self.__curr_frame: Frame = Frame()
        self.__valid: bool = False
        # List of integers that contain the file (without ID3) data
        self.__file_data: list = []
        self.__buffer: list = []
        self.__file_length: int = 0
        # self.__file_path = file_path
        self.__wav_file_path: str = wav_file_path

        actual_offset = offset
        if actual_offset < len(file_data) and not (
                file_data[actual_offset] == 0xFF
                and actual_offset + 1 < len(file_data)
                and file_data[actual_offset + 1] >= 0xE0):
            scan_limit = min(len(file_data) - 1, actual_offset + 4096)
            for i in range(actual_offset, scan_limit):
                if file_data[i] == 0xFF and file_data[i + 1] >= 0xE0:
                    actual_offset = i
                    break

        buf_end = min(actual_offset + 4096, len(file_data))
        self.__buffer: list = file_data[actual_offset:buf_end]

        if self.__buffer[0] == 0xFF and self.__buffer[1] >= 0xE0:
            self.__valid: bool = True
            self.__file_data: list = file_data
            self.__file_length: int = len(file_data)
            self.__offset: int = actual_offset
            self.__init_curr_header()
            self.__curr_frame.set_frame_size()
        else:
            self.__valid: bool = False

        self.frame_data: list = []
        self.side_info: list = []
        self.output_bits: str = ""
        self.__audio_offset: int = actual_offset

    @property
    def audio_offset(self):
        """Actual byte offset where the first audio frame begins."""
        return self.__audio_offset

    def __init_curr_header(self):
        if self.__buffer[0] == 0xFF and self.__buffer[1] >= 0xE0:
            self.__curr_frame.init_header_params(self.__buffer)
        else:
            self.__valid = False

    def __init_curr_frame(self):
        self.__curr_frame.init_frame_params(self.__buffer, self.__file_data, self.__offset)

    def parse_file(self) -> (int, np.ndarray):
        num_of_parsed_frames = 0

        pbar = tqdm(total=self.__file_length + 1 - HEADER_SIZE, desc='decoding')
        while self.__valid and self.__file_length > self.__offset + HEADER_SIZE:
            self.__init_curr_header()
            if self.__valid:
                # Check if complete frame fits in file
                if self.__offset + self.__curr_frame.frame_size > self.__file_length:
                    break
                
                self.__init_curr_frame()
                # get all bits from the huffman tables
                self.output_bits += util.bit_from_huffman_tables(self.__curr_frame.all_huffman_tables)
                num_of_parsed_frames += 1
                
                # Only append valid frames
                self.frame_data.append(copy.deepcopy(self.__curr_frame))
                
                pbar.update(self.__curr_frame.frame_size)
                
                self.__offset += self.__curr_frame.frame_size
                buf_end = min(self.__offset + 4096, self.__file_length)
                self.__buffer = self.__file_data[self.__offset:buf_end]
                # print(f'Parsed: {num_of_parsed_frames}')

        pbar.close()
        
        # Convert list of quantized samples to numpy array
        if self.frame_data:
            self.frame_data = np.array(self.frame_data)
        else:
            self.frame_data = np.array([])

        return num_of_parsed_frames, self.frame_data

    def get_bitrate(self) -> int:
        return self.__curr_frame.get_bitrate()

    def get_sample_rate(self) -> int:
        return self.__curr_frame.sampling_rate

    def get_header(self):
        return self.__curr_frame.get_header()
