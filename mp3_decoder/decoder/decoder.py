import os
import sys
import time

from mp3_decoder.decoder.ID3_Parser import ID3
from mp3_decoder.decoder.MP3_Parser import MP3Parser


class Decoder:

    def __init__(self, file_path: str, output_file_path: str):
        self.__file_path: str = file_path
        self.__output_file_path: str = output_file_path

        if not os.path.exists(self.__file_path):
            sys.exit(f'File {self.__file_path} not found.')

        with open(self.__file_path, 'rb') as f:
            self.__hex_data: list = [c for c in f.read()]

        self.__id3_decoder: ID3 = ID3(self.__hex_data)
        if self.__id3_decoder.is_valid:
            offset = self.__id3_decoder.offset
        else:
            offset = 0

        self.__parser: MP3Parser = MP3Parser(self.__hex_data, offset, self.__output_file_path)

    def __parse_metadata(self, id3_parser: ID3):
        with open('METADATA.txt', 'w') as metadata:
            metadata.write(f'METADATA FOR FILE: {self.__file_path}\n')
            metadata.write('################################\n\n\n')
            metadata.write(f'ID3 Version: {id3_parser.version}\n')
            if len(id3_parser.id3_flags) > 0:
                metadata.write('ID3 Flags:\n')
                for flag in id3_parser.id3_flags:
                    metadata.write(f'- {flag}\n')
                metadata.write('\n')

            metadata.write('\nID3 Frames:\n')
            for i, frame in enumerate(id3_parser.id3_frames):
                metadata.write(f'Frame number: {i}\n')
                metadata.write(f'Frame ID: {frame.id}\n')
                metadata.write(f'Content: {frame.content}\n')
                if len(frame.frame_flags) > 0:
                    metadata.write('Frame Flags:\n')
                    for flag in frame.frame_flags:
                        metadata.write(f'- {flag}\n')
                metadata.write('\n')

    def decode(self, quiet: bool = True, reveal: bool = False, txt_file_path: str = "") -> int:
        if not quiet and self.__id3_decoder.is_valid:
            self.__parse_metadata(self.__id3_decoder)

        start = time.time()
        num_of_parsed_frames, frame_data = self.__parser.parse_file()
        parsing_time = time.time() - start
        if not quiet:
            print('\nParsed', num_of_parsed_frames, 'frames in', parsing_time, 'seconds.')

        return self.__parser, frame_data, num_of_parsed_frames