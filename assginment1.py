
def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

import re

def tokenize(text, doc_id):
    tokens = re.split(r'\s+', text)
    return [(token, doc_id) for token in tokens if token]

import string
from nltk.stem import PorterStemmer

def normalize_token(token):
    stemmer = PorterStemmer()
    token = token.lower()
    token = token.translate(str.maketrans('', '', string.punctuation))
    token = stemmer.stem(token)
    return token

def process_tokens(tokens):
    return [(normalize_token(token), doc_id) for token, doc_id in tokens]

import psutil
import os
import time


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def spimi_invert(token_stream, block_size):
    dictionary = {}
    blocks = []
    current_block_tokens = []

    for token, doc_id in token_stream:
        current_block_tokens.append((token, doc_id))
        if len(current_block_tokens) >= block_size:
            # Process the current block
            for t, d in current_block_tokens:
                if t not in dictionary:
                    dictionary[t] = []
                dictionary[t].append(d)
            blocks.append(dictionary)
            dictionary = {}
            current_block_tokens = []

    # Process remaining tokens
    if current_block_tokens:
        for t, d in current_block_tokens:
            if t not in dictionary:
                dictionary[t] = []
            dictionary[t].append(d)
        blocks.append(dictionary)

    return blocks


def merge_blocks(blocks):
    start_time = time.time()
    merged_dict = {}
    for block in blocks:
        for token, postings in block.items():
            if token not in merged_dict:
                merged_dict[token] = []
            merged_dict[token].extend(postings)
    merge_time = time.time() - start_time
    return merged_dict, merge_time


def main(input_directory, block_size, output_file):
    total_start_time = time.time()
    initial_memory = get_memory_usage()

    # Timing for file listing
    files_start_time = time.time()
    files = list_files(input_directory)
    files_time = time.time() - files_start_time

    token_stream = []
    processing_start_time = time.time()

    # Process files and collect tokens
    for file_path in files:
        text = read_file(file_path)
        tokens = tokenize(text, file_path)
        processed_tokens = process_tokens(tokens)
        token_stream.extend(processed_tokens)

    processing_time = time.time() - processing_start_time

    # SPIMI indexing
    indexing_start_time = time.time()
    blocks = spimi_invert(token_stream, block_size)
    indexing_time = time.time() - indexing_start_time

    # Merge blocks and measure time
    index, merge_time = merge_blocks(blocks)

    # Write index
    writing_start_time = time.time()
    write_index_to_file(index, output_file)
    writing_time = time.time() - writing_start_time

    total_time = time.time() - total_start_time
    peak_memory = get_memory_usage()
    memory_used = peak_memory - initial_memory

    # Print statistics
    print(f"\nPerformance Statistics:")
    print(f"File listing time: {files_time:.2f} seconds")
    print(f"Token processing time: {processing_time:.2f} seconds")
    print(f"Indexing time: {indexing_time:.2f} seconds")
    print(f"Merging time: {merge_time:.2f} seconds")
    print(f"Writing time: {writing_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Number of blocks created: {len(blocks)}")


def write_index_to_file(index, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for token in sorted(index.keys()):
            # Replace backslashes with forward slashes in all paths
            normalized_paths = [path.replace('\\', '/') for path in index[token]]
            file.write(f"{token}: {', '.join(normalized_paths)}\n")



import configparser

def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)

    input_directory = config['Settings']['input_directory']
    block_size = int(config['Settings']['block_size'])
    output_file = config['Settings']['output_file']

    return input_directory, block_size, output_file

if __name__ == "__main__":
    input_directory, block_size, output_file = read_config()
    main(input_directory, block_size, output_file)