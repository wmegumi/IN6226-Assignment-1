
def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

import re

def tokenize(text, doc_id):
    tokens = re.split(r'\s+', text)
    tokens = normalize_token(tokens)
    return [(token, doc_id) for token in tokens if token]

from nltk.stem import PorterStemmer

def normalize_token(tokens):
    stemmer = PorterStemmer()
    normalized_tokens = []
    for token in tokens:
        token = token.lower()
        token = re.sub(r'[^a-zA-Z]', '', token)
        if token:
            normalized_tokens.append(stemmer.stem(token))  # Apply stemming
    return normalized_tokens

import psutil
import os
import time
from pympler import asizeof

def get_memory_usage():
    process = psutil.Process()
    # return process.memory_info().rss / 1024 / 1024  # Convert to MB
    return process.memory_info().peak_wset / 1024 / 1024  # Convert to MB

from collections import defaultdict

def spimi_invert(files, block_size):
    dictionary = defaultdict(set)
    blocks = [] # List to store block file names
    block_count = 0
    token_count = 0 # Counter for tokens processed in current block

    peak_dict_memory = 0 # Track peak memory usage of dictionary (in MB)

    for file_path in files:
        text = read_file(file_path)

        for token, doc_id in tokenize(text, file_path):
            dictionary[token].add(doc_id)

            # If block size is reached, write to disk and reset dictionary
            if token_count >= block_size:
            # if len(dictionary) >= block_size:
            # if sys.getsizeof(dictionary) >= block_size:
                peak_dict_memory = max(peak_dict_memory, asizeof.asizeof(dictionary) / (1024 * 1024))
                block_file = f"./blocks/block_{block_count}.txt"
                write_block_to_disk(dictionary, block_file)
                blocks.append(block_file)

                dictionary.clear()
                block_count += 1
                token_count = 0
            else:
                token_count += 1

    # Write remaining dictionary to disk if any data is left
    if dictionary:
        peak_dict_memory = max(peak_dict_memory, asizeof.asizeof(dictionary) / (1024 * 1024))
        block_file = f"./blocks/block_{block_count}.txt"
        write_block_to_disk(dictionary, block_file)
        blocks.append(block_file)

    return blocks, peak_dict_memory

def write_block_to_disk(dictionary, block_file):
    if not os.path.exists('blocks'):
        os.makedirs('blocks')

    with open(block_file, 'w', encoding='utf-8') as f:
        for token in sorted(dictionary.keys()):
            f.write(f"{token}: {', '.join(dictionary[token])}\n")

import heapq
def merge_blocks(blocks, output_file):
    heap = []
    file_pointers = {block: open(block, 'r', encoding='utf-8') for block in blocks}

    # init heap
    for block, f in file_pointers.items():
        line = f.readline().strip()
        if line:
            token, postings = line.split(": ")
            postings = postings.split(", ")
            heapq.heappush(heap, (token, postings, block))

    with open(output_file, 'w', encoding='utf-8') as out:
        last_token, merged_postings = None, set()

        while heap:
            token, postings, block = heapq.heappop(heap)

            # merge identical token
            if last_token and token != last_token:
                out.write(f"{last_token}: {', '.join(sorted(merged_postings))}\n")
                merged_postings = set()

            merged_postings.update(postings)
            last_token = token

            # read nextline of block
            next_line = file_pointers[block].readline().strip()
            if next_line:
                next_token, next_postings = next_line.split(": ")
                heapq.heappush(heap, (next_token, next_postings.split(", "), block))

        # write last token
        if last_token:
            out.write(f"{last_token}: {', '.join(sorted(merged_postings))}\n")

    # close all files
    for f in file_pointers.values():
        f.close()


def main(input_directory, block_size, output_file):
    total_start_time = time.time()
    initial_memory = get_memory_usage()

    # Timing for file listing
    files_start_time = time.time()
    files = list_files(input_directory)
    files_time = time.time() - files_start_time

    # SPIMI indexing
    indexing_start_time = time.time()
    blocks, peak_dict_memory = spimi_invert(files, block_size)
    indexing_time = time.time() - indexing_start_time

    # Merge blocks and measure time
    merge_start_time = time.time()
    merge_blocks(blocks, output_file)
    merge_time = time.time() - merge_start_time

    total_time = time.time() - total_start_time
    peak_memory = get_memory_usage()
    memory_used = peak_memory - initial_memory

    # Print statistics
    print(f"\n{'Performance Statistics':<20}")
    print(f"{'File listing time:':<20} {files_time:>10.2f} seconds")
    print(f"{'Indexing time:':<20} {indexing_time:>10.2f} seconds")
    print(f"{'Merging time:':<20} {merge_time:>10.2f} seconds")
    print(f"{'Total time:':<20} {total_time:>10.2f} seconds")
    print(f"{'Memory usage:':<20} {memory_used:>10.2f} MB")
    print(f"{'Peak memory:':<20} {peak_memory:>10.2f} MB")
    print(f"{'Peak dict memory:':<20} {peak_dict_memory:>10.2f} MB")
    print(f"{'Number of blocks:':<20} {len(blocks):>10} blocks")



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