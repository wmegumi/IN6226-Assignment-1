
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

import string
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


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

from collections import defaultdict

def spimi_invert(files, block_size):
    # files = list_files(input_directory)
    dictionary = defaultdict(set)
    blocks = []
    block_count = 0
    token_count = 0
    # current_block_tokens = []

    for file_path in files:
        text = read_file(file_path)

        for token, doc_id in tokenize(text, file_path):
            # normalized_token = normalize_token(token)
            dictionary[token].add(doc_id)

            if token_count == block_size:
                block_file = f"block_{block_count}.txt"
                write_block_to_disk(dictionary, block_file)
                blocks.append(block_file)
                dictionary.clear()
                block_count += 1
                token_count = 0
            else:
                token_count += 1

    if dictionary:
        block_file = f"block_{block_count}.txt"
        write_block_to_disk(dictionary, block_file)
        blocks.append(block_file)

    return blocks

def write_block_to_disk(dictionary, block_file):
    with open(block_file, 'w', encoding='utf-8') as f:
        for token in sorted(dictionary.keys()):
            f.write(f"{token}: {', '.join(dictionary[token])}\n")

import heapq
def merge_blocks(blocks, output_file):
    start_time = time.time()

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

    merge_time = time.time() - start_time
    return merge_time


def main(input_directory, block_size, output_file):
    total_start_time = time.time()
    initial_memory = get_memory_usage()

    # Timing for file listing
    files_start_time = time.time()
    files = list_files(input_directory)
    files_time = time.time() - files_start_time

    # SPIMI indexing
    indexing_start_time = time.time()
    blocks = spimi_invert(files, block_size)
    indexing_time = time.time() - indexing_start_time

    # Merge blocks and measure time
    merge_time = merge_blocks(blocks, output_file)

    total_time = time.time() - total_start_time
    peak_memory = get_memory_usage()
    memory_used = peak_memory - initial_memory

    # Print statistics
    print(f"\nPerformance Statistics:")
    print(f"File listing time: {files_time:.2f} seconds")
    # print(f"Token processing time: {processing_time:.2f} seconds")
    print(f"Indexing time: {indexing_time:.2f} seconds")
    print(f"Merging time: {merge_time:.2f} seconds")
    # print(f"Writing time: {writing_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Number of blocks created: {len(blocks)}")


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