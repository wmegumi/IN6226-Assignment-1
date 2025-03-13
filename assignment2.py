import os
import re
import time
import struct
import pickle
import psutil
import heapq
import configparser
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer


# Reuse existing functions from assignment1.py
def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()


def tokenize(text, doc_id):
    tokens = re.split(r'\s+', text)
    tokens = normalize_token(tokens)
    return [(token, doc_id) for token in tokens if token]


def normalize_token(tokens):
    stemmer = PorterStemmer()
    normalized_tokens = []
    for token in tokens:
        token = token.lower()
        token = re.sub(r'[^a-zA-Z]', '', token)
        if token:
            normalized_tokens.append(stemmer.stem(token))  # Apply stemming
    return normalized_tokens


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().peak_wset / 1024 / 1024  # Convert to MB


# ======= New Compression Implementation =======

class CompressedIndex:
    """Compressed inverted index using dictionary-as-a-string approach"""

    def __init__(self):
        self.dictionary_string = ""  # All terms concatenated
        self.term_offsets = []  # List of (term_start, term_length, posting_offset)
        self.postings_data = bytearray()  # Binary postings data
        self.doc_count = 0  # Total number of documents
        self.all_doc_ids = set()  # Set of all document IDs

    def build_from_index_file(self, index_file):
        """Build a compressed index from a traditional index file"""
        # First read the entire index to gather statistics
        print("Loading index file...")
        start_time = time.time()

        term_dict = {}  # Term -> postings list

        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    term, postings_str = line.strip().split(': ', 1)
                    postings = postings_str.split(', ')
                    term_dict[term] = postings
                    self.all_doc_ids.update(postings)

        self.doc_count = len(self.all_doc_ids)
        load_time = time.time() - start_time
        print(f"Index loaded in {load_time:.2f} seconds")

        # Now build the compressed structures
        print("Building compressed structures...")
        comp_start_time = time.time()

        sorted_terms = sorted(term_dict.keys())
        postings_offset = 0

        # Build the dictionary string and offsets
        for term in sorted_terms:
            term_start = len(self.dictionary_string)
            term_len = len(term)
            self.dictionary_string += term

            # Store term information: (start position, length, postings offset)
            self.term_offsets.append((term_start, term_len, postings_offset))

            # Encode the postings list
            postings = term_dict[term]
            postings_count = len(postings)

            # Format: [count][doc_id1][doc_id2]...
            postings_bytes = struct.pack('!I', postings_count)
            self.postings_data.extend(postings_bytes)

            # Variable-length gap encoding for document IDs
            sorted_postings = sorted(postings)

            # Store first doc ID as is
            first_id = sorted_postings[0].encode('utf-8')
            id_len = len(first_id)
            self.postings_data.extend(struct.pack('!I', id_len))
            self.postings_data.extend(first_id)

            # Store gaps for the rest
            prev = sorted_postings[0]
            for doc_id in sorted_postings[1:]:
                # In a real gap encoding, we'd convert these to integers and store deltas
                # Since we're using string IDs, we'll just store the full ID for now
                doc_id_bytes = doc_id.encode('utf-8')
                id_len = len(doc_id_bytes)
                self.postings_data.extend(struct.pack('!I', id_len))
                self.postings_data.extend(doc_id_bytes)

            # Update offset for next postings list
            postings_offset = len(self.postings_data)

        comp_time = time.time() - comp_start_time
        print(f"Compression completed in {comp_time:.2f} seconds")

        # Calculate compression ratio
        original_size = sum(len(term) + sum(len(doc_id) for doc_id in postings)
                            for term, postings in term_dict.items())
        compressed_size = len(self.dictionary_string) + len(self.term_offsets) * 12 + len(self.postings_data)
        ratio = original_size / compressed_size if compressed_size > 0 else 0

        print(f"Original size: {original_size / 1024:.2f} KB")
        print(f"Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"Compression ratio: {ratio:.2f}x")

        return load_time + comp_time

    def save(self, filename):
        """Save the compressed index to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'dictionary_string': self.dictionary_string,
                'term_offsets': self.term_offsets,
                'postings_data': self.postings_data,
                'doc_count': self.doc_count,
                'all_doc_ids': self.all_doc_ids
            }, f)

    @classmethod
    def load(cls, filename):
        """Load a compressed index from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        index = cls()
        index.dictionary_string = data['dictionary_string']
        index.term_offsets = data['term_offsets']
        index.postings_data = data['postings_data']
        index.doc_count = data['doc_count']
        index.all_doc_ids = data['all_doc_ids']
        return index

    def get_term_id(self, term):
        """Binary search for a term in the dictionary and return its ID"""
        left, right = 0, len(self.term_offsets) - 1

        while left <= right:
            mid = (left + right) // 2
            start, length, _ = self.term_offsets[mid]
            mid_term = self.dictionary_string[start:start + length]

            if mid_term == term:
                return mid
            elif mid_term < term:
                left = mid + 1
            else:
                right = mid - 1

        return -1  # Term not found

    def get_postings(self, term):
        """Get the postings list for a term"""
        term_id = self.get_term_id(term)

        if term_id == -1:  # Term not found
            return []

        _, _, postings_offset = self.term_offsets[term_id]

        # If we have a next term, we can determine the end of this term's postings
        next_offset = len(self.postings_data)
        if term_id + 1 < len(self.term_offsets):
            _, _, next_offset = self.term_offsets[term_id + 1]

        # Read postings data
        offset = postings_offset

        # Read count
        count = struct.unpack('!I', self.postings_data[offset:offset + 4])[0]
        offset += 4

        postings = []
        for _ in range(count):
            # Read document ID length
            id_len = struct.unpack('!I', self.postings_data[offset:offset + 4])[0]
            offset += 4

            # Read document ID
            doc_id = self.postings_data[offset:offset + id_len].decode('utf-8')
            offset += id_len

            postings.append(doc_id)

        return postings


# ======= Boolean Search Implementation =======

class BooleanSearchEngine:
    """Boolean search engine supporting AND, OR, NOT operators"""

    def __init__(self, index):
        """Initialize with either a compressed or standard index"""
        self.index = index
        self.stemmer = PorterStemmer()

    def normalize_query(self, query):
        """Normalize and tokenize the query"""
        # Split on space but keep operators
        tokens = []
        current_token = ""
        operators = ['AND', 'OR', 'NOT', '(', ')']

        # Check if query has explicit operators
        has_explicit_ops = any(op in query.upper().split() for op in ['AND', 'OR', 'NOT'])

        # If no explicit operators, treat as AND query
        if not has_explicit_ops:
            query_terms = query.split()
            if len(query_terms) > 1:
                # Insert AND between each term
                query = " AND ".join(query_terms)

        i = 0
        while i < len(query):
            if query[i:].upper().startswith(tuple(operators)):
                # Found an operator
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""

                # Find which operator it is
                for op in operators:
                    if query[i:].upper().startswith(op):
                        tokens.append(op)
                        i += len(op)
                        break
            else:
                current_token += query[i]
                i += 1

        # Add the last token if there is one
        if current_token.strip():
            tokens.append(current_token.strip())

        # Normalize non-operator tokens
        normalized_tokens = []
        for token in tokens:
            if token.upper() in operators or token in ['(', ')']:
                normalized_tokens.append(token.upper())
            else:
                # Normalize and stem
                norm_token = normalize_token([token])
                if norm_token:
                    normalized_tokens.append(norm_token[0])

        return normalized_tokens

    def parse_query(self, query):
        """Parse a query into a syntax tree using shunting yard algorithm"""
        tokens = self.normalize_query(query)
        output_queue = []
        operator_stack = []

        precedence = {
            'NOT': 3,
            'AND': 2,
            'OR': 1
        }

        for token in tokens:
            if token not in ['AND', 'OR', 'NOT', '(', ')']:
                # It's a term - add to output
                output_queue.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                # Pop operators until matching left parenthesis
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())

                # Remove the left parenthesis
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
            else:
                # It's an operator
                while (operator_stack and
                       operator_stack[-1] != '(' and
                       ((token != 'NOT' and precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)) or
                        (token == 'NOT' and precedence.get(operator_stack[-1], 0) > precedence.get(token, 0)))):
                    output_queue.append(operator_stack.pop())

                operator_stack.append(token)

        # Pop any remaining operators to the output queue
        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue

    def evaluate_query(self, query):
        """Evaluate a Boolean query and return matching documents"""
        postfix_tokens = self.parse_query(query)
        stack = []

        for token in postfix_tokens:
            if token == 'AND':
                right = stack.pop()
                left = stack.pop()
                stack.append(self._intersect(left, right))
            elif token == 'OR':
                right = stack.pop()
                left = stack.pop()
                stack.append(self._union(left, right))
            elif token == 'NOT':
                operand = stack.pop()
                stack.append(self._complement(operand))
            else:
                # It's a term - get its postings list
                if isinstance(self.index, CompressedIndex):
                    postings = self.index.get_postings(token)
                else:
                    # For standard index: assume it's a dictionary mapping terms to doc IDs
                    postings = self.index.get(token, [])

                stack.append(set(postings))

        if not stack:
            return set()  # No results

        return stack[0]

    def _intersect(self, set1, set2):
        """Return the intersection of two sets"""
        return set1.intersection(set2)

    def _union(self, set1, set2):
        """Return the union of two sets"""
        return set1.union(set2)

    def _complement(self, doc_set):
        """Return the complement of a set (NOT operation)"""
        if isinstance(self.index, CompressedIndex):
            all_docs = self.index.all_doc_ids
        else:
            # For standard index, all_docs should be provided or computed
            all_docs = set()
            for postings in self.index.values():
                all_docs.update(postings)

        return all_docs.difference(doc_set)

    def search(self, query):
        """Execute a search and return results"""
        start_time = time.time()
        results = self.evaluate_query(query)
        search_time = time.time() - start_time

        return list(results), search_time


# ======= Main Application Code =======

def load_standard_index(index_file):
    """Load a standard index from a file"""
    index = {}
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                term, postings_str = line.strip().split(': ', 1)
                index[term] = postings_str.split(', ')
    return index


def build_compressed_index(index_file, compressed_file):
    """Build and save a compressed index"""
    start_time = time.time()
    initial_memory = get_memory_usage()

    compressed_index = CompressedIndex()
    compressed_index.build_from_index_file(index_file)
    compressed_index.save(compressed_file)

    end_time = time.time()
    final_memory = get_memory_usage()

    print(f"Compressed index built and saved in {end_time - start_time:.2f} seconds")
    print(f"Memory used: {final_memory - initial_memory:.2f} MB")

    return compressed_index


def interactive_search(engine, engine_type):
    """Run an interactive search prompt"""
    print(f"\n=== {engine_type} Search Engine ===")
    print("Enter your Boolean queries (e.g., 'term1 AND term2', 'term1 OR (term2 AND NOT term3)')")
    print("Enter 'quit' to exit")

    while True:
        query = input("\nEnter query: ")
        if query.lower() == 'quit':
            break

        results, search_time = engine.search(query)

        print(f"Found {len(results)} results in {search_time:.5f} seconds")
        if results:
            # Show first 5 results
            print("First 5 documents:")
            for i, doc in enumerate(results[:5]):
                print(f"{i + 1}. {doc}")

            if len(results) > 5:
                print(f"... and {len(results) - 5} more")


def compare_performance(standard_file, compressed_file, test_queries):
    """Compare performance of standard and compressed indexes"""
    print("\n=== Performance Comparison ===")

    # Load standard index
    start_std = time.time()
    std_memory_start = get_memory_usage()
    standard_index = load_standard_index(standard_file)
    std_engine = BooleanSearchEngine(standard_index)
    std_load_time = time.time() - start_std
    std_memory = get_memory_usage() - std_memory_start

    # Load compressed index
    start_comp = time.time()
    comp_memory_start = get_memory_usage()
    compressed_index = CompressedIndex.load(compressed_file)
    comp_engine = BooleanSearchEngine(compressed_index)
    comp_load_time = time.time() - start_comp
    comp_memory = get_memory_usage() - comp_memory_start

    print(f"Standard index load time: {std_load_time:.4f} seconds, memory: {std_memory:.2f} MB")
    print(f"Compressed index load time: {comp_load_time:.4f} seconds, memory: {comp_memory:.2f} MB")

    # Testing queries
    print("\nQuery Performance:")
    print(f"{'Query':<30} {'Standard Time (s)':<15} {'Compressed Time (s)':<15} {'Speed Ratio':<10}")
    print("-" * 70)

    total_std_time = 0
    total_comp_time = 0

    for query in test_queries:
        _, std_time = std_engine.search(query)
        _, comp_time = comp_engine.search(query)

        ratio = std_time / comp_time if comp_time > 0 else float('inf')

        print(f"{query[:30]:<30} {std_time:.8f}       {comp_time:.8f}       {ratio:.2f}x")

        total_std_time += std_time
        total_comp_time += comp_time

    avg_ratio = total_std_time / total_comp_time if total_comp_time > 0 else float('inf')
    print("-" * 70)
    print(
        f"Average:                      {total_std_time / len(test_queries):.8f}       {total_comp_time / len(test_queries):.8f}       {avg_ratio:.2f}x")

    memory_ratio = std_memory / comp_memory if comp_memory > 0 else float('inf')
    print(f"\nMemory Usage Ratio: {memory_ratio:.2f}x")


def main():
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    input_directory = config['Settings']['input_directory']
    index_file = config['Settings']['output_file']
    compressed_file = "compressed_index.bin"

    # Check if compressed index already exists, otherwise build it
    if not os.path.exists(compressed_file):
        print("Building compressed index...")
        compressed_index = build_compressed_index(index_file, compressed_file)
    else:
        print(f"Loading compressed index from {compressed_file}...")
        compressed_index = CompressedIndex.load(compressed_file)

    # Create the search engine with compressed index
    search_engine = BooleanSearchEngine(compressed_index)

    # Compare performance (optional)
    test_queries = [
        "Stevens",
        "Mills AND Cheryl",
        "Coleman OR Jones",
        "Stevens NOT condolence",
    ]
    compare_performance(index_file, compressed_file, test_queries)

    # Run interactive search
    interactive_search(search_engine, "Compressed")


if __name__ == "__main__":
    main()