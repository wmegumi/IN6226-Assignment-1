import os
import re
import time
import struct
import pickle
import psutil
import heapq
import math
import numpy as np
import configparser
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer


# Reuse existing functions from assignment1.py and assignment2.py
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


# ======= Enhanced Compressed Index with Term Statistics =======

class EnhancedCompressedIndex:
    """Enhanced compressed inverted index with term statistics for ranking"""

    def __init__(self):
        self.dictionary_string = ""  # All terms concatenated
        self.term_offsets = []  # List of (term_start, term_length, posting_offset)
        self.postings_data = bytearray()  # Binary postings data
        self.doc_count = 0  # Total number of documents
        self.all_doc_ids = set()  # Set of all document IDs
        self.doc_lengths = {}  # Document lengths (word count)
        self.avg_doc_length = 0  # Average document length
        self.term_df = {}  # Document frequency for each term
        self.doc_terms = {}  # Terms in each document with frequencies

    def build_from_index_file(self, index_file, raw_docs_dir=None):
        """Build a compressed index from a traditional index file with term statistics"""
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
                    self.term_df[term] = len(postings)  # Document frequency

        self.doc_count = len(self.all_doc_ids)
        load_time = time.time() - start_time
        print(f"Index loaded in {load_time:.2f} seconds")

        # Calculate term frequencies and document lengths
        if raw_docs_dir and os.path.exists(raw_docs_dir):
            print("Computing document statistics for ranking...")
            self._compute_doc_statistics(raw_docs_dir)

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
                doc_id_bytes = doc_id.encode('utf-8')
                id_len = len(doc_id_bytes)
                self.postings_data.extend(struct.pack('!I', id_len))
                self.postings_data.extend(doc_id_bytes)

            # Update offset for next postings list
            postings_offset = len(self.postings_data)

        comp_time = time.time() - comp_start_time
        print(f"Compression completed in {comp_time:.2f} seconds")

        return load_time + comp_time

    def _compute_doc_statistics(self, raw_docs_dir):
        """Compute document statistics for ranking (TF, document length)"""
        total_tokens = 0
        files = list_files(raw_docs_dir)

        for doc_id in files:
            try:
                content = read_file(doc_id)
                tokens = [token for token, _ in tokenize(content, doc_id)]

                # Count token frequencies for this document
                term_freqs = Counter(tokens)
                self.doc_terms[doc_id] = dict(term_freqs)

                # Store document length
                doc_length = len(tokens)
                self.doc_lengths[doc_id] = doc_length
                total_tokens += doc_length
            except Exception as e:
                print(f"Error processing {doc_id}: {e}")
                # Use default values if file can't be processed
                self.doc_lengths[doc_id] = 1
                self.doc_terms[doc_id] = {}

        # Calculate average document length
        if files:
            self.avg_doc_length = total_tokens / len(files)
        else:
            self.avg_doc_length = 0

        print(f"Processed {len(files)} documents, avg length: {self.avg_doc_length:.2f} tokens")

    def save(self, filename):
        """Save the compressed index to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'dictionary_string': self.dictionary_string,
                'term_offsets': self.term_offsets,
                'postings_data': self.postings_data,
                'doc_count': self.doc_count,
                'all_doc_ids': self.all_doc_ids,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'term_df': self.term_df,
                'doc_terms': self.doc_terms
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
        index.doc_lengths = data.get('doc_lengths', {})
        index.avg_doc_length = data.get('avg_doc_length', 0)
        index.term_df = data.get('term_df', {})
        index.doc_terms = data.get('doc_terms', {})
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

    def get_document_frequency(self, term):
        """Get the document frequency for a term"""
        return self.term_df.get(term, 0)

    def get_term_frequency(self, term, doc_id):
        """Get the term frequency in a specific document"""
        if doc_id in self.doc_terms:
            return self.doc_terms[doc_id].get(term, 0)
        return 0

    def get_document_length(self, doc_id):
        """Get the length of a document"""
        return self.doc_lengths.get(doc_id, 0)

    def get_all_terms(self):
        """Get all terms in the index"""
        terms = []
        for i in range(len(self.term_offsets)):
            start, length, _ = self.term_offsets[i]
            term = self.dictionary_string[start:start + length]
            terms.append(term)
        return terms


# ======= Enhanced Boolean Search with Ranking =======

class RankedSearchEngine:
    """Enhanced search engine supporting Boolean operations and ranking"""

    def __init__(self, index):
        """Initialize with an enhanced index"""
        self.index = index
        self.stemmer = PorterStemmer()
        self.ranking_method = "tfidf"  # Default ranking method
        self.bm25_k1 = 1.2  # BM25 parameter k1
        self.bm25_b = 0.75  # BM25 parameter b

    def set_ranking_method(self, method):
        """Set the ranking method to use"""
        valid_methods = ["none", "tfidf", "bm25", "vector"]
        if method not in valid_methods:
            print(f"Invalid ranking method: {method}. Using default (tfidf).")
            method = "tfidf"
        self.ranking_method = method

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

        # Track query terms for ranking
        query_terms = []

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
                query_terms.append(token)  # Track for ranking
                postings = self.index.get_postings(token)
                stack.append(set(postings))

        if not stack:
            return set(), query_terms  # No results

        return stack[0], query_terms

    def _intersect(self, set1, set2):
        """Return the intersection of two sets"""
        return set1.intersection(set2)

    def _union(self, set1, set2):
        """Return the union of two sets"""
        return set1.union(set2)

    def _complement(self, doc_set):
        """Return the complement of a set (NOT operation)"""
        all_docs = self.index.all_doc_ids
        return all_docs.difference(doc_set)

    def search(self, query, with_ranking=True):
        """Execute a search and return results, optionally with ranking"""
        start_time = time.time()

        # First, get matching documents using boolean search
        matching_docs, query_terms = self.evaluate_query(query)

        results = list(matching_docs)

        # If ranking is requested and we have matching documents
        if with_ranking and self.ranking_method != "none" and results and query_terms:
            # Count term frequencies in the query
            query_term_counts = Counter(query_terms)

            # Calculate scores based on the selected ranking method
            if self.ranking_method == "tfidf":
                scores = self._score_tfidf(results, query_term_counts)
            elif self.ranking_method == "bm25":
                scores = self._score_bm25(results, query_term_counts)
            elif self.ranking_method == "vector":
                scores = self._score_vector_space(results, query_term_counts)
            else:
                # Default to no ranking
                scores = {doc_id: 1.0 for doc_id in results}

            # Sort results by score in descending order
            results = sorted(results, key=lambda doc_id: scores.get(doc_id, 0.0), reverse=True)

        search_time = time.time() - start_time

        return results, search_time, len(matching_docs)

    def _score_tfidf(self, docs, query_terms):
        """Score documents using TF-IDF"""
        scores = {}
        N = self.index.doc_count  # Number of documents in corpus

        for doc_id in docs:
            score = 0.0
            for term, query_count in query_terms.items():
                # Term frequency in the document
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue

                # Document frequency of the term
                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                # TF-IDF calculation
                idf = math.log(N / df)
                score += tf * idf * query_count

            scores[doc_id] = score

        return scores

    def _score_bm25(self, docs, query_terms):
        """Score documents using BM25"""
        scores = {}
        N = self.index.doc_count  # Number of documents in corpus
        avdl = self.index.avg_doc_length  # Average document length
        k1 = self.bm25_k1
        b = self.bm25_b

        for doc_id in docs:
            score = 0.0
            doc_len = self.index.get_document_length(doc_id)
            if doc_len == 0:
                continue

            for term, query_count in query_terms.items():
                # Term frequency in the document
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue

                # Document frequency of the term
                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                # IDF component
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

                # BM25 TF component
                tf_component = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (doc_len / avdl)) + tf)

                score += idf * tf_component * query_count

            scores[doc_id] = score

        return scores

    def _score_vector_space(self, docs, query_terms):
        """Score documents using Vector Space Model with cosine similarity"""
        scores = {}
        all_terms = set(query_terms.keys())
        N = self.index.doc_count  # Number of documents in corpus

        # Calculate query vector
        query_vector = {}
        query_magnitude = 0.0

        for term, count in query_terms.items():
            df = self.index.get_document_frequency(term)
            if df == 0:
                continue

            # TF-IDF weight for the term in the query
            idf = math.log(N / df)
            weight = count * idf
            query_vector[term] = weight
            query_magnitude += weight ** 2

        query_magnitude = math.sqrt(query_magnitude)

        # Calculate document vectors and cosine similarity
        for doc_id in docs:
            doc_vector = {}
            doc_magnitude = 0.0

            for term in all_terms:
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue

                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                # TF-IDF weight for the term in the document
                idf = math.log(N / df)
                weight = tf * idf
                doc_vector[term] = weight
                doc_magnitude += weight ** 2

            doc_magnitude = math.sqrt(doc_magnitude)

            # Calculate cosine similarity
            dot_product = 0.0
            for term, q_weight in query_vector.items():
                dot_product += q_weight * doc_vector.get(term, 0.0)

            if doc_magnitude > 0 and query_magnitude > 0:
                similarity = dot_product / (doc_magnitude * query_magnitude)
            else:
                similarity = 0.0

            scores[doc_id] = similarity

        return scores


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


def build_enhanced_index(index_file, raw_docs_dir, compressed_file):
    """Build and save an enhanced index with ranking statistics"""
    start_time = time.time()
    initial_memory = get_memory_usage()

    print(f"Building enhanced index with statistics from {raw_docs_dir}...")

    enhanced_index = EnhancedCompressedIndex()
    enhanced_index.build_from_index_file(index_file, raw_docs_dir)
    enhanced_index.save(compressed_file)

    end_time = time.time()
    final_memory = get_memory_usage()

    print(f"Enhanced index built and saved in {end_time - start_time:.2f} seconds")
    print(f"Memory used: {final_memory - initial_memory:.2f} MB")

    return enhanced_index


def interactive_search(engine, engine_type):
    """Run an interactive search prompt with ranking options"""
    print(f"\n=== {engine_type} Search Engine ===")
    print("Enter your Boolean queries (e.g., 'term1 AND term2', 'term1 OR (term2 AND NOT term3)')")
    print("Additional commands:")
    print("  !ranking [none|tfidf|bm25|vector] - Change ranking method")
    print("  !compare <query> - Compare ranked vs. unranked results")
    print("  !quit - Exit the search engine")

    while True:
        query = input("\nEnter query or command: ")

        if query.lower() == "!quit" or query.lower() == "quit":
            break

        elif query.lower().startswith("!ranking "):
            method = query.lower().split()[1]
            engine.set_ranking_method(method)
            print(f"Ranking method set to: {engine.ranking_method}")

        elif query.lower().startswith("!compare "):
            compare_query = query[9:]  # Remove !compare prefix
            _compare_search_results(engine, compare_query)

        else:
            # Standard search
            results, search_time, total_matches = engine.search(query)

            print(f"Found {total_matches} matches in {search_time:.5f} seconds")
            if results:
                print(f"Top 10 results (ranked by {engine.ranking_method}):")
                for i, doc in enumerate(results[:10]):
                    print(f"{i + 1}. {os.path.basename(doc)}")

                if len(results) > 10:
                    print(f"... and {len(results) - 10} more")


def _compare_search_results(engine, query):
    """Compare ranked and unranked search results"""
    print(f"\nComparing results for query: {query}")

    # Unranked results
    current_method = engine.ranking_method
    engine.set_ranking_method("none")
    unranked_results, unranked_time, total_matches = engine.search(query)

    # Ranked results (restore original method)
    engine.set_ranking_method(current_method)
    ranked_results, ranked_time, _ = engine.search(query)

    print(f"Found {total_matches} matching documents")
    print(f"Unranked search time: {unranked_time:.5f} seconds")
    print(f"Ranked search time ({engine.ranking_method}): {ranked_time:.5f} seconds")

    print("\nTop 5 results comparison:")
    print(f"{'Rank':<5} {'Unranked':<40} {'Ranked ({0})':<40}".format(engine.ranking_method))
    print("-" * 85)

    for i in range(min(5, max(len(ranked_results), len(unranked_results)))):
        unranked = os.path.basename(unranked_results[i]) if i < len(unranked_results) else "-"
        ranked = os.path.basename(ranked_results[i]) if i < len(ranked_results) else "-"
        print(f"{i + 1:<5} {unranked:<40} {ranked:<40}")


def evaluate_ranking_methods(engine, test_queries):
    """Evaluate different ranking methods on a set of test queries"""
    print("\n=== Ranking Methods Evaluation ===")

    methods = ["none", "tfidf", "bm25", "vector"]
    results = {}

    for method in methods:
        engine.set_ranking_method(method)

        method_times = []
        for query in test_queries:
            _, search_time, _ = engine.search(query)
            method_times.append(search_time)

        avg_time = sum(method_times) / len(method_times)
        results[method] = avg_time

    print("\nAverage search time by ranking method:")
    print(f"{'Method':<10} {'Time (seconds)':<15}")
    print("-" * 25)

    for method, avg_time in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:<10} {avg_time:.6f}")


def main():
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    input_directory = config['Settings']['input_directory']
    index_file = config['Settings']['output_file']
    enhanced_file = "enhanced_index.bin"

    # Check if enhanced index already exists, otherwise build it
    if not os.path.exists(enhanced_file):
        print("Building enhanced index...")
        enhanced_index = build_enhanced_index(index_file, input_directory, enhanced_file)
    else:
        print(f"Loading enhanced index from {enhanced_file}...")
        enhanced_index = EnhancedCompressedIndex.load(enhanced_file)

    # Create the search engine with enhanced index
    search_engine = RankedSearchEngine(enhanced_index)

    # Evaluate different ranking methods
    test_queries = [
        "meeting AND schedule",
        "important OR urgent",
        "report AND data",
        "conference AND NOT meeting",
        "email"
    ]
    evaluate_ranking_methods(search_engine, test_queries)

    # Run interactive search
    interactive_search(search_engine, "Ranked Search")


if __name__ == "__main__":
    main()