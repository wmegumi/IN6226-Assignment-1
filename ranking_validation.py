import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer


# Basic utility functions
def read_file(file_path):
    """Read file content"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()


def normalize_token(tokens):
    """Normalize tokens and apply stemming"""
    stemmer = PorterStemmer()
    normalized_tokens = []
    for token in tokens:
        token = token.lower()
        token = token.strip()
        if token:
            normalized_tokens.append(stemmer.stem(token))
    return normalized_tokens


def tokenize(text, doc_id):
    """Tokenize text and associate with document ID"""
    tokens = text.split()
    tokens = normalize_token(tokens)
    return [(token, doc_id) for token in tokens if token]


def list_files(directory):
    """List all files in a directory"""
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


# Test document collection creation
def create_test_collection():
    """Create test document collection with known characteristics"""
    test_dir = "test_collection"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # Documents designed to test various ranking scenarios
    documents = {
        "doc1.txt": "apple apple apple banana",  # High TF(apple)
        "doc2.txt": "apple banana banana banana",  # High TF(banana)
        "doc3.txt": "apple banana orange grape",  # Term diversity
        "doc4.txt": "apple banana cherry date elderberry fig grape",  # Long document
        "doc5.txt": "rare_term apple banana",  # Contains rare term
        "doc8.txt": "common common common rare_term",  # Mix of common and rare
        "doc9.txt": "apple apple",  # Short document, high TF
        "doc10.txt": "apple banana common"  # Mixed document
    }

    # Create files
    for filename, content in documents.items():
        with open(os.path.join(test_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    # Create feature records for validation
    doc_features = {}
    for filename, content in documents.items():
        tokens = content.lower().split()
        stemmed_tokens = normalize_token(tokens)
        token_counts = Counter(stemmed_tokens)

        doc_features[filename] = {
            "apple_count": token_counts.get(normalize_token(["apple"])[0], 0),
            "banana_count": token_counts.get(normalize_token(["banana"])[0], 0),
            "rare_term_count": token_counts.get(normalize_token(["rare_term"])[0], 0),
            "total_terms": len(stemmed_tokens),
            "unique_terms": len(token_counts),
            "has_rare_term": normalize_token(["rare_term"])[0] in stemmed_tokens
        }

    return test_dir, list(documents.keys()), doc_features


# Enhanced index implementation
class TestEnhancedIndex:
    """Simplified enhanced index for testing ranking algorithms"""

    def __init__(self):
        self.index = {}  # term -> document ID list
        self.doc_terms = {}  # document ID -> {term -> frequency}
        self.doc_lengths = {}  # document ID -> length
        self.term_df = {}  # term -> document frequency
        self.doc_count = 0  # total number of documents
        self.all_doc_ids = set()  # all document IDs
        self.avg_doc_length = 0  # average document length

    def build_from_files(self, test_dir):
        """Build index from test files"""
        files = list_files(test_dir)
        self.doc_count = len(files)
        self.all_doc_ids = set(files)

        total_tokens = 0

        # Process each document
        for doc_id in files:
            text = read_file(doc_id)
            tokens = [token for token, _ in tokenize(text, doc_id)]

            # Document length
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_tokens += doc_length

            # Term frequency
            term_freq = Counter(tokens)
            self.doc_terms[doc_id] = dict(term_freq)

            # Document frequency
            for term in set(tokens):
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(doc_id)
                self.term_df[term] = self.term_df.get(term, 0) + 1

        # Average document length
        self.avg_doc_length = total_tokens / max(1, self.doc_count)
        return self

    def get_postings(self, term):
        """Get posting list for a term"""
        return self.index.get(term, [])

    def get_document_frequency(self, term):
        """Get document frequency for a term"""
        return self.term_df.get(term, 0)

    def get_term_frequency(self, term, doc_id):
        """Get term frequency in a document"""
        if doc_id in self.doc_terms:
            return self.doc_terms[doc_id].get(term, 0)
        return 0

    def get_document_length(self, doc_id):
        """Get document length"""
        return self.doc_lengths.get(doc_id, 0)


# Ranking engine implementation
class TestRankedSearchEngine:
    """Search engine implementing multiple ranking algorithms"""

    def __init__(self, index):
        self.index = index
        self.ranking_method = "tfidf"
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75

    def set_ranking_method(self, method):
        """Set the ranking method"""
        self.ranking_method = method

    def search(self, query):
        """Execute query and return ranked results"""
        results, scores, count = self.search_with_scores(query)
        return results, 0.001, count

    def search_with_scores(self, query):
        """Execute query and return ranked results with scores"""
        # Tokenize
        query_terms = normalize_token(query.split())
        query_term_counts = Counter(query_terms)

        # Get matching documents
        matching_docs = set()
        for term in query_terms:
            matching_docs.update(self.index.get_postings(term))

        if not matching_docs:
            return [], {}, 0

        # Calculate scores based on method
        if self.ranking_method == "tfidf":
            scores = self._score_tfidf(matching_docs, query_term_counts)
        elif self.ranking_method == "bm25":
            scores = self._score_bm25(matching_docs, query_term_counts)
        elif self.ranking_method == "vector":
            scores = self._score_vector_space(matching_docs, query_term_counts)
        else:
            scores = {doc_id: 1.0 for doc_id in matching_docs}

        # Sort results
        results = sorted(matching_docs, key=lambda doc_id: scores.get(doc_id, 0.0), reverse=True)
        return results, scores, len(matching_docs)

    def _score_tfidf(self, docs, query_terms):
        """TF-IDF scoring algorithm"""
        scores = {}
        N = self.index.doc_count

        for doc_id in docs:
            score = 0.0
            for term, query_count in query_terms.items():
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue

                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                idf = math.log(N / df) if df > 0 else 0
                score += tf * idf * query_count

            scores[doc_id] = score

        return scores

    def _score_bm25(self, docs, query_terms):
        """BM25 scoring algorithm"""
        scores = {}
        N = self.index.doc_count
        avdl = self.index.avg_doc_length
        k1 = self.bm25_k1
        b = self.bm25_b

        for doc_id in docs:
            score = 0.0
            doc_len = self.index.get_document_length(doc_id)
            if doc_len == 0:
                continue

            for term, query_count in query_terms.items():
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue

                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                # BM25 formula
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                tf_component = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (doc_len / avdl)) + tf)

                score += idf * tf_component * query_count

            scores[doc_id] = score

        return scores

    def _score_vector_space(self, docs, query_terms):
        """Vector space model scoring algorithm"""
        scores = {}
        N = self.index.doc_count

        # Build query vector
        query_vector = {}
        query_magnitude = 0.0

        for term, count in query_terms.items():
            df = self.index.get_document_frequency(term)
            if df == 0:
                continue

            idf = math.log(N / df) if df > 0 else 0
            weight = count * idf
            query_vector[term] = weight
            query_magnitude += weight ** 2

        query_magnitude = math.sqrt(query_magnitude)

        # Calculate document vectors and cosine similarity
        for doc_id in docs:
            dot_product = 0.0
            doc_magnitude = 0.0

            for term in query_terms:
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue

                df = self.index.get_document_frequency(term)
                if df == 0:
                    continue

                idf = math.log(N / df) if df > 0 else 0
                doc_weight = tf * idf

                dot_product += doc_weight * query_vector.get(term, 0)
                doc_magnitude += doc_weight ** 2

            doc_magnitude = math.sqrt(doc_magnitude)

            if doc_magnitude > 0 and query_magnitude > 0:
                similarity = dot_product / (doc_magnitude * query_magnitude)
            else:
                similarity = 0.0

            scores[doc_id] = similarity

        return scores


# Robust validation methods
class RobustValidation:
    """Provides multiple robust ranking validation methods"""

    @staticmethod
    def validate_principles(search_engine, doc_features, query, ranking_method):
        """Validate ranking principles rather than exact order"""
        print(f"\nValidating {ranking_method.upper()} ranking principles - Query: '{query}'")

        # Execute search
        search_engine.set_ranking_method(ranking_method)
        results, _, _ = search_engine.search(query)
        result_files = [os.path.basename(doc) for doc in results[:5]]

        print(f"  Top 5 results: {result_files}")

        # 1. Validate basic principles
        if "apple" in query.lower() and "rare" not in query.lower():
            # High TF documents should rank higher
            if ranking_method !="vector":

                apple_docs = [doc for doc in result_files if doc in ["doc1.txt", "doc9.txt"]]
                print(
                    f"  Principle 1: High apple frequency docs rank highly: {'PASS' if apple_docs and apple_docs[0] in result_files[:3] else 'FAIL'}")

        if "rare_term" in query.lower():
            # All documents with rare terms should appear
            rare_docs = [doc for doc in result_files if doc in ["doc5.txt", "doc8.txt"]]
            print(f"  Principle 2: All docs with rare terms appear: {'PASS' if len(rare_docs) == 2 else 'FAIL'}")

        if "apple" in query.lower() and "banana" in query.lower():
            # Documents containing both terms should rank higher
            both_docs = []
            for doc in result_files:
                if doc in ["doc1.txt", "doc2.txt", "doc3.txt", "doc5.txt", "doc10.txt"]:
                    both_docs.append(doc)

            # Check if they appear in top 3
            high_rank = any(doc in result_files[:3] for doc in both_docs)
            print(f"  Principle 3: Docs with both query terms rank highly: {'PASS' if high_rank else 'FAIL'}")

        # 2. Algorithm-specific principles
        if ranking_method == "tfidf":
            # High TF documents should rank higher
            if "apple" in query.lower() and "rare" not in query.lower():
                doc1_pos = result_files.index("doc1.txt") if "doc1.txt" in result_files else 999
                doc3_pos = result_files.index("doc3.txt") if "doc3.txt" in result_files else 999
                if doc1_pos < 999 and doc3_pos < 999:
                    print(
                        f"  TF-IDF Principle: High TF priority (doc1 vs doc3): {'PASS' if doc1_pos < doc3_pos else 'FAIL'}")

        # elif ranking_method == "bm25":
        #     # Short documents should get a boost
        #     if "apple" in query.lower():
        #         doc9_pos = result_files.index("doc9.txt") if "doc9.txt" in result_files else 999
        #         print(
        #             f"  BM25 Principle: Short document advantage: {'PASS' if doc9_pos != 999 and doc9_pos <= 2 else 'FAIL'}")

        elif ranking_method == "vector":
            # Vector space model should consider term similarity
            if "apple" in query.lower() and "banana" in query.lower():
                # Check if at least one document with diverse terms appears in top results
                diverse_docs = [doc for doc in result_files[:3]
                                if doc in ["doc3.txt", "doc4.txt", "doc10.txt"]]
                print(f"  Vector Space Principle: Term diversity advantage: {'PASS' if diverse_docs else 'FAIL'}")

    @staticmethod
    def validate_statistical_properties(search_engine, test_queries):
        """Validate statistical properties of ranking algorithms"""
        print("\nValidating statistical properties of ranking algorithms:")

        methods = ["tfidf", "bm25", "vector"]

        for method in methods:
            search_engine.set_ranking_method(method)
            scores = []

            print(f"\n{method.upper()} ranking statistics:")

            for query in test_queries:
                # Get score data
                results, scores_dict, _ = search_engine.search_with_scores(query)

                if scores_dict:
                    # Analyze score distribution
                    values = list(scores_dict.values())
                    scores.extend(values)

                    print(
                        f"  Query '{query}': highest {max(values):.4f}, lowest {min(values):.4f}, range {max(values) - min(values):.4f}")

                    # Verify score distinction
                    score_diff = max(values) - min(values)
                    has_distinction = score_diff > 0.01
                    print(f"    Distinction: {'sufficient' if has_distinction else 'insufficient'}")

                    # Check vector space model score range
                    if method == "vector":
                        valid_range = all(0 <= s <= 1.001 for s in values)  # Allow small floating-point errors
                        print(f"    Score range: {'correct (0-1)' if valid_range else 'incorrect'}")

            # Overall score distribution
            if scores:
                mean = sum(scores) / len(scores)
                std = np.std(scores) if len(scores) > 1 else 0
                print(f"  Overall scores: mean {mean:.4f}, std dev {std:.4f}")
                print(f"  {'Good score distribution' if std > 0.01 else 'Score distribution too concentrated'}")

    @staticmethod
    def visualize_scores(search_engine, query):
        """Visualize score distribution for different algorithms"""
        methods = ["tfidf", "bm25", "vector"]
        plt.figure(figsize=(15, 5))

        for i, method in enumerate(methods):
            search_engine.set_ranking_method(method)
            results, scores_dict, _ = search_engine.search_with_scores(query)

            # Sort results and scores
            sorted_results = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            docs = [os.path.basename(doc)[:6] for doc, _ in sorted_results[:10]]
            scores = [score for _, score in sorted_results[:10]]

            # Normalize scores for comparison
            if scores:
                max_score = max(scores)
                if max_score > 0:
                    scores = [s / max_score for s in scores]

            # Create bar chart
            plt.subplot(1, 3, i + 1)
            plt.bar(docs, scores)
            plt.title(f"{method.upper()}")
            plt.xticks(rotation=45)
            plt.ylabel("Normalized Score")

        plt.tight_layout()
        plt.savefig(f"query_{query.replace(' ', '_')}_scores.png")
        print(f"Score distribution chart saved as query_{query.replace(' ', '_')}_scores.png")

    @staticmethod
    def test_ranking_stability(search_engine, query, iterations=10):
        """Test stability of ranking results"""
        print(f"\nTesting stability for query '{query}' ({iterations} iterations):")

        methods = ["tfidf", "bm25", "vector"]
        stability = {}

        for method in methods:
            search_engine.set_ranking_method(method)
            all_results = []

            # Run the same query multiple times
            for i in range(iterations):
                results, _, _ = search_engine.search(query)
                top_5 = [os.path.basename(doc) for doc in results[:5]]
                all_results.append(top_5)

            # Calculate stability of top 5 positions
            stable_positions = [0] * 5

            for pos in range(5):
                if all(len(run) > pos and run[pos] == all_results[0][pos] for run in all_results):
                    stable_positions[pos] = 1

            stability_score = sum(stable_positions) / 5
            stability[method] = stability_score

            print(f"  {method.upper()} stability: {stability_score * 100:.1f}%")

            # If not completely stable, show variations
            if stability_score < 1.0:
                print("  Position variations:")
                for pos in range(5):
                    if stable_positions[pos] == 0:
                        variations = set(run[pos] for run in all_results if len(run) > pos)
                        print(f"    Position {pos + 1}: {variations}")

        return stability


# Complete validation workflow
def run_robust_validation():
    """Run complete robust ranking validation"""
    print("\n=== Robust Ranking Algorithm Validation ===")

    # 1. Create test collection
    print("\n1. Creating test collection...")
    test_dir, doc_names, doc_features = create_test_collection()

    # 2. Build index
    print("\n2. Building test index...")
    enhanced_index = TestEnhancedIndex().build_from_files(test_dir)
    search_engine = TestRankedSearchEngine(enhanced_index)

    # 3. Show test document features
    print("\n3. Test document features:")
    print(f"{'Document':<10} {'Length':<8} {'Apple#':<8} {'Banana#':<8} {'RareTerm':<8} {'Unique#':<8}")
    print("-" * 50)
    for doc_name, features in doc_features.items():
        print(f"{doc_name:<10} {features['total_terms']:<8} {features['apple_count']:<8} "
              f"{features['banana_count']:<8} {features['rare_term_count']:<8} {features['unique_terms']:<8}")

    # 4. Validate ranking principles
    print("\n4. Validating ranking principles...")
    test_queries = [
        "apple",
        "rare_term",
        "apple banana"
    ]

    for query in test_queries:
        for method in ["tfidf", "bm25", "vector"]:
            RobustValidation.validate_principles(search_engine, doc_features, query, method)

    # # 5. Validate statistical properties
    # print("\n5. Validating statistical properties...")
    # RobustValidation.validate_statistical_properties(search_engine, test_queries)

    # 6. Visualize score distributions
    print("\n5. Visualizing score distributions...")
    for query in test_queries:
        RobustValidation.visualize_scores(search_engine, query)


    print("\nValidation complete! Test collection located at:", test_dir)


# Run validation
if __name__ == "__main__":
    try:
        run_robust_validation()
    except Exception as e:
        print(f"Error occurred during validation: {e}")
        # Provide installation guide if missing necessary libraries
        print("\nYou may need to install these dependencies:")
        print("  pip install numpy matplotlib nltk")

        # Ensure NLTK data is downloaded
        print("\nFor first run, download NLTK data:")
        print("  import nltk")
        print("  nltk.download('punkt')")