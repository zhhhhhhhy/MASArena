# coding: utf-8
from typing import List

from langchain.tools import StructuredTool

from benchmark.src.tools.base import ToolFactory

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

DOCUMENT_ANALYSIS = "document_analysis"

class DocumentAnalysis:
    def __init__(self):
        self.documents = {}

    def read_document(self, file_path: str) -> str:
        """Read a document from a file path."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents[file_path] = content
                return f"Successfully read document: {file_path}"
        except Exception as e:
            return f"Error reading document: {e}"

    def analyze_document(self, file_path: str, query: str = None) -> str:
        """
        Analyzes a document by extracting key sentences.
        If a query is provided, it finds sentences relevant to the query.
        Otherwise, it generates a generic summary.
        """
        if file_path not in self.documents:
            return f"Error: Document not found. Please read the document first: {file_path}"
        
        if not NLTK_AVAILABLE:
            return "Error: NLTK is not installed. Analysis functions are unavailable. Please install it (`pip install nltk`)."

        text = self.documents[file_path]
        sentences = sent_tokenize(text)

        if not sentences:
            return "Document is empty or could not be parsed into sentences."

        if query:
            query_words = set(word.lower() for word in word_tokenize(query))
            relevant_sentences = [s for s in sentences if any(w in s.lower() for w in query_words)]
            if not relevant_sentences:
                return f"No sentences found related to the query: '{query}'"
            return "Relevant Sentences:\n" + "\n".join(f"- {s}" for s in relevant_sentences[:5]) # Return top 5

        # Generic summary if no query
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        word_frequencies = {}
        for word in words:
            if word.isalnum() and word not in stop_words:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        if not word_frequencies:
            return "Document contains no significant words for summarization."
            
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = (word_frequencies[word] / max_frequency)

        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

        # Select top 3 sentences for summary
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]
        return "Summary:\n" + "\n".join(f"- {s}" for s in summary_sentences)

@ToolFactory.register(name=DOCUMENT_ANALYSIS, desc="A tool for analyzing documents.")
class DocumentAnalysisTool:
    def __init__(self):
        self.document_analysis = DocumentAnalysis()

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.document_analysis.read_document,
                name="read_document",
                description="Read a document from a file path into memory.",
            ),
            StructuredTool.from_function(
                func=self.document_analysis.analyze_document,
                name="analyze_document",
                description="Analyzes a document to extract key sentences or query for relevant information.",
            ),
        ] 