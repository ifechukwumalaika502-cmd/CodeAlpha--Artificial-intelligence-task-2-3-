import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# AI Automation FAQs Database
FAQS_DATABASE = {
    1: {
        "question": "What is AI automation?",
        "answer": "AI automation combines artificial intelligence technologies (like machine learning, natural language processing, or computer vision) with automation tools. This enables systems to make decisions, perform tasks, or optimize processes with minimal human intervention."
    },
    2: {
        "question": "How does AI automation differ from traditional automation?",
        "answer": "Traditional automation follows pre-defined, rule-based scripts to perform tasks. AI automation can analyze data, learn from patterns, adapt to changes, and handle exceptions—making it more flexible, scalable, and intelligent."
    },
    3: {
        "question": "What are common use cases for AI automation?",
        "answer": "Customer support chatbots, automated data entry and document processing, predictive maintenance in manufacturing, email sorting and sentiment analysis, fraud detection in banking."
    },
    4: {
        "question": "What are the benefits of AI automation?",
        "answer": "Increases efficiency and productivity, reduces operational costs, improves accuracy and reduces human error, enables 24/7 operations, frees employees for higher-value tasks."
    },
    5: {
        "question": "Does AI automation replace human jobs?",
        "answer": "AI automation can replace repetitive and mundane tasks but often complements human work. Many organizations use AI to augment their workforce, allowing employees to focus on creative, strategic, or customer-facing activities."
    },
    6: {
        "question": "What are the challenges in implementing AI automation?",
        "answer": "Integrating with legacy systems, data quality and availability, change management and employee training, monitoring and managing bias in AI models, ensuring security and compliance."
    },
    7: {
        "question": "How do I know if my business is ready for AI automation?",
        "answer": "Consider automation if your business has repetitive tasks that follow clear patterns, high volumes of data, a willingness to invest in new technology and upskilling staff, and clear business objectives for automation."
    },
    8: {
        "question": "Is it expensive to implement AI automation?",
        "answer": "Costs vary depending on complexity and scope. Cloud-based AI services and prebuilt tools can minimize upfront expenses. Start with a pilot and scale as you realize value."
    },
    9: {
        "question": "How can I ensure data privacy when using AI automation?",
        "answer": "Use data encryption and secure access controls, anonymize sensitive information, choose vendors with strong privacy policies, regularly audit and monitor AI systems."
    },
    10: {
        "question": "How do I get started with AI automation?",
        "answer": "Identify high-impact processes suitable for automation, assess available AI tools and platforms, start with a pilot project, measure results and iterate, train staff and communicate changes."
    }
}

class FAQMatcher:
    """
    Matches user questions with the most similar FAQ using multiple techniques:
    1. Cosine Similarity (TF-IDF)
    2. Intent Matching with spaCy
    3. Semantic Similarity (Word embeddings)
    """
    
    def __init__(self, faqs: Dict):
        self.faqs = faqs
        self.faq_questions = [faq["question"] for faq in faqs.values()]
        self.faq_ids = list(faqs.keys())
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=100
        )
        
        # Fit vectorizer on FAQ questions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.faq_questions)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using spaCy (tokenization, lemmatization)
        """
        doc = nlp(text.lower())
        processed_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(processed_tokens)
    
    def cosine_similarity_match(self, user_question: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Match user question using TF-IDF Cosine Similarity
        
        Returns:
            List of tuples: (faq_id, similarity_score, faq_question)
        """
        # Preprocess user question
        processed_question = self.preprocess_text(user_question)
        
        # Transform user question using the same vectorizer
        user_tfidf = self.vectorizer.transform([processed_question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)[0]
        
        # Get top K matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = [
            (self.faq_ids[idx], similarities[idx], self.faq_questions[idx])
            for idx in top_indices
        ]
        
        return matches
    
    def intent_matching(self, user_question: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Match user question based on intent using spaCy NLP features
        - Extracts named entities, noun chunks, and verb patterns
        - Compares structural and semantic similarity
        
        Returns:
            List of tuples: (faq_id, similarity_score, faq_question)
        """
        user_doc = nlp(user_question)
        
        # Extract features from user question
        user_entities = set([ent.label_ for ent in user_doc.ents])
        user_nouns = set([chunk.root.text.lower() for chunk in user_doc.noun_chunks])
        user_verbs = set([token.text.lower() for token in user_doc if token.pos_ == "VERB"])
        
        similarities = []
        
        for faq_id, faq in self.faqs.items():
            faq_doc = nlp(faq["question"])
            
            # Extract features from FAQ question
            faq_entities = set([ent.label_ for ent in faq_doc.ents])
            faq_nouns = set([chunk.root.text.lower() for chunk in faq_doc.noun_chunks])
            faq_verbs = set([token.text.lower() for token in faq_doc if token.pos_ == "VERB"])
            
            # Calculate similarity based on overlapping features
            noun_similarity = len(user_nouns & faq_nouns) / max(len(user_nouns | faq_nouns), 1)
            verb_similarity = len(user_verbs & faq_verbs) / max(len(user_verbs | faq_verbs), 1)
            entity_similarity = len(user_entities & faq_entities) / max(len(user_entities | faq_entities), 1)
            
            # Combined intent similarity score
            intent_score = (noun_similarity * 0.5 + verb_similarity * 0.3 + entity_similarity * 0.2)
            similarities.append((faq_id, intent_score, faq["question"]))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def semantic_similarity_match(self, user_question: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Match user question using spaCy word vectors (semantic similarity)
        Compares the meaning of questions based on word embeddings
        
        Returns:
            List of tuples: (faq_id, similarity_score, faq_question)
        """
        user_doc = nlp(user_question)
        
        similarities = []
        
        for faq_id, faq in self.faqs.items():
            faq_doc = nlp(faq["question"])
            
            # Calculate semantic similarity using word vectors
            if user_doc.has_vector and faq_doc.has_vector:
                semantic_score = user_doc.similarity(faq_doc)
            else:
                semantic_score = 0.0
            
            similarities.append((faq_id, semantic_score, faq["question"]))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def hybrid_match(self, user_question: str, top_k: int = 3) -> Dict:
        """
        Combine multiple matching techniques for robust results
        Uses weighted average of all methods
        
        Returns:
            Dictionary with results from all methods and final recommendation
        """
        # Get matches from all methods
        cosine_matches = self.cosine_similarity_match(user_question, top_k=5)
        intent_matches = self.intent_matching(user_question, top_k=5)
        semantic_matches = self.semantic_similarity_match(user_question, top_k=5)
        
        # Create scoring dictionary
        scores = {}
        
        # Add cosine similarity scores (weight: 0.4)
        for faq_id, score, _ in cosine_matches:
            scores[faq_id] = scores.get(faq_id, 0) + (score * 0.4)
        
        # Add intent matching scores (weight: 0.35)
        for faq_id, score, _ in intent_matches:
            scores[faq_id] = scores.get(faq_id, 0) + (score * 0.35)
        
        # Add semantic similarity scores (weight: 0.25)
        for faq_id, score, _ in semantic_matches:
            scores[faq_id] = scores.get(faq_id, 0) + (score * 0.25)
        
        # Sort by combined score
        hybrid_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return {
            "cosine_similarity": cosine_matches,
            "intent_matching": intent_matches,
            "semantic_similarity": semantic_matches,
            "hybrid_results": hybrid_results,
            "top_match": hybrid_results[0] if hybrid_results else None
        }

def display_results(user_question: str, matcher: FAQMatcher):
    """
    Display comprehensive matching results
    """
    print("\n" + "=" * 100)
    print(f"USER QUESTION: {user_question}")
    print("=" * 100)
    
    # Get hybrid results
    results = matcher.hybrid_match(user_question, top_k=3)
    
    # 1. Cosine Similarity Results
    print("\n1. COSINE SIMILARITY MATCHING (TF-IDF)")
    print("-" * 100)
    for i, (faq_id, score, question) in enumerate(results["cosine_similarity"], 1):
        print(f"   {i}. [FAQ #{faq_id}] Similarity Score: {score:.4f}")
        print(f"      Q: {question}")
        print(f"      A: {matcher.faqs[faq_id]['answer'][:100]}...")
        print()
    
    # 2. Intent Matching Results
    print("\n2. INTENT MATCHING (spaCy NLP)")
    print("-" * 100)
    for i, (faq_id, score, question) in enumerate(results["intent_matching"], 1):
        print(f"   {i}. [FAQ #{faq_id}] Intent Score: {score:.4f}")
        print(f"      Q: {question}")
        print(f"      A: {matcher.faqs[faq_id]['answer'][:100]}...")
        print()
    
    # 3. Semantic Similarity Results
    print("\n3. SEMANTIC SIMILARITY MATCHING (Word Vectors)")
    print("-" * 100)
    for i, (faq_id, score, question) in enumerate(results["semantic_similarity"], 1):
        print(f"   {i}. [FAQ #{faq_id}] Semantic Score: {score:.4f}")
        print(f"      Q: {question}")
        print(f"      A: {matcher.faqs[faq_id]['answer'][:100]}...")
        print()
    
    # 4. Hybrid Results (Final Recommendation)
    print("\n4. HYBRID MATCHING (Combined Weighted Score)")
    print("-" * 100)
    for i, (faq_id, score) in enumerate(results["hybrid_results"], 1):
        print(f"   {i}. [FAQ #{faq_id}] Combined Score: {score:.4f}")
        print(f"      Q: {matcher.faqs[faq_id]['question']}")
        print(f"      A: {matcher.faqs[faq_id]['answer']}")
        print()
    
    # 5. Top Match Recommendation
    print("\n5. TOP RECOMMENDED MATCH")
    print("-" * 100)
    if results["top_match"]:
        faq_id, score = results["top_match"]
        print(f"FAQ #{faq_id} (Combined Score: {score:.4f})")
        print(f"\nQuestion: {matcher.faqs[faq_id]['question']}")
        print(f"\nAnswer: {matcher.faqs[faq_id]['answer']}")

def main():
    """
    Main function to demonstrate FAQ matching with sample user questions
    """
    print("\n" + "=" * 100)
    print("FAQ MATCHING SYSTEM - AI AUTOMATION")
    print("Matching user questions with most similar FAQs using multiple techniques")
    print("=" * 100)
    
    # Initialize FAQ Matcher
    matcher = FAQMatcher(FAQS_DATABASE)
    
    # Sample user questions to test
    sample_questions = [
        "Can AI automation replace jobs?",
        "What are the costs associated with AI automation implementation?",
        "How do I protect data privacy with AI systems?",
        "What is machine learning automation?",
        "Is AI automation secure?",
        "How to start implementing automation with AI?",
        "What problems can automation solve in manufacturing?",
    ]
    
    # Test with each sample question
    for user_question in sample_questions:
        display_results(user_question, matcher)
    
    # Interactive mode
    print("\n\n" + "=" * 100)
    print("INTERACTIVE MODE")
    print("=" * 100)
    print("\nEnter your own questions about AI Automation (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Your Question: ").strip()
        if user_input.lower() == 'quit':
            print("\nThank you for using the FAQ Matching System!")
            break
        if user_input:
            display_results(user_input, matcher)

if __name__ == "__main__":
    main()
