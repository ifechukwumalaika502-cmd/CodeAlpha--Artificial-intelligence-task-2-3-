import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict
import time

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

class ChatbotFAQMatcher:
    """
    Intelligent Chatbot that matches user questions with FAQs and responds naturally
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
        
        # Conversation history
        self.conversation_history = []
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using spaCy"""
        doc = nlp(text.lower())
        processed_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(processed_tokens)
    
    def cosine_similarity_match(self, user_question: str) -> Tuple[int, float]:
        """Get best match using TF-IDF Cosine Similarity"""
        processed_question = self.preprocess_text(user_question)
        user_tfidf = self.vectorizer.transform([processed_question])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)[0]
        best_idx = np.argmax(similarities)
        return self.faq_ids[best_idx], similarities[best_idx]
    
    def intent_matching(self, user_question: str) -> Tuple[int, float]:
        """Get best match using Intent Matching"""
        user_doc = nlp(user_question)
        user_nouns = set([chunk.root.text.lower() for chunk in user_doc.noun_chunks])
        user_verbs = set([token.text.lower() for token in user_doc if token.pos_ == "VERB"])
        
        best_score = 0
        best_faq_id = 1
        
        for faq_id, faq in self.faqs.items():
            faq_doc = nlp(faq["question"])
            faq_nouns = set([chunk.root.text.lower() for chunk in faq_doc.noun_chunks])
            faq_verbs = set([token.text.lower() for token in faq_doc if token.pos_ == "VERB"])
            
            noun_similarity = len(user_nouns & faq_nouns) / max(len(user_nouns | faq_nouns), 1)
            verb_similarity = len(user_verbs & faq_verbs) / max(len(user_verbs | faq_verbs), 1)
            intent_score = (noun_similarity * 0.6 + verb_similarity * 0.4)
            
            if intent_score > best_score:
                best_score = intent_score
                best_faq_id = faq_id
        
        return best_faq_id, best_score
    
    def get_best_match(self, user_question: str) -> Dict:
        """
        Get the best matching FAQ using hybrid approach
        """
        cosine_id, cosine_score = self.cosine_similarity_match(user_question)
        intent_id, intent_score = self.intent_matching(user_question)
        
        # Weighted combination
        combined_scores = {}
        combined_scores[cosine_id] = combined_scores.get(cosine_id, 0) + (cosine_score * 0.6)
        combined_scores[intent_id] = combined_scores.get(intent_id, 0) + (intent_score * 0.4)
        
        best_faq_id = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_faq_id]
        
        return {
            "faq_id": best_faq_id,
            "confidence": confidence,
            "question": self.faqs[best_faq_id]["question"],
            "answer": self.faqs[best_faq_id]["answer"]
        }
    
    def typing_effect(self, text: str, delay: float = 0.03):
        """Simulate typing effect for chatbot response"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def display_chatbot_response(self, user_question: str):
        """Display response in chatbot format"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_question})
        
        # Get best match
        match = self.get_best_match(user_question)
        
        # Chatbot response header
        print("\n" + "╔" + "═" * 98 + "╗")
        print("║" + " " * 98 + "║")
        print("║" + "🤖 AI Automation Assistant".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("╚" + "═" * 98 + "╝")
        
        # User message
        print("\n👤 You:")
        print("   " + user_question)
        
        # Confidence indicator
        confidence_percent = int(match["confidence"] * 100)
        confidence_bar = "█" * (confidence_percent // 5) + "░" * ((100 - confidence_percent) // 5)
        print(f"\n⚡ Confidence: [{confidence_bar}] {confidence_percent}%")
        
        # Chatbot typing indicator
        print("\n🤖 Assistant:")
        print("   ", end="")
        self.typing_effect(match["answer"], delay=0.01)
        
        # Add to history
        self.conversation_history.append({"role": "assistant", "content": match["answer"]})
        
        # Related FAQ
        print("\n📌 Related Question: " + match["question"])
        
        # Helpful suggestions
        print("\n💡 Other FAQ Topics:")
        other_faqs = [faq_id for faq_id in self.faq_ids if faq_id != match["faq_id"]]
        for i, faq_id in enumerate(other_faqs[:3], 1):
            print(f"   {i}. {self.faqs[faq_id]['question']}")
        
        print("\n" + "─" * 100)
    
    def display_welcome_message(self):
        """Display welcome message"""
        print("\n")
        print("╔" + "═" * 98 + "╗")
        print("║" + " " * 98 + "║")
        print("║" + "🚀 AI AUTOMATION FAQ CHATBOT 🚀".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("║" + "Welcome! I'm here to help you with AI Automation questions.".center(98) + "║")
        print("║" + "Ask me anything about AI Automation, and I'll find the best answer for you!".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("║" + "Type 'quit' to exit, 'history' to see conversation history".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("╚" + "═" * 98 + "╝")
    
    def display_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("\n📋 No conversation history yet!")
            return
        
        print("\n" + "╔" + "═" * 98 + "╗")
        print("║" + " " * 98 + "║")
        print("║" + "📋 CONVERSATION HISTORY".center(98) + "║")
        print("║" + " " * 98 + "║")
        print("╚" + "═" * 98 + "╝\n")
        
        for i, msg in enumerate(self.conversation_history, 1):
            if msg["role"] == "user":
                print(f"{i}. 👤 You:\n   {msg['content']}\n")
            else:
                print(f"{i}. 🤖 Assistant:\n   {msg['content']}\n")
        
        print("─" * 100)
    
    def run_interactive_chatbot(self):
        """Run interactive chatbot"""
        self.display_welcome_message()
        
        # Sample questions for demonstration
        sample_questions = [
            "Can AI automation replace human jobs?",
            "What are the costs of implementing AI automation?",
            "How do I ensure data privacy with AI?"
        ]
        
        print("\n\n📝 DEMONSTRATION WITH SAMPLE QUESTIONS:")
        print("=" * 100)
        
        for question in sample_questions:
            self.display_chatbot_response(question)
            time.sleep(1)
        
        # Interactive mode
        print("\n\n" + "═" * 100)
        print("💬 NOW IT'S YOUR TURN - ASK ANYTHING ABOUT AI AUTOMATION")
        print("═" * 100)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("\n" + "╔" + "═" * 98 + "╗")
                print("║" + " " * 98 + "║")
                print("║" + "Thank you for using AI Automation FAQ Chatbot! Goodbye! 👋".center(98) + "║")
                print("║" + " " * 98 + "║")
                print("╚" + "═" * 98 + "╝\n")
                break
            elif user_input.lower() == 'history':
                self.display_history()
            elif user_input:
                self.display_chatbot_response(user_input)
            else:
                print("⚠️  Please enter a valid question.")

def main():
    """Main function"""
    chatbot = ChatbotFAQMatcher(FAQS_DATABASE)
    chatbot.run_interactive_chatbot()

if __name__ == "__main__":
    main()
