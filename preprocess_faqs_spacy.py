import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# AI Automation FAQs Text
faqs_text = """
What is AI automation?
AI automation combines artificial intelligence technologies like machine learning, natural language processing, or computer vision with automation tools. This enables systems to make decisions, perform tasks, or optimize processes with minimal human intervention.

How does AI automation differ from traditional automation?
Traditional automation follows pre-defined, rule-based scripts to perform tasks. AI automation can analyze data, learn from patterns, adapt to changes, and handle exceptions—making it more flexible, scalable, and intelligent.

What are common use cases for AI automation?
Customer support chatbots, automated data entry and document processing, predictive maintenance in manufacturing, email sorting and sentiment analysis, fraud detection in banking.

What are the benefits of AI automation?
Increases efficiency and productivity, reduces operational costs, improves accuracy and reduces human error, enables 24/7 operations, frees employees for higher-value tasks.

Does AI automation replace human jobs?
AI automation can replace repetitive and mundane tasks but often complements human work. Many organizations use AI to augment their workforce, allowing employees to focus on creative, strategic, or customer-facing activities.

What are the challenges in implementing AI automation?
Integrating with legacy systems, data quality and availability, change management and employee training, monitoring and managing bias in AI models, ensuring security and compliance.

How do I know if my business is ready for AI automation?
Consider automation if your business has repetitive tasks that follow clear patterns, high volumes of data, a willingness to invest in new technology and upskilling staff, and clear business objectives for automation.

Is it expensive to implement AI automation?
Costs vary depending on complexity and scope. Cloud-based AI services and prebuilt tools can minimize upfront expenses. Start with a pilot and scale as you realize value.

How can I ensure data privacy when using AI automation?
Use data encryption and secure access controls, anonymize sensitive information, choose vendors with strong privacy policies, regularly audit and monitor AI systems.

How do I get started with AI automation?
Identify high-impact processes suitable for automation, assess available AI tools and platforms, start with a pilot project, measure results and iterate, train staff and communicate changes.
"""

def preprocess_text(text):
    """
    Preprocess text using spaCy with tokenization, lemmatization, 
    and removal of stop words and punctuation
    """
    doc = nlp(text)
    
    # Filter tokens: remove stop words and punctuation
    processed_tokens = [
        token.lemma_.lower() for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    
    return processed_tokens, doc

def extract_entities(doc):
    """
    Extract named entities from the document
    """
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    return entities

def extract_noun_phrases(doc):
    """
    Extract noun phrases (noun chunks) from the document
    """
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append({
            'text': chunk.text,
            'root': chunk.root.text,
            'root_pos': chunk.root.pos_
        })
    return noun_phrases

def analyze_pos_tags(doc):
    """
    Extract Part-of-Speech tags for all tokens
    """
    pos_tags = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            pos_tags.append({
                'text': token.text,
                'pos': token.pos_,
                'tag': token.tag_,
                'lemma': token.lemma_
            })
    return pos_tags

def main():
    print("=" * 80)
    print("TEXT PREPROCESSING WITH SPACY - AI AUTOMATION FAQs")
    print("=" * 80)
    
    # Preprocess the text
    print("\n1. PREPROCESSING (Tokenization, Lemmatization, Stop Word Removal)")
    print("-" * 80)
    processed_tokens, doc = preprocess_text(faqs_text)
    print(f"Original text length: {len(faqs_text)} characters")
    print(f"Number of tokens (original): {len(doc)}")
    print(f"Number of tokens (after preprocessing): {len(processed_tokens)}")
    print(f"\nProcessed tokens (first 50):\n{processed_tokens[:50]}")
    
    # Extract entities
    print("\n\n2. NAMED ENTITY RECOGNITION (NER)")
    print("-" * 80)
    entities = extract_entities(doc)
    if entities:
        print(f"Total entities found: {len(entities)}\n")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. Text: '{entity['text']}' | Type: {entity['label']}")
    else:
        print("No named entities found in the text.")
    
    # Extract noun phrases
    print("\n\n3. NOUN PHRASE EXTRACTION")
    print("-" * 80)
    noun_phrases = extract_noun_phrases(doc)
    print(f"Total noun phrases found: {len(noun_phrases)}\n")
    for i, phrase in enumerate(noun_phrases[:30], 1):  # Show first 30
        print(f"{i}. '{phrase['text']}' (root: {phrase['root']}, POS: {phrase['root_pos']})")
    
    # Part-of-Speech tagging
    print("\n\n4. PART-OF-SPEECH (POS) TAGGING")
    print("-" * 80)
    pos_tags = analyze_pos_tags(doc)
    print(f"Total tokens with POS tags: {len(pos_tags)}\n")
    print("Token | POS Tag | Fine-grained Tag | Lemma")
    print("-" * 50)
    for i, token_info in enumerate(pos_tags[:40], 1):  # Show first 40
        print(f"{token_info['text']:15} | {token_info['pos']:7} | {token_info['tag']:16} | {token_info['lemma']}")
    
    # Summary statistics
    print("\n\n5. PREPROCESSING SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total sentences: {len(list(doc.sents))}")
    print(f"Total tokens: {len(doc)}")
    print(f"Unique tokens (processed): {len(set(processed_tokens))}")
    print(f"Stop words removed: {sum(1 for token in doc if token.is_stop)}")
    print(f"Punctuation removed: {sum(1 for token in doc if token.is_punct)}")
    print(f"Named entities found: {len(entities)}")
    print(f"Noun phrases found: {len(noun_phrases)}")
    
    # Display sample sentences with their analysis
    print("\n\n6. SAMPLE SENTENCE ANALYSIS")
    print("-" * 80)
    for i, sent in enumerate(list(doc.sents)[:3], 1):  # Show first 3 sentences
        print(f"\nSentence {i}: {sent.text}")
        print("Tokens (with POS tags):")
        for token in sent:
            if not token.is_punct and not token.is_space:
                print(f"  {token.text:20} -> POS: {token.pos_:10} Lemma: {token.lemma_}")

if __name__ == "__main__":
    main()
