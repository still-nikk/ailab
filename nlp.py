import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from textblob import Word
import pandas as pd

# Download necessary NLTK data files (only once)
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


# Function to clean the text
def clean_text(text):
    # Remove punctuation, numbers, and special characters using regular expression
    text = re.sub(r"[^A-Za-z\s]", "", text)

    # Remove extra white spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Function to convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()


# Function to perform tokenization
def tokenize_text(text):
    # Tokenize the text into words
    return word_tokenize(text)


# Function to remove stop words
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


# Function to correct misspelled words
def correct_spelling(tokens):
    corrected_tokens = [Word(word).correct() for word in tokens]
    return corrected_tokens


# Function to perform stemming and lemmatization
def stem_and_lemmatize(text):
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Stemming and Lemmatization
    stemmed_words = [
        stemmer.stem(word) for word in tokens if word not in stopwords.words("english")
    ]
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stopwords.words("english")
    ]

    return stemmed_words, lemmatized_words


# Function to generate 3 consecutive words (trigrams) after lemmatization
def generate_trigrams(lemmatized_words):
    # Generate trigrams (3 consecutive words)
    trigrams = list(ngrams(lemmatized_words, 3))
    return trigrams


# Function to perform One-Hot Encoding
def one_hot_encode(text_list):
    vectorizer = OneHotEncoder(sparse_output=False)
    one_hot_matrix = vectorizer.fit_transform([[text] for text in text_list])
    return one_hot_matrix


# Function to perform Bag of Words (BOW)
def bag_of_words(text_list):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(text_list)
    return bow_matrix.toarray(), vectorizer.get_feature_names_out()


# Function to perform TF-IDF
def tfidf(text_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()


# Read the text from the sample files
with open("text_files/tech1.txt", "r") as file1, open(
    "text_files/tech2.txt", "r"
) as file2, open("text_files/tech3.txt", "r") as file3:
    text1 = file1.read()
    text2 = file2.read()
    text3 = file3.read()
with open("text_files/text_file.txt", "r") as file4:
    text_file = file4.read()


# Combine all three text files
combined_text = text1 + "\n" + text2 + "\n" + text3

# 1. Clean the text
cleaned_text = clean_text(text_file)

# 2. Convert the text to lowercase
lowercase_text = convert_to_lowercase(cleaned_text)

# 3. Perform Tokenization
tokens = tokenize_text(lowercase_text)

# 4. Remove stop words
tokens_no_stopwords = remove_stopwords(tokens)

# 5. Correct misspelled words
corrected_tokens = correct_spelling(tokens_no_stopwords)

# 6. Perform stemming and lemmatization
stemmed_words, lemmatized_words = stem_and_lemmatize(" ".join(corrected_tokens))

# 7. Create a list of 3 consecutive words after lemmatization (trigrams)
trigrams = generate_trigrams(lemmatized_words)

# 8. Perform One-Hot Encoding
one_hot_matrix = one_hot_encode(combined_text.split())
print("\nOne-Hot Encoding:")
print(one_hot_matrix)

# 9. Perform Bag of Words (BoW)
bow_matrix, bow_features = bag_of_words([combined_text])
print("\nBag of Words (BoW):")
print("Feature Names:", bow_features)
print("BoW Matrix:\n", bow_matrix)

# 10. Perform TF-IDF
tfidf_matrix, tfidf_features = tfidf([combined_text])
print("\nTF-IDF:")
print("Feature Names:", tfidf_features)
print("TF-IDF Matrix:\n", tfidf_matrix)

# Output the results
print("\nOriginal Text:\n", combined_text)
print("\nCleaned Text:\n", cleaned_text)
print("\nLowercase Text:\n", lowercase_text)
print("\nTokens:\n", tokens)
print("\nTokens without Stopwords:\n", tokens_no_stopwords)
print("\nCorrected Tokens:\n", corrected_tokens)
print("\nStemmed Words:\n", stemmed_words)
print("\nLemmatized Words:\n", lemmatized_words)
print("\nTrigrams:\n", trigrams)


"""
    ### **Assignment 22: Text Cleaning, Tokenization, Stop Words Removal, Misspelling Correction**
---

#### **Viva Questions and Answers:**
1. **What is the purpose of text cleaning in this assignment?**
   - **Answer:** It removes unwanted characters like punctuation, numbers, and extra spaces to make the text uniform for further processing.

2. **What is tokenization?**
   - **Answer:** Tokenization is the process of splitting a text into smaller units like words or phrases (tokens).

3. **What is the role of stop word removal?**
   - **Answer:** Stop words are common words like "the," "is," etc., that don't add significant meaning to the text. Removing them reduces noise in analysis.

4. **What is spell correction?**
   - **Answer:** Spell correction uses algorithms or libraries (like `TextBlob`) to correct common spelling mistakes in the text.

5. **What does `word_tokenize` do in tokenization?**
   - **Answer:** It splits the text into individual words using punctuation and whitespace as delimiters.

6. **Why do we use stemming and lemmatization in NLP?**
   - **Answer:** They reduce words to their root form to ensure different variations of a word (like "running" and "ran") are treated as the same word.

#### **Real-Life Application:**
- **Example:** In customer feedback analysis, cleaning text and removing stop words helps in accurately analyzing sentiments by focusing on important terms.
- **Application:** Used in sentiment analysis, spam filtering, and information retrieval systems.

---

### **Assignment 23: Stemming, Lemmatization, and Trigrams**
---

#### **Viva Questions and Answers:**
1. **What is the difference between stemming and lemmatization?**
   - **Answer:** Stemming reduces words to their base form (e.g., "running" → "run"), while lemmatization reduces words to their root word (e.g., "better" → "good").

2. **What is the purpose of generating trigrams?**
   - **Answer:** Trigrams help in understanding the context by analyzing sequences of three consecutive words in the text, which is useful for tasks like predictive text or language modeling.

3. **Why do we use `PorterStemmer`?**
   - **Answer:** `PorterStemmer` is a widely used algorithm for stemming words in NLP to reduce variations of words to a common root.

4. **How do lemmatization and stemming help in text analysis?**
   - **Answer:** Both help in reducing word variations, improving the consistency of text for better analysis and prediction.

#### **Real-Life Application:**
- **Example:** In automated text classification (e.g., spam vs. non-spam), stemming and lemmatization ensure that different forms of a word are treated as the same.
- **Application:** Used in search engines, chatbots, and language modeling for predictive typing.

---

### **Assignment 24: One-Hot Encoding on Technical Texts**
---

#### **Viva Questions and Answers:**
1. **What is One-Hot Encoding?**
   - **Answer:** One-Hot Encoding represents each word in the text as a vector with all zeros except for the position corresponding to that word, which is marked as 1.

2. **Why do we use One-Hot Encoding?**
   - **Answer:** It helps convert categorical data (words) into numerical form, allowing machine learning models to understand and process the data.

3. **What does `OneHotEncoder` do in scikit-learn?**
   - **Answer:** It transforms categorical values (like words) into a format that can be used for machine learning models by encoding each unique category as a binary vector.

#### **Real-Life Application:**
- **Example:** One-Hot Encoding is used in NLP applications like text classification to convert words into vectors for easier model processing.
- **Application:** Used in machine learning models, speech recognition, and recommendation systems.

---

### **Assignment 25: Bag of Words on Movie Reviews**
---

#### **Viva Questions and Answers:**
1. **What is the Bag of Words (BoW) model?**
   - **Answer:** BoW represents text as a set of words without considering grammar or word order but keeping track of word frequencies.

2. **Why is BoW important for text analysis?**
   - **Answer:** It simplifies text representation, making it easy for machine learning algorithms to process.

3. **What does `CountVectorizer` do?**
   - **Answer:** It converts text into a matrix of word counts, which can then be used for analysis or as input for machine learning models.

#### **Real-Life Application:**
- **Example:** In sentiment analysis, the Bag of Words model is used to classify reviews as positive or negative by counting word frequencies.
- **Application:** Used in text classification, sentiment analysis, and spam detection.

---

### **Assignment 26: TF-IDF on Tourist Reviews**
---

#### **Viva Questions and Answers:**
1. **What is TF-IDF?**
   - **Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a corpus.

2. **Why is TF-IDF used in NLP?**
   - **Answer:** It helps identify important words by giving higher weights to words that are frequent in a specific document but rare in the entire corpus.

3. **What is the difference between TF-IDF and BoW?**
   - **Answer:** BoW counts word frequency, while TF-IDF adjusts word importance by considering the frequency of a word in a specific document relative to the entire corpus.

4. **How does `TfidfVectorizer` work?**
   - **Answer:** It transforms the text into a matrix where each word is weighted by its TF-IDF score, helping prioritize important words in text analysis.

#### **Real-Life Application:**
- **Example:** In a search engine, TF-IDF helps rank results by ensuring more relevant documents are prioritized based on unique keyword importance.
- **Application:** Used in information retrieval, document clustering, and search engine optimization.

---

### **General Application Question for All Assignments:**
- **How do these NLP techniques (Tokenization, Stop Words Removal, BoW, TF-IDF) apply to real-life systems?**
   - **Answer:** These techniques help in understanding and processing natural language for applications like sentiment analysis, machine translation, recommendation systems, and search engines. They allow systems to effectively understand, classify, and extract valuable insights from large volumes of text data.
"""