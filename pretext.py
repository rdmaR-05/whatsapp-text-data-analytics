import re
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
chat_files = [
    "marshmallows.txt",
    "Clima_Jan_June.txt",
    "Cocomelons.txt",
    "TMI_Central.txt"
]
output_dir = "preprocessed_chats"
os.makedirs(output_dir, exist_ok=True)
# function to preprocess chat text
def preprocess_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # removal of metadata, omitted messages, and timestamps
    cleaned_lines = []
    for line in lines:
        if ('omitted' in line.lower() or 'message edited' in line.lower()):
            continue

        # sender and timestamps remove karo
        line = re.sub(r'^\[\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{1,2}(:\d{1,2})? [APap][Mm]\] .*?: ', '', line, flags=re.IGNORECASE)

        # am and pm removal
        line = re.sub(r'\b(pm|am)\b', '', line, flags=re.IGNORECASE).strip()

        if line:
            cleaned_lines.append(line)
    cleaned_text = " ".join(cleaned_lines).lower()
    cleaned_text = re.sub(r'[^a-z\s]', '', cleaned_text)
    # tokenization
    tokens = word_tokenize(cleaned_text)
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # convert back to text
    return " ".join(lemmatized_tokens)

# har chat ko seprately process kar
for chat_file in chat_files:
    processed_text = preprocess_chat(chat_file)
    
    output_file = os.path.join(output_dir, f"preprocessed_{chat_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(processed_text)

    print(f"Preprocessed chat saved: {output_file}")

print("\n All chats processed and saved")
