# WhatsApp Text Data Analytics

This project analyzes exported WhatsApp chat data using Python. It performs text preprocessing, sentiment analysis, and feature extraction techniques like TF-IDF and Bag of Words to uncover insights from chat conversations.

## 📁 Project Structure

- `pretext.py`: Parses and cleans raw WhatsApp `.txt` chat exports.
- `chatsentiment.py`: Performs sentiment analysis on messages.
- `TF_IDF&BoW.py`: Extracts features using TF-IDF and Bag of Words models.
- `figures/`: Contains visualizations generated from the analysis.

## 🔧 Features

- **Data Preprocessing**: Cleans and structures raw chat data for analysis.
- **Sentiment Analysis**: Evaluates the emotional tone of messages.
- **Feature Extraction**: Applies TF-IDF and Bag of Words to identify key terms and patterns.
- **Visualizations**: Generates plots to represent findings (located in the `figures` directory).


## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rdmaR-05/whatsapp-text-data-analytics.git
   cd whatsapp-text-data-analytics
Install dependencies:


pip install -r requirements.txt
Export your WhatsApp chat:

Open the chat in WhatsApp.

Tap on the menu > More > Export Chat > Without Media.

Save the .txt file to the project directory.

Run the preprocessing script:

python pretext.py
Perform sentiment analysis:


python chatsentiment.py
Extract features:

python TF_IDF&BoW.py
🛠 Technologies Used
Python
 
pandas

scikit-learn

matplotlib

seaborn

nltk
