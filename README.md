# Sentiment Analysis Project

This project performs **sentiment analysis** on movie reviews using machine learning techniques. It demonstrates text preprocessing, feature extraction, model training, evaluation, and visualization.

---

## Dataset

We use the **IMDB 50K Movie Reviews** dataset:

- 50,000 movie reviews labeled as **positive** or **negative**
- Original source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)

> **Note:** The dataset is **not included** in this repository due to size limitations. Download the dataset and place it in the same folder as the notebook.

---

## Project Steps

1. **Data Cleaning & Preprocessing**  
   - Convert text to lowercase  
   - Remove punctuation, numbers, URLs, HTML tags  
   - Remove stopwords  
   - Apply stemming using NLTK's SnowballStemmer

2. **Exploratory Data Analysis (EDA)**  
   - Check for missing values and duplicates  
   - Plot sentiment distribution  
   - Generate **word clouds** for positive and negative reviews  
   - Analyze **top words** used in reviews  

3. **Feature Extraction**  
   - Convert text to numerical features using **TF-IDF Vectorizer**  
   - Consider unigrams and bigrams  

4. **Model Training**  
   - Train a **Logistic Regression** classifier on the training set  
   - Evaluate on the test set using **accuracy, classification report, and confusion matrix**

5. **Custom Prediction**  
   - Function to predict sentiment of new reviews  
   - Example: `"I absolutely loved this movie!"` → Positive  

---

## Visualizations

- **Sentiment Distribution:** Count of positive vs negative reviews  
- **Word Clouds:** Most frequent words in positive and negative reviews  

---

## How to Run

1. Clone the repository and open the notebook.  
2. Install required Python libraries: `pandas, nltk, scikit-learn, matplotlib, seaborn, wordcloud`  
3. Place the downloaded IMDB dataset (`IMDB Dataset.csv`) in the same folder as the notebook.  
4. Run the Jupyter notebook to see preprocessing, model training, evaluation, and visualizations.  

---

## Notes

- You can replace the dataset with other review datasets (e.g., Amazon, Yelp) with minor changes.  
- Optional enhancements: top bigram analysis, emotion detection, lexicon-based sentiment comparison (TextBlob).  

---



