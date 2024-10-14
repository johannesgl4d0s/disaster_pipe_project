import sys
import pandas as pd
from sqlalchemy import create_engine, text
from nltk.tokenize import word_tokenize
import nltk
nltk.download(['punkt', 'wordnet','punkt_tab' ])
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.metrics import classification_report
def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    with engine.connect() as conn:
        # Use the connection to execute the query and read into a DataFrame
        df = pd.read_sql(text("SELECT * FROM Message"), conn)

    X = df['message'].values
    Y = df.iloc[:, 6:].values
    category_names = df.columns[6:]
    
    return X,Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
   


def build_model():
    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return model


#         pipeline = Pipeline([
#             ('vect', CountVectorizer(tokenizer=tokenize)),
#             ('tfidf', TfidfTransformer()),
#             ('clf', MultiOutputClassifier(RandomForestClassifier()))
#         ])
#         return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names= category_names))


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()