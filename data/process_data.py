import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath).drop_duplicates(subset='id')
    categories = pd.read_csv(categories_filepath).drop_duplicates(subset='id')
    df = messages.merge(categories,how='inner', on='id')
    categories = categories.categories.str.split(';',expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda column: column[:-2] )
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda column: column[-1] ).astype(int)
    df.drop(columns='categories')
    df= pd.merge(df, categories, left_index=True, right_index=True)
    return df

def clean_data(df):
    if df[df.duplicated()].empty:
        pass
    else:
        df.drop_duplicates( keep='first', inplace=True, ignore_index=False)
    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()