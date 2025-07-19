# feature_engineering.py
import pandas as pd

def preprocess_data(df):
    # Filter to only include wide receivers (WR)
    df = df[df['FantPos'] == 'WR'].copy()

    # Filter data to only include the last 11 years
    current_year = df['Year'].max()
    df = df[df['Year'] >= current_year - 11]

    # Calculate new features
    df.loc[:, 'TD/G'] = df['TD'] / df['G']
    df.loc[:, 'RecYds/G'] = df['RecYds'] / df['G']
    df.loc[:, 'FantPt/G'] = df['FantPt'] / df['G']
    df.loc[:, 'Tgt/G'] = df['Tgt'] / df['G']
    df.loc[:, 'FantPtHalf/G'] = df['FantPtHalf'] / df['G']

    # Handle division by zero and fill NaNs
    df.replace([float('inf'), -float('inf')], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def add_rolling_averages(df):
    rolling_features = ['FantPtHalf/G', 'Tgt/G', 'RecYds/G']
    for feature in rolling_features:
        df[f'{feature}Last2Y'] = df.groupby('Player')[feature].transform(lambda x: x.rolling(2, min_periods=1).mean())
        df[f'{feature}Last3Y'] = df.groupby('Player')[feature].transform(lambda x: x.rolling(3, min_periods=1).mean())

    return df

def shift_target(df):
    # Shift FantPt to represent the following year's points per game
    df['NextYearFantPt/G'] = df.groupby('Player')['FantPtHalf/G'].shift(-1)

    # Remove rows where the target is NaN (i.e., no following year data)
    df = df[df['NextYearFantPt/G'].notna()]
    
    return df

def add_season_flags(df):
    df['SeasonNumber'] = df.groupby('Player').cumcount() + 1
    df['#ofY'] = df['SeasonNumber'].apply(lambda x: 'rookie' if x == 1 else ('second' if x == 2 else '3 or more'))
    df['#ofY'] = df['#ofY'].astype('category').cat.codes
    
    return df
