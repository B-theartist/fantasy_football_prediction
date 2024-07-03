import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# Define the years for which we want to fetch data
years = range(2003, 2024)

# Initialize an empty DataFrame to store the data
all_data = pd.DataFrame()

# Loop through each year and fetch the data
for year in years:
    URL = f'https://www.pro-football-reference.com/years/{year}/fantasy.htm'
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract table data
    table = soup.find('table', {'id': 'fantasy'})
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Add a column for the year
    df['Year'] = year
    
    # Append the data to the all_data DataFrame
    all_data = pd.concat([all_data, df], ignore_index=True)

# Flatten MultiIndex columns
all_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in all_data.columns]

# Save the scraped data to an Excel file
all_data.to_excel('fantasy_football_data.xlsx', sheet_name='Scraped Data', index=False)

print("Data has been written to fantasy_football_data.xlsx")
