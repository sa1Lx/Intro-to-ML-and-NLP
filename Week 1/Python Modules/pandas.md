# Pandas

More about Pandas at [Pandas by GFG](https://www.geeksforgeeks.org/pandas-python-tutorial/).

Some Features of Pandas:
* ***[Series](#panda-series)***: One-dimensional labeled array.
* ***[DataFrame](#panda-dataframe)***: Two-dimensional labeled data structure (like a table).
* ***[Data Handling](#reading-files-with-pandas)***: Importing and exporting data from various formats (CSV, Excel, SQL, JSON).  
* ***[Data Manipulation](#writing-to-files-in-python)***: Filtering, sorting, grouping, and aggregation.


`pip install pandas` to install pandas  
`import pandas as pd` to import pandas

# Panda Series

A Series is a one-dimensional labeled array capable of holding data of any type (integers, strings, floats, Python objects, etc.).  

It is like a single column of a DataFrame but can also be used independently. Each element has an index (default is integer starting from 0 or user-defined).<br>

## Key Features of a Series:

    * Supports vectorized operations (like NumPy arrays).

    * Can hold any data type.

    * Can be sliced and filtered by index or condition.

    * Has many built-in methods (mean(), sum(), unique(), etc.).

    Note: In Python, list = [a, b, c]; dictionary = {'a': 1, 'b': 2, 'c': 3}
        # From a list
        s = pd.Series([10, 20, 30, 40]) # Default Index- integers from zero  

        # From a list with custom index labels
        s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

        # From a dictionary
        s = pd.Series({'x': 100, 'y': 200, 'z': 300})
        print(s)
        # Output:
        # x    100
        # y    200
        # z    300
        # dtype: int64

# Panda DataFrame

A DataFrame is a two-dimensional, size-mutable, and heterogeneous tabular data structure with labeled axes (rows and columns). Think of it like a spreadsheet or SQL table.<br>

## Key Features of a DataFrame:

    * Columns can hold different data types.

    * Rows and columns are both labeled.

    * Supports missing data.

    * Easy to slice, filter, and aggregate data.

    * Integrates well with other data libraries and file formats.

Note: 

When you pass lists to pd.DataFrame(), Pandas automatically converts each list into a Series behind the scenes.

        # From a dictionary of lists
        data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'Paris', 'London']
        }
        df = pd.DataFrame(data)

        # From a list of dictionaries
        data2 = [
            {'Name': 'Alice', 'Age': 25},
            {'Name': 'Bob', 'Age': 30}
        ]
        df2 = pd.DataFrame(data2)

# Reading files with Pandas

- CSV File
```
df = pd.read_csv('file.csv')
```

- Excel File
```
df = pd.read_excel('file.xlsx')
```

 - JSON File
 ``` 
 df = pd.read_json('file.json')
 ```

 - From URL
 ``` 
 df = pd.read_csv('https://example.com/data.csv')
 ```

 - From SQL Database
 ```
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM tablename', conn)
 ```

 - From Clipboard (e.g., copied from Excel)
```
df = pd.read_clipboard()
```
# Writing To Files In Python 
 - Write to CSV:
```
df.to_csv('file.csv', index=False)
```

- Write to Excel:
```
df.to_excel('file.xlsx', index=False)
```

- Write to JSON:
```
df.to_json('file.json', orient='records', lines=True)
```

- Write to SQL:
```
df.to_sql('tablename', conn, if_exists='replace', index=False)
```

- Write to Clipboard:
```
df.to_clipboard(index=False)
```
# Common Pandas Functions Explained with Examples and Output

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, None, 40, 35],
    'City': ['NY', 'LA', 'NY', 'Chicago', None],
    'Score': [85, 90, 78, 92, 88]
}
df = pd.DataFrame(data)

df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Frank'],
    'Salary': [70000, 80000, 65000]
})
```

---

- **`head()`**  
  Show the first 5 rows (default):
  ```python
  print(df.head())
  ```
  **Output:**
  ```
       Name   Age     City  Score
  0    Alice  25.0       NY     85
  1      Bob  30.0       LA     90
  2  Charlie   NaN       NY     78
  3    David  40.0  Chicago     92
  4      Eva  35.0     None     88
  ```

---

- **`tail()`**  
  Show the last 3 rows:
  ```python
  print(df.tail(3))
  ```
  **Output:**
  ```
       Name   Age     City  Score
  2  Charlie   NaN       NY     78
  3    David  40.0  Chicago     92
  4      Eva  35.0     None     88
  ```

---

- **`isnull()`**  
  Check missing values:
  ```python
  print(df.isnull())
  ```
  **Output:**
  ```
      Name    Age   City  Score
  0  False  False  False  False
  1  False  False  False  False
  2  False   True  False  False
  3  False  False  False  False
  4  False  False   True  False
  ```

---

- **`fillna()`**  
  Replace missing values:
  ```python
  df_filled = df.fillna({'Age': 0, 'City': 'Unknown'})
  print(df_filled)
  ```
  **Output:**
  ```
       Name   Age     City  Score
  0    Alice  25.0       NY     85
  1      Bob  30.0       LA     90
  2  Charlie   0.0       NY     78
  3    David  40.0  Chicago     92
  4      Eva  35.0  Unknown     88
  ```

---

- **`rename()`**  
  Rename columns:
  ```python
  df_renamed = df.rename(columns={'Score': 'ExamScore'})
  print(df_renamed.head())
  ```
  **Output:**
  ```
       Name   Age     City  ExamScore
  0    Alice  25.0       NY         85
  1      Bob  30.0       LA         90
  2  Charlie   NaN       NY         78
  3    David  40.0  Chicago         92
  4      Eva  35.0     None         88
  ```

---

- **`groupby()`**  
  Group by 'City' and calculate average 'Score':
  ```python
  grouped = df.groupby('City')['Score'].mean()
  print(grouped)
  ```
  **Output:**
  ```
  City
  Chicago    92.0
  LA         90.0
  NY         81.5
  Name: Score, dtype: float64
  ```

---

- **`merge()`**  
  Merge `df` and `df2` on 'Name':
  ```python
  merged_df = pd.merge(df, df2, on='Name', how='inner')
  print(merged_df)
  ```
  **Output:**
  ```
     Name   Age City  Score  Salary
  0  Alice  25.0   NY     85   70000
  1    Bob  30.0   LA     90   80000
  ```


