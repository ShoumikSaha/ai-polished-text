import pandas as pd

def extract_data_from_csv(data_path, num=-1):
    """
    Extracts the first num of data from a csv file, and returns a pandas dataframe.
    """
    df = pd.read_csv(data_path)
    if num > 0:
        df = df.head(num)
    return df

def main():
    # Load the dataset
    data_path = "data/test.csv"
    df = extract_data_from_csv(data_path, num=10)

    print(f"Loaded dataset with name {data_path}...")
    print(df)

if __name__ == "__main__":
    main()
