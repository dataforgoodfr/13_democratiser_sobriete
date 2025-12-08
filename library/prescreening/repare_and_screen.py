""""
SHOULDN'T BE NECESSARY ANYMORE: what this script does is now handled by predict_sufficiency.py
Part of stage 2 of pre-screening.
Takes raw model predictions and assigns them a class and a filter status.
Also repares a mistake in the process.
"""


import os
import boto3
from dotenv import load_dotenv
import pandas as pd


load_dotenv()


s3 = boto3.client(
        service_name='s3',
        endpoint_url='https://s3.fr-par.scw.cloud',
        aws_access_key_id=os.environ['S3_ACCESS_KEY'],
        aws_secret_access_key=os.environ['S3_SECRET_KEY'],
    )


def get_chunk(i):
    return pd.read_parquet(f"https://sufficiency-library.s3.fr-par.scw.cloud/stage-2/preds/chunk_{i}.parquet")


def repare(df):
    """Function to recover from a mistake where I initially forgot to reset the index before concatenating the predictions to the data."""
    part1 = df[['id', 'title', 'abstract', 'language']]
    part2 = df[['proba_other', 'proba_planetary_boundaries', 'proba_well_being', 'proba_resources','proba_justice']]
    return pd.concat((part1.dropna().reset_index(drop=True), part2.sort_index().dropna()), axis=1)


def screen(df):
    preds_df = df[['proba_other', 'proba_planetary_boundaries', 'proba_well_being', 'proba_resources', 'proba_justice']]
    prescreening_high = (preds_df.idxmax(axis = 1) != "proba_other")
    pred_class = preds_df.idxmax(axis = 1).map(lambda x : x.replace("proba_",""))

    prescreening_medium = prescreening_high.copy()
    prescreening_medium.loc[preds_df.max(axis = 1) < 0.5] = False

    prescreening_low = prescreening_high.copy()
    prescreening_low.loc[preds_df.max(axis = 1) < 0.8] = False

    df["prescreening_high"] = prescreening_high
    df["prescreening_medium"] = prescreening_medium
    df["prescreening_low"] = prescreening_low
    df["pred_class"] = pred_class
    
    return df


def upload(df, i):
    os.makedirs('tmp', exist_ok=True)
    path = f"tmp/chunk_{i}.parquet"
    df.to_parquet(path)
    s3.upload_file(
        Filename=path,
        Bucket="sufficiency-library",
        Key=f"stage-2/preds/clean_chunk_{i}.parquet"
    )
    print(f'chunk_{i}.parquet saved to s3')
    os.remove(path)


def main():
    for i in range(13, 15):
        print('Get', i)
        df = get_chunk(i)
        print(len(df))
        df = repare(df) # you might not need this line
        print(len(df))
        df = screen(df)
        print(df['prescreening_low'].sum())
        upload(df, i)
        del df


if __name__ == "__main__":
    main()
