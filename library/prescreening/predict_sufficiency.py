"""
Script to run stage 2 of the pre-screening process: run a SetFit classifier on abstracts to determine if articles are about sufficiency or not.
Predicts and saves probabilities for 5 classes : ["other","planetary_boundaries","well_being","resources","justice"].
A threshold is then applied to filter articles. That step is done after running this script.
Adapted from Théo Alves' Collab notebook.
"""

import os
import sys

import boto3
from dotenv import load_dotenv
import pandas as pd
from setfit import SetFitModel


load_dotenv()


def get_s3_client():
    s3 = boto3.client(
            service_name='s3',
            endpoint_url='https://s3.fr-par.scw.cloud',
            aws_access_key_id=os.environ['S3_ACCESS_KEY'],
            aws_secret_access_key=os.environ['S3_SECRET_KEY'],
        )
    return s3


def get_model(device: int):
    # download Theo's pretrained model from the HF hub (only works in English)
    model = SetFitModel.from_pretrained("TheoLvs/wsl-prescreening-multi-v0.0")
    model.to(f'cuda:{device}')
    return model


def get_chunk_url(i: int):
    return f"https://sufficiency-library.s3.fr-par.scw.cloud/stage-1/chunk_{i}.parquet"


def prepare_chunk(i: int):
    df = pd.read_parquet(get_chunk_url(i), columns=['id', 'title', 'abstract', 'language'])
    df = df[df['abstract'].notna()]
    # Theo's SetFit model only works in English, so we keep only English articles
    # TODO 1: use langdetect on the abstract to filter more accurately
    # TODO 2: train with SetFit a multilingual model like bge-m3 to process more languages
    df = df[df['language'] == 'en']
    return df


def get_preds(df, model, batch_size=100):
    preds = model.predict_proba(df['abstract'].to_list(), batch_size=batch_size, show_progress_bar=True)
    labelnames = ["other","planetary_boundaries","well_being","resources","justice"]
    cols = ["proba_"+x for x in labelnames]
    preds_df = pd.DataFrame(preds[:,:,1].numpy(), columns=cols)
    return pd.concat((df.reset_index(drop=True), preds_df), axis=1)


def add_screen_class(df):
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


def save_to_s3(df, i, s3):
    df.to_parquet(f"/tmp/chunk_{i}.parquet")
    s3.upload_file(
        Filename=f"/tmp/chunk_{i}.parquet",
        Bucket="sufficiency-library",
        Key=f"stage-2/preds/chunk_{i}.parquet"
    )
    print(f'chunk_{i}.parquet saved to s3')


def run_inference(start_idx: int, end_idx: int, chunk_size: int, batch_size: int,  device: int):
    s3 = get_s3_client()
    model = get_model(device)

    for i in range(start_idx, end_idx + 1):
        print(f'processing chunk_{i}')
        df = prepare_chunk(i)
        if chunk_size >= len(df):
            preds = get_preds(df, model, batch_size)
            preds = add_screen_class(preds)
            save_to_s3(preds, i, s3)
        else:
            start = 0
            j = 0
            while start < len(df):
                end = start + chunk_size
                batch = df.iloc[start:end]
                preds = get_preds(batch, model, batch_size)
                save_to_s3(preds, f'{i}_{j}', s3)
                start = end
                j += 1


def main():
    args = sys.argv
    assert len(args) == 6, "Please provide 5 command line arguments in this order: start file index, end file index, chunk size, batch_size, device (0 or 1)"
    start = int(args[1])
    end = int(args[2])
    chunk_size = int(args[3])
    batch_size = int(args[4])
    device = int(args[5])
    run_inference(start, end, chunk_size, batch_size, device)


if __name__ == '__main__':
    main()
