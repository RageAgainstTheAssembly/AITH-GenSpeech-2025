import pandas as pd
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm


def normalize_dataframe(df: pd.DataFrame, normalizer: Normalizer) -> pd.DataFrame:
    tqdm.pandas()
    df['normalized_transcription'] = df['transcription'].progress_apply(
        lambda x: normalizer.normalize(str(x), punct_post_process=True)
    )
    return df


def main():
    normalizer = Normalizer(input_case='cased', lang='ru', deterministic=False)

    train_df = pd.read_csv('train.csv')
    dev_df = pd.read_csv('dev.csv')

    train_df = normalize_dataframe(train_df, normalizer)
    dev_df = normalize_dataframe(dev_df, normalizer)

    train_df.to_csv('train_normalized.csv', index=False)
    dev_df.to_csv('dev_normalized.csv', index=False)


if __name__ == '__main__':
    main()
