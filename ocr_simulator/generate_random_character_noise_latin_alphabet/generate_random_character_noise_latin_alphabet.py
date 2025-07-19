# a function for modifying an input string
from jiwer import cer
import random
import string
import pandas as pd

def apply_ocr_noise(input_string, target_cer=0.05):
    if type(input_string) != str:
        input_string = str(input_string)
    latin_alphabet_charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ    öüäéèà ÜÄÖ'
    n_changes = int(len(input_string) * target_cer)
    mutated = list(input_string)

    for _ in range(n_changes):
        error_type = random.choice(["substitution", "insertion", "deletion"])

        if error_type == "substitution":
            # Choose a random index to substitute a character
            index = random.randrange(len(mutated))
            mutated[index] = random.choice(latin_alphabet_charset)

        elif error_type == "insertion":
            # Choose a random index to insert a character
            index = random.randrange(len(mutated))
            mutated.insert(index, random.choice(latin_alphabet_charset))

        elif error_type == "deletion":
            # Choose a random index to delete a character, if the string is not empty
            if mutated:
                index = random.randrange(len(mutated))
                del mutated[index]

    return ''.join(mutated)

def apply_noise_to_dataframe(df, target_cer=0.05):
    """
    Applies OCR noise to each cell of a given DataFrame.

    :param df: Pandas DataFrame to be modified
    :param target_cer: Target Character Error Rate for the OCR noise
    :return: Modified DataFrame
    """
    modified_df = df.copy()
    for col in modified_df.columns:
        if col == 'Index':
            continue
        for row in range(len(modified_df)):
            modified_df.at[row, col] = apply_ocr_noise(modified_df.at[row, col], target_cer)
    return modified_df

sample_dataset = pd.read_csv('sample_dataset.csv')

sample_dataset_corrupted = apply_noise_to_dataframe(sample_dataset)

sample_dataset_corrupted.to_csv('sample_dataset_random_noise.csv')

