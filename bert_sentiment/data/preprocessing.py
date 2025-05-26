import numpy as np 
import pandas as pd
import random
import nltk
import nlpaug.augmenter.word as naw

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet') 



# Data Augmentation to handle imbalance.

# Technique 1: Synonym Replacement
def synonym_replacement(text, replace_ratio=0.2):
    aug = naw.SynonymAug(aug_src='wordnet', aug_max=1)
    return aug.augment(text)

# Technique 2: Random Insertion
def random_insertion(text):
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    return aug.augment(text)

# Technique 3: Random Swap
def random_swap(text):
    words = text.split()
    if len(words) > 1:
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

# Technique 4: Random Deletion
def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 1:
        return text
    remaining = [word for word in words if random.random() > p]
    return ' '.join(remaining) if remaining else words[0]


# Apply to DataFrame

def load_and_augment(path):
    # Load data
    df = pd.read_csv(path)
    # Drop nan values and unwanted columns
    df.drop(['emotion_in_tweet_is_directed_at'],axis=1,inplace=True)
    df.columns = ['text','sentiment']
    df.dropna(inplace=True)
    df['sentiment'] = df['sentiment'].map({'Negative emotion': 2, 'Positive emotion':1,'No emotion toward brand or product':0,"I can't tell":0}) 


    augmented_rows = []
    df_to_aug = df[df.sentiment!=0] # dataset to augment (minority)
    majority_df = df[df.sentiment==0]

    for _, row in df_to_aug.iterrows():
        original = row['text']
        sent=row['sentiment']
        augmented_rows.append({'text': original,'sentiment':sent})  # keep original
        augmented_rows.append({'text': synonym_replacement(original),'sentiment':sent})

        if sent!=1: # more augmentation on the minority class
            augmented_rows.append({'text': random_swap(original),'sentiment':sent})
            augmented_rows.append({'text': random_insertion(original),'sentiment':sent})
            augmented_rows.append({'text': random_deletion(original),'sentiment':sent})


    df = pd.DataFrame(augmented_rows)
    df = pd.concat([df,majority_df])
    df = df.sample(frac = 1).reset_index(drop=True) # shuffle
    # Some augmented texts are returned as list
    df['text'] = df['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    return df