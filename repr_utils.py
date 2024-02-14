import itertools
import operator
import os

import numpy as np

linewidth = 200
np.set_printoptions(linewidth=linewidth)
np.set_printoptions(precision=3, suppress=True)

from collections import Counter

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertModel, BertTokenizer

import csv
import random
import re
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

MODELS = {
    'bbc': (BertModel, BertTokenizer, 'bert-base-cased'),
    'bbu': (BertModel, BertTokenizer, 'bert-base-uncased')
}

# all paths can be either absolute or relative to this utils file
DATA_FOLDER_PATH = os.path.join('episode_dataset')
INTERMEDIATE_DATA_FOLDER_PATH = os.path.join('intermediate_data')
# this is also defined in run_train_text_classifier.sh, make sure to change both when changing.
FINETUNE_MODEL_PATH = os.path.join('..', 'models')


def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def dot_product_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b))


def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)


def pairwise_distances(x, y):
    return cdist(x, y, 'euclidean')


def most_common(L):
    c = Counter(L)
    return c.most_common(1)[0][0]


def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    f1_macro = f1_score(true_class, predicted_class, average='macro')
    f1_micro = f1_score(true_class, predicted_class, average='micro')
    if output_to_console:
        print("F1 macro: " + str(f1_macro))
        print("F1 micro: " + str(f1_micro))
    if return_tuple:
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }

# mainly for agnews
def clean_html(string: str):
    left_mark = '&lt;'
    right_mark = '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = string.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = string.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + string)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_html.clean_links.append(string[next_left_start: next_right_start + len(right_mark)])
        string = string[:next_left_start] + " " + string[next_right_start + len(right_mark):]
    return string


clean_html.clean_links = []


# mainly for 20news
def clean_email(string: str):
    return " ".join([s for s in string.split() if "@" not in s])


def clean_str(string):
    string = clean_html(string)
    string = clean_email(string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_clean_text(data_dir):
    text = load_text(data_dir)
    return [clean_str(doc) for doc in text]


def load_text(data_dir):
    with open(os.path.join(data_dir, 'dataset.txt'), mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    return text


def load_labels(data_dir):
    with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: int(x.strip()), label_file.readlines()))
    return labels


def load_classnames(data_dir):
    with open(os.path.join(data_dir, 'classes.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    return class_names


def text_statistics(text, name="default"):
    sz = len(text)

    tmp_text = [s.split(" ") for s in text]
    tmp_list = [len(doc) for doc in tmp_text]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print(f"\n### Dataset statistics for {name}: ###")
    print('# of documents is: {}'.format(sz))
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))
    print(f"#######################################")


def load(theme, title):
    data_dir = os.path.join(DATA_FOLDER_PATH, theme, title)
    text = load_text(data_dir)
    class_names = load_classnames(data_dir)
    text = [s.strip() for s in text]
    text_statistics(text, "raw_txt")

    cleaned_text = [clean_str(doc) for doc in text]
    print(f"Cleaned {len(clean_html.clean_links)} html links")
    text_statistics(cleaned_text, "cleaned_txt")

    result = {
        "class_names": class_names,
        "raw_text": text,
        "cleaned_text": cleaned_text,
    }
    return result