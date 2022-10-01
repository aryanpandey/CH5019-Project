from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

## Creating a parser
parser = argparse.ArgumentParser(description = "File Parser")

parser.add_argument("correct_answer",
                    metavar='ca',
                    type = str,
                    help = 'the path to a txt file containing the correct answer')

parser.add_argument("student_answer",
                    metavar='sa',
                    type = str,
                    help = "the path to a txt file containing the student's answer")

args = parser.parse_args()           

ca_path = args.correct_answer
sa_path = args.student_answer

with open(ca_path) as f:
    ca = f.readlines()

with open(sa_path) as f:
    sa = f.readlines()

def clean_text(answer):
    text = ''
    for line in answer:
        text = text + line.strip()
        text = text + ' '
    
    return text

def create_embeddings(answer):
    sents = answer.rstrip().split('.')
    sents = model.encode(sents)
    sents = np.mean(sents, axis = 0)

    return sents


ca = clean_text(ca)
sa = clean_text(sa)

ca = create_embeddings(ca)
sa = create_embeddings(sa)

print("The similarity between the two answers is: {:.2f} %".format(
       cosine_similarity(ca.reshape(1,-1), sa.reshape(1,-1))[0][0]*100))