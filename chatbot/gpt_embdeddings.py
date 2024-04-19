
import openai  
import pandas as pd  

openai.api_key = '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'

df = pd.read_csv("GPT4_part_final.csv")
def openai_embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )

    return response['data'][0]['embedding']

def openai_embedding_create():
    df['embedding'] = df['questions'].apply(openai_embedding)
    df.to_csv("openai_embdeddings.csv",header=True)
    return df