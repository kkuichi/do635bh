import google.generativeai as genai
import pandas as pd
import pathlib
import textwrap


genai.configure(api_key='AIzaSyCkDIG7zfXvJf_AJ1D3xl0n1uGjOqPrYvQ')

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def gemini_embdeddings_create():
  model = 'models/embedding-001'
  df = pd.read_csv("GPT4_part_final.csv")
  print(df)

  embeddings_list = []
  for index, row in df.iterrows():
      #print(row['questions'])
      embdeddings=genai.embed_content(model=model,
                              content=row['questions'],
                              task_type="retrieval_document"
                              )["embedding"]
      embeddings_list.append(embdeddings)

  if len(embeddings_list) == len(df):
      # Add the embeddings to the DataFrame
      df['Embeddings'] = embeddings_list
  else:
      print("Lengths do not match. Something went wrong with embedding generation.")
  df.to_csv("gemini_q&a_embdeddings.csv",header=True)
  return df