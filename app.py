from flask import Flask, render_template, request, jsonify
import time
import pandas as pd
import textwrap
import numpy as np
import openai
import google.generativeai as genai
from chatbot.gemini_embdeddings import gemini_embdeddings_create
from chatbot.gpt_embdeddings import openai_embedding_create
import gspread
import threading

from oauth2client.service_account import ServiceAccountCredentials
#print openai version
#print(genai.__version__)


#finalny rozsah otazaok 

#https://docs.google.com/spreadsheets/d/1SjeFuHlpfqQm3TXDtSCmntFtl2lAby6UAlHecg_s09I/edit#gid=0

spreadsheet_url = "https://docs.google.com/spreadsheets/d/1SjeFuHlpfqQm3TXDtSCmntFtl2lAby6UAlHecg_s09I/edit?usp=sharing"
worksheet_name = 'test'
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(credentials)


app = Flask(__name__)
app.config['WEBSOCKET_ENABLED'] = False
# Set your OpenAI API key
openai.api_key = '###########################################################'
genai.configure(api_key='###########################################################')

print("--------------------------------CREATING GEMINI EMBEDDEDINGS--------------------------------")
#gemini_embd=gemini_embdeddings_create()
gemini_embd=pd.read_csv('gemini_embdeddings.csv')
gemini_embd['Embeddings'] = gemini_embd.Embeddings.apply(eval).apply(np.array)
print("--------------------------------CREATING OPENAI EMBEDDEDINGS--------------------------------")
#openai_embd=openai_embedding_create()
openai_embd=pd.read_csv('openai_embdeddings.csv')
openai_embd['embedding'] = openai_embd.embedding.apply(eval).apply(np.array)
print("--------------------------------CREATING EMBEDDEDINGS FINISHED--------------------------------")
def find_best_passage(query, dataframe):
        model='models/embedding-001'
        query_embedding = genai.embed_content(model=model,
                                                content=query,
                                                task_type="retrieval_query")
        dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
        idx = np.argmax(dot_products)
        #print(dataframe.iloc[idx]['Text'])
        return dataframe.iloc[idx]['Text'] # Return text from index with max value

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  maxOutputTokens=150
  prompt = textwrap.dedent("""si pomocnik ktory pomaha odpovedat studentom ohladom studia.Informacie ziskaj z passage ktore je pridane,odpoved vygeneruj maximalne v dlzke 256 tokenov 
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt


def gemini_resp(query):
    print("GEMINI START")
    #mesure time of execution 
    start_time = time.time()
    generation_config = {
    "temperature": 0.9,
    "top_p": 1,
     "top_k": 25
    }
    passage =find_best_passage(query, gemini_embd)
    #print(passage)
    prompt = make_prompt(query, passage)
    model = genai.GenerativeModel('models/gemini-pro')
    answer = model.generate_content(prompt,generation_config=generation_config)
    end_time = time.time()
    print("Execution time GEMINI:", end_time - start_time, "seconds")
    return answer.text


def calculate_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1, dtype=np.float32)
    embedding2 = np.array(embedding2, dtype=np.float32)

    # Calculate cosine similarity between two embeddings
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # Check for zero division to avoid runtime warnings
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

def custombased(text):
    print("GPT START")
    start_time = time.time()
    response = openai.Embedding.create(
            model="text-embedding-3-large",  
            input=text
        )
    question_embedding=response['data'][0]['embedding']


    best_match = None
    best_similarity = 0.0

    for index, row in openai_embd.iterrows():
            entry_embedding = row['embedding']   
        
            similarity = calculate_similarity(question_embedding, entry_embedding)


            if similarity > best_similarity:
                best_similarity = similarity
                best_match=row['answers']
                

    conversation = [
            {"role": "system", "content": "Si skolsky asistent na otazku skus odpovedat vedomostami ktore ziskat v contexte, odpoved vygeneruj maximalne v dlzke 256 tokenov"},
            {"role": "user", "content": f"Question: {text}"},
            {"role": "assistant", "content": f"Context: {best_match}"}
        ]


    response = openai.ChatCompletion.create(
    model="gpt-4-turbo-preview",
    temperature=0.5,
    top_p=0.5,
    max_tokens=256
    )


    answer = response['choices'][0]['message']['content'].strip()
    end_time = time.time()
    print("Execution time GPT:", end_time - start_time, "seconds")
    return answer

attempts_count = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_gpt_response', methods=['POST'])
def handle_user_input():
    user_input = request.form['user_input']

    if user_input in attempts_count:
        if attempts_count[user_input] >= 4:
            return jsonify(error="You have exceeded the maximum number of attempts."), 400
        
    def run_custombased():
        global custombased_result
        custombased_result = custombased(user_input)

    def run_gemini_resp():
        global gemini_resp_result
        gemini_resp_result = gemini_resp(user_input)

    # Create threads
    thread1 = threading.Thread(target=run_custombased)
    thread2 = threading.Thread(target=run_gemini_resp)

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()
    
    attempts_count[user_input] = attempts_count.get(user_input, 0) + 1

    return jsonify({
        'chat_response_a': custombased_result,
        'chat_response_b': gemini_resp_result
    })

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_data = request.form.to_dict()

    # Open the specified worksheet
    worksheet = client.open_by_url(spreadsheet_url).worksheet(worksheet_name)
    print(worksheet)
    # Append the form data to the worksheet
    worksheet.append_row([feedback_data['quality_a_1'],feedback_data['quality_a_2'],feedback_data['usability_a_3'],feedback_data['quality_a_4'],feedback_data['usability_a_5'],feedback_data['quality_a_6'],\
                          feedback_data['quality_b_1'],feedback_data['quality_b_2'],feedback_data['usability_b_3'],feedback_data['quality_b_4'],feedback_data['usability_b_5'],feedback_data['quality_b_6'],\
                        feedback_data['choice'],feedback_data['choice2'],feedback_data['choice3'],feedback_data['choice4'],feedback_data['choice5'],feedback_data['choice6'], feedback_data['feedback']])

    return 'Feedback submitted successfully!'

if __name__ == '__main__':
    app.run(port=5000)