# from flask import Flask, jsonify, request
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# import torch

# from model import Comment

# from database import (
#     fetch_comments,
#     fetch_all_comments,
#     create_comment,
# )


# app = Flask(__name__)

# model_load_path = "./model"
# loaded_model = BertForSequenceClassification.from_pretrained(model_load_path)
# loaded_tokenizer = BertTokenizer.from_pretrained(model_load_path)

# id2label = {
#     "0": "NOT",
#     "1": "OFF"
# }

# @app.route("/",methods=['GET'])
# async def read_root():
#     return {"Hello": "World"}


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         text_to_classify = data['text']
#         inputs = loaded_tokenizer(text_to_classify, return_tensors='pt', padding=True, truncation=True)
#         outputs = loaded_model(**inputs)
#         logits = outputs.logits
#         probabilities = torch.softmax(logits, dim=1)
#         predicted_class_id = torch.argmax(probabilities, dim=1).item()
#         predicted_label = id2label.get(str(predicted_class_id), "Unknown")


#         new_doc = {
#             "comment":text_to_classify,
#             "label":predicted_label
#         }
#         response = await create_comment(new_doc)
#         return response
#     except Exception as e:
#         return jsonify({"error": str(e)})


# if __name__ == '__main__':
#     app.run(debug=True)



# --------


from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import Comment
from database import fetch_comments, fetch_all_comments, create_comment
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

app = FastAPI()

origins = [
    "https://laughing-trout-7x7rj4575g93rprw-3000.app.github.dev",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_load_path = "./model"
loaded_model = BertForSequenceClassification.from_pretrained(model_load_path)
loaded_tokenizer = BertTokenizer.from_pretrained(model_load_path)

id2label = {
    "0": "NOT",
    "1": "OFF"
}

def pred(comment):
    inputs = loaded_tokenizer(comment, return_tensors='pt', padding=True, truncation=True)
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_label = id2label.get(str(predicted_class_id), "Unknown")
    return predicted_label

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/api/comment/")
async def post_comment(comment: str = Form(...)):
    try:
        print(comment)
        label = pred(comment)

        new_doc = {
            "comment": comment,
            "label": label
        }
        response = await create_comment(new_doc)
        return response
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/comment/")
async def get_comment():
    response = await fetch_all_comments()
    return response

@app.get("/api/comment/{label}")
async def get_comment_by_label(label):
    print(label)
    response = await fetch_comments(label)
    if response:
        return response
    raise HTTPException(404, f"There are no comments with label {label}")
