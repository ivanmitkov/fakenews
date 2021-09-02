from flask import Flask, request, render_template
import re
import pickle
import pandas as pd

model=pickle.load(open('automl_model.pkl','rb'))
tfidf=pickle.load(open('tfidf_vectorizer.pkl','rb'))

# flask names
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify',methods=['POST'])
def classify():
      
    if request.method == 'POST':    
        query_title=request.form['news_title']
        query_content=request.form['news_content']
        total=query_title+query_content
        total = re.sub('<[^>]*>', '', total)
        total = re.sub(r'[^\w\s]','', total)
        pred=model.predict([total])
    
    return render_template('index.html', prediction_text='The news is : {}'.format(pred[0]))
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)