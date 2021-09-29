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
        query_content=request.form['news_content']
        query_content = re.sub('<[^>]*>', '', query_content)
        query_content = re.sub(r'[^\w\s]','', query_content)
        data_for_prediction = tfidf.transform(pd.Series(query_content))
        pred = model.predict(data_for_prediction.toarray())
        pred = ['FAKE' if x == 1 else 'REAL' for x in pred]
    
    return render_template('index.html', prediction_text='The news is : {}'.format(pred[0]))
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)