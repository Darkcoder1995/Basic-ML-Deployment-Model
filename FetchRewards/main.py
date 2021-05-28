from flask import Flask, render_template
from flask import request
import numpy as np
import math
import re
import pandas as pd
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('home.html')
    else:
        text1=request.form['text1']
        text2 = request.form['text2']
        l1 = main_function()
        final_score = l1.run(text1, text2)
        return str(final_score)


class tokenizer:
    def tokenize(self, sentences):
        for i in sentences:
            tokens = sentences.split()
            return tokens

    def stop_words_removal(self, sentences):
        new_text = ""
        tokens = [i.lower() for i in sentences]
        list_stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
                           'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do',
                           'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am',
                           'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
                           'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself',
                           'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
                           'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
                           'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                           'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
                           'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
                           'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
                           'here', 'than']
        stop_words_removed_string = [word for word in tokens if word not in list_stop_words]
        for i in stop_words_removed_string:
            new_text = new_text + " " + i
        return new_text

    def regex1(self, sentences):
        removeSpecialChars = sentences.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
        return removeSpecialChars

    def remove_articles(self, sentences):
        removeArticles = re.sub('\s+(a|an|and|the)(\s+)', '\2', sentences)
        return removeArticles

    def excess_white_space_removal(self, sentences):
        return " ".join(sentences.split())

    def apostrophes(self, sentences):
        return sentences.replace("'", "")


class tf_idf1:
    def count_terms_dictionary(self, sentences):
        DF = {}
        words = sentences.split()
        for i in range(len(words)):
            if words[i] not in DF:
                DF[words[i]] = 1
            else:
                DF[words[i]] += 1
        return DF

    def computeTF(self, sentences):
        tfDict = {}
        s1 = tf_idf1()
        counter = s1.count_terms_dictionary(sentences)
        words_count = len(sentences.split())
        for w, c in counter.items():
            tfDict[w] = c / float(words_count)
        return tfDict

    def computeIDF(self, sentences):
        s1 = tf_idf1()
        idfDict = {}
        N = len(sentences.split())
        idfDict = s1.computeTF(sentences)
        for w, c in idfDict.items():
            idfDict[w] = math.log10(N / (float(c) + 1))

        return idfDict

    def computeTFIDF(self, sentences):
        s1 = tf_idf1()
        tfidf = {}
        counter = s1.computeTF(sentences)
        idfs = s1.computeIDF(sentences)
        for w, c in counter.items():
            tfidf[w] = c * idfs[w]
        return tfidf

class calc:
    def calculate(self,sent):
        s=tokenizer()
        s1=tf_idf1()
        sent1=s.tokenize(sent)
        sent2=s.stop_words_removal(sent1)
        sent3=s.regex1(sent2)
        sent4=s.remove_articles(sent3)
        sent5=s.excess_white_space_removal(sent4)
        sent6=s.apostrophes(sent5)
        tfidfs=s1.computeTFIDF(sent6)
        return tfidfs

class scoring:
    def score(self,a,b):
        c=pd.DataFrame([a,b])
        k=c.fillna(0)
        k1=k.iloc[0]
        k2=k.iloc[1]
        dot_prod=np.dot(k1,k2)
        squares1=0
        squares2=0
        squares1=[i*i for i in k1]
        sum_squares1=sum(squares1)
        mag_squares1=np.sqrt(sum_squares1)
        squares2=[i*i for i in k2]
        sum_squares2=sum(squares2)
        mag_squares2=np.sqrt(sum_squares2)
        cosine=dot_prod/(mag_squares1*mag_squares2)
        return cosine

class main_function:
    def run(self,text1,text2):
        t1=calc()
        t2=scoring()
        tfidf_1=t1.calculate(text1)
        tfidf_2=t1.calculate(text2)
        scores=t2.score(tfidf_1,tfidf_2)
        return scores

if __name__=='__main__':
    app.run()
