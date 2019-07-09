# -*- coding: utf-8 -*-
from flask import Flask, request, redirect
import jpype
with open("/Users/zac/Downloads/stopwords.txt","r") as f: stopwords=[i.strip("\n") for i in f.readlines()]
with open("/Users/zac/Downloads/word2idfMap.txt","r") as f: word2idfMap=dict([tuple(i.strip("\n").split("\t")[:2]) for i in f.readlines()])

# ==============
print("JVM already on") if jpype.isJVMStarted() else jpype.startJVM(jpype.getDefaultJVMPath(),"-ea","-Djava.cass.path=/Users/zac/Downloads/nlp_combined.jar")
jpype.java.lang.System.out.println("java print")
obj=jpype.JPackage("com").apus.algo.nlp.RAKE

JDClass = jpype.JClass("com.apus.algo.nlp.RAKE")
jd = JDClass()

jpype.shutdownJVM()
# ==============
app = Flask(__name__)

@app.route("/")
def to_index():
    return redirect("/index")

@app.route('/index/')
def index():
    return "index-page"

# 如果你想采用post请求，那么要写明
@app.route('/keyword/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return "get"
    else:
        text = request.form.get('text')
        result = "use jpy to parse text: \n%s"%text
        return result


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8081)
