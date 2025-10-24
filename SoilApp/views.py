from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from django.core.files.storage import FileSystemStorage

global uname, elm_model, scaler
global X_train, X_test, y_train, y_test, X, Y, dataset
accuracy, precision, recall, fscore = [], [], [], []
labels = ['Less Fertile', 'Fertile', 'Highly Fertile']

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore, labels
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()     
    return img_b64

def ProcessDataset(request):
    if request.method == 'GET':
        global X, Y, dataset, scaler, X_train, X_test, y_train, y_test
        dataset = pd.read_csv("Dataset/soil_fertitlity_dataset.csv")
        dataset.fillna(0, inplace = True)
        columns = dataset.columns
        dataset = dataset.values
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        scaler = StandardScaler((0,1))
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="" color="black">'+columns[i]+'</th>'
        output += "</tr>"
        for i in range(len(dataset)):
            output += "<tr>"
            for j in range(len(dataset[i])):
                output+='<td><font size="" color="black">'+str(dataset[i, j])+'</td>'
            output += "</tr>"
        output += "</table><br/><br/><br/>"    
        context= {'data': output}
        return render(request, 'ViewResult.html', context)    

def runExisting(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        svm_cls = svm.SVC()
        svm_cls.fit(X_train, y_train)
        predict = svm_cls.predict(X_test)        
        img = calculateMetrics("Existing SVM", predict, y_test)
        algorithms = ["Existing SVM"]
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table>"
        context= {'data':output, 'img': img}
        return render(request, 'ViewResult.html', context)

def runPropose(request):
    if request.method == 'GET':
        global normal_records, X, Y
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore, elm_model
        X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1, random_state=0)
        srhl_tanh = MLPRandomLayer(n_hidden=1500, activation_func='hardlim')
        elm_model = GenELMClassifier(hidden_layer=srhl_tanh)
        elm_model.fit(X_train, y_train)
        predict = elm_model.predict(X_test)        
        img = calculateMetrics("Propose ELM", predict, y_test)
        algorithms = ["Existing SVM", "Propose ELM"]
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table>"
        context= {'data':output, 'img': img}
        return render(request, 'ViewResult.html', context)    

def Graphs(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore
        df = pd.DataFrame([['Existing SVM','Precision',precision[0]],['Existing SVM','Recall',recall[0]],['Existing SVM','F1 Score',fscore[0]],['Existing SVM','Accuracy',accuracy[0]],
                           ['Propose ELM','Precision',precision[1]],['Propose ELM','Recall',recall[1]],['Propose ELM','F1 Score',fscore[1]],['Propose ELM','Accuracy',accuracy[1]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(5, 3))
        plt.title("All Algorithms Performance Graph")
        #plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':'Comparison Graph', 'img': img_b64}
        return render(request, 'ViewResult.html', context)   

def PredictAction(request):
    if request.method == 'POST':
        global scaler, elm_model, labels
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("SoilApp/static/Data.csv"):
            os.remove("SoilApp/static/Data.csv")
        fs = FileSystemStorage()
        filename = fs.save('SoilApp/static/Data.csv', myfile)
        dataset = pd.read_csv('SoilApp/static/Data.csv')
        dataset.fillna(0, inplace = True)
        temp = dataset.values
        data = scaler.transform(temp)
        predict = elm_model.predict(data)
        print(predict)
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Data Values</th><th><font size="" color="black">Predicted Fertility</th></tr>'
        for i in range(len(predict)):
            output+='<tr><td><font size="" color="black">'+str(temp[i])+'</td><td><font size="" color="black">'+labels[int(predict[i])]+'</td></tr>'
        output += "</table><br/><br/><br/><br/>"    
        context= {'data':output}
        return render(request, 'ViewResult.html', context)    

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})
    
def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global userid
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "admin" and password == "admin":
            context= {'data':'Welcome '+user}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'AdminLogin.html', context)

