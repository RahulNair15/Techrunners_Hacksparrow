import re
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render, HttpResponse
import random
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth
from .models import UserInput
from textblob import TextBlob
from .models import ClientInfo, UploadImag

#from .models import Response, models

# Remove the comments to download additional nltk packages
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow import keras
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

q1 = ""

def index(request):
   return render(request , 'index.html')


def chatBot(request):
   query = str(request.GET.get("query"))
   print(query)
   q1 = query

   obj = UserInput.objects.create(username = request.user.username, chats=query)
   obj.save()

   lemmatizer=WordNetLemmatizer()

   with open('jsonfile.json') as fileobj:
      readobj=json.load(fileobj)

   words=[]
   classes=[]
   documents=[]
   stop_words=['is','and','the','a','are','i','it']
   cleaned_word=[]

   for intent in readobj['intents']:
      for pattern in intent['patterns']:
         word_list=word_tokenize(pattern)
         words.extend(word_list)
         documents.append((word_list,intent['tag']))
         if intent['tag'] not in classes:
               classes.append(intent['tag'])

   # vectorizer = TfidfVectorizer()
   # X = vectorizer.fit_transform(cleaned_word)

   words = sorted(set(words))
   classes = sorted(set(classes))

   def clean_corpus(words):
      words = [doc.lower() for doc in words]
      for w in words:
         if w not in stop_words and w.isalpha():
               w=lemmatizer.lemmatize(w)
               cleaned_word.append(w)
      return(cleaned_word)

   cleaned_list = clean_corpus(words)

   words = sorted(set(words))
   classes = sorted(set(classes))
   # y = encoder.fit_transfrom(np.array(classes), reshape(-1, 1))

   training =[]
   output_empty = [0] * len(classes)


   for document in documents:
      bag =[]
      word_patterns = document[0]

      word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

      for word in words :
         bag.append(1) if word in word_patterns else bag.append(0)
      output_row = list(output_empty)
      output_row[classes.index(document[1])] = 1 
      training.append([bag,output_row])

   '''Neural Network'''
   random.shuffle(training)
   training = np.array(training,dtype=object)
   train_x = list(training[:,0])
   train_y = list(training[:,1])

   model = Sequential([Dense(128, input_shape = (len(train_x[0]),),activation='relu'),Dropout(0.2),Dense(64, activation='relu'),Dropout(0.2),Dense(len(train_y[0]),activation = 'softmax')])
   initial_learning_rate = 0.01
   lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)
   sgd = tensorflow.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
   model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
   history = model.fit(np.array(train_x),np.array(train_y),epochs=20,batch_size=1,verbose=7)
   # model.save('chatbot_model.model',history)


   def cleaning_up_message(message):
      message_word = word_tokenize(message)
      message_word = [lemmatizer.lemmatize(word.casefold()) for word in message_word]
      return message_word


   def bag_of_words(message):
      message_word = cleaning_up_message(message)
      bag = [0]*len(words)
      for w in message_word:
         for i, word in enumerate(words):
               if word == w:
                  bag[i] = 1
      return np.array(bag)
      
   INTENT_NOT_FOUND_THRESHOLD = 0.25

   def predict_intent_tag(message):
      BOW = bag_of_words(message)
      res = model.predict(np.array([BOW]))[0]
      results = [[i,r] for i,r in enumerate(res) if r > INTENT_NOT_FOUND_THRESHOLD ]
      results.sort(key = lambda x : x[1] , reverse = True)
      return_list = []
      for r in results:
         return_list.append({ 'intent': classes[r[0]], 'probability': str(r[1]) })
      return return_list

   def get_response(intents_list , intents_json):
      tag = intents_list[0]['intent']
      list_of_intents = intents_json['intents']
      for i in list_of_intents:
         if i['tag'] == tag :
               result= random.choice (i['responses'])
               break
         # elif i['tag']!=tag:
         #     result=random.choice(["Invalid response","Sorry:), I didn't get what you are saying","Pardon me..!"])
         #     break
      return result

   print(" Aapi is Running ! ")
   message = query
   ints = predict_intent_tag(message)
   bot_response = get_response(ints, readobj)
   return JsonResponse({"Bot": bot_response})
   

def login(request):
   if request.method =='POST':
      username = request.POST['username']
      password = request.POST['password']

      user = auth.authenticate(username=username, password=password)
      if user is not None:
         auth.login(request, user)
         return redirect('/')
      else:
         messages.info(request, "Invalid credentials!!!")
         return redirect('/login')
   else:
      return render(request, 'login.html')
   
def signup(request):
    if request.method == 'POST':
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        if User.objects.filter(username = username).exists():
            messages.info(request, 'Username taken, signup again!!!')
            return redirect('/signup')
        elif User.objects.filter(email = email).exists():
            messages.info(request, 'e-mail taken, signup again!!!')
            return redirect('/signup')
        else:
            user = User.objects.create_user(username = username, first_name = fname, last_name = lname, password = password, email = email)
            user.save()
            return redirect('/login')
    else:    
        return render(request, 'signup.html')  
    
def profile(request):
    return render(request, "profile.html")

def logout(request):
    auth.logout(request)
    return redirect("/")

# new code

def load_response_data(jsonfile):
    print("okkkkk")
    with open(jsonfile.json, 'r') as file:
        print("test1")
        response_data = json.load(file)
    return response_data

def tailor_response(q1, response_data):
    predicted_intent = predict_intent(q1)
    sentiment_score, sentiment_label = analyze_sentiment(q1)

    # Select response based on predicted intent and sentiment label from the loaded response data
    if predicted_intent in response_data['intents']:
        intent_responses = response_data['intents'][predicted_intent]['responses']
        if sentiment_label in intent_responses:
            response = random.choice(intent_responses[sentiment_label])
        else:
            # If sentiment-specific response not available, use a generic response for the intent
            response = random.choice(intent_responses.values())
    else:
        # If intent not recognized, provide a default response
        response = "I'm sorry, I'm not sure how to respond to that."

    # Modify response based on specific conditions if needed
    if sentiment_label == "positive":
         response += " Your positivity is infectious!"
         print(response)
    elif sentiment_label == "negative":
        response += " I'm here to help. What seems to be the problem?"
        print(response)
    return response

def predict_intent(q1):
    # Replace with your intent prediction logic
    print("hurrey")
    return "greeting"  # Example value

def analyze_sentiment():
    blob = TextBlob("Food was very bad")
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        sentiment_label = 'positive'
        print(sentiment_label)
    elif sentiment_score < 0:
        sentiment_label = 'negative'
        print(sentiment_label)
    else:
        sentiment_label = 'neutral'
        print(sentiment_label)
    return sentiment_score, sentiment_label

def clientInform(request):
    clientInfos =  ClientInfo.objects.filter(username = request.user.username)
    return render(request, 'clientInfo.html',  {"clientInfos": clientInfos})

def uploadImg(request):
    if request.method == "POST":
        username = request.POST['username']
        img = request.FILES['img']
        obj = UploadImag.objects.create(username = username, img = img)
        
        obj.save()
        return redirect("/")
    return render(request, 'uploadImage.html')
    