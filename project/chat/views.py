# pylint: disable=all

from django.shortcuts import render, redirect
from chat.models import Room, Message
from django.http import JsonResponse
from joblib import load
import pandas as pd
from sklearn.svm import SVC
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404, redirect
from .models import Room
# result=load('tdf_vectorizer')
# result = load('model.bin')
# Create your views here.
def home(request):
    return render(request, 'home.html')

def room(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    return render(request, 'room.html', {
        'username': username,
        'room': room,
        'room_details': room_details
    })
def checkview(request):
    room_name = request.POST['room_name']
    username = request.POST['username']

    # Check if the room exists
    if Room.objects.filter(name=room_name).exists():
        room = Room.objects.get(name=room_name)
    else:
        # If the room doesn't exist, create a new room
        room = Room.objects.create(name=room_name)
        room.save()

    # Redirect the user to the room and include the username and room ID in the URL
    url = f'/{room_name}/?username={username}&room_id={room.id}'
    return redirect(url)

def send(request):
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']
    #svm_predictions=model.predict("I will kill ,age 50")
    
    #message = message # + ':===== Cyberbullying type : [' + custom_input_prediction(message)+']'
    #message = message # + is_bullying(message)
    
    new_message = Message.objects.create(value=message, user=username, room=room_id)
    new_message.save()
    result = custom_input_prediction(message)
    prediction_list = ["Age", "Ethnicity", "Gender","Other Cyberbullying", "Religion"]
    if result in prediction_list :
     #return render(request, 'django-chat-app-main/alert.html', {'alert_message': "Warning: This message may be bullying!"})
     return JsonResponse({'status': 'Warning', 'message' : result })
    else :
     #return HttpResponse('Message sent successfully')
     return JsonResponse({'status': 'success'})

def getMessages(request, room):
    room_details = Room.objects.get(name=room)
    messages = Message.objects.filter(room=room_details.id)
    return JsonResponse({"messages":list(messages.values())})



def exit_chatroom(request,room_id):
    if request.method == 'POST':
        # Get the room ID from the POST request
        room_id = request.POST.get('room_id')

        # Remove the user from the chatroom
        # Your code to remove the user from the chatroom goes here

        # Redirect the user to the homepage
        return redirect('home')



#===========================================================================================================================
# preprocessing functions

# converting tweet text to lower case
def text_lower(text):
    return text.str.lower()

# removing stopwoords from the tweet text
def clean_stopwords(text):
    # stopwords list that needs to be excluded from the data
    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
    STOPWORDS = set(stopwordlist)
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# cleaning and removing punctuations
import re
import string
def clean_puctuations(text):
    english_puctuations = string.punctuation
    translator = str.maketrans('','', english_puctuations)
    return text.translate(translator)

# cleaning and removing repeating characters
def clean_repeating_characters(text):
    return re.sub(r'(.)1+', r'1', text)

# cleaning and removing URLs
def clean_URLs(text):
    return re.sub(r"((www.[^s]+)|(http\S+))","",text)

# cleaning and removing numeric data
def clean_numeric(text):
    return re.sub('[0-9]+', '', text)

# Tokenization of tweet text
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer,  WordNetLemmatizer
def tokenize_tweet(text):
    tokenizer = RegexpTokenizer('\w+')
    text = text.apply(tokenizer.tokenize)
    return text

# stemming    
def text_stemming(text):
    st = PorterStemmer()
    text = [st.stem(word) for word in text]
    return text

# lemmatization
from nltk import PorterStemmer, WordNetLemmatizer
def text_lemmatization(text):
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text]
    return text

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
def preprocess(text):
    text = text_lower(text)
    text = text.apply(lambda text: clean_stopwords(text))
    text = text.apply(lambda x : clean_puctuations(x))
    text = text.apply(lambda x: clean_repeating_characters(x))
    text = text.apply(lambda x : clean_URLs(x))
    text = text.apply(lambda x: clean_numeric(x))
    text = tokenize_tweet(text)
    text = text.apply(lambda x: text_stemming(x))
    text = text.apply(lambda x: text_lemmatization(x))
    text = text.apply(lambda x : " ".join(x))
    return text

# Function for custom input prediction
import pickle
def custom_input_prediction(text):
    # import nltk
    # nltk.download('omw-1.4')
    text = pd.Series(text)
    text = preprocess(text)
    text = [text[0],]
    vectoriser = pickle.load(open(r'C:\Users\DELL\Desktop\project\savedModels\tdf_vectorizer', "rb"))
    text = vectoriser.transform(text)
    model1 = pickle.load(open(r'C:\Users\DELL\Desktop\project\savedModels\model.bin', "rb"))
    prediction = model1.predict(text)
    prediction = prediction[0]

    interpretations = {
        0 : "Age",
        1 : "Ethnicity",
        2 : "Gender",
        3 : "Not Cyberbullying",
        4 : "Other Cyberbullying",
        5 : "Religion"
    }

    for i in interpretations.keys():
        if i == prediction:
            return interpretations[i]
    
