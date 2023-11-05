from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth import authenticate,login
from signup.models import Signup
from video.models import Video
import os
from django.core.files import File
from   video.models import vid,Video
import csv
import pandas as pd
import openai
import numpy as np
import re
# Create your views here.

def loginaction(request):
    # if request.method=='POST':
    #     name=request.POST.get('name')
    #     password=request.POST.get('password')
    # signup=Signup.objects.all()
    # for x in signup:
    #     if x.name==name:
    #         return render(request,"play.html")    
    # return HttpResponse("login.html")
    return render(request,"login.html")  
def open(request):
    video=Video.objects.filter(status='verify')
    return render(request,"open.html",{'video':video})  
def analysis(request):
    fighting_active=0
    shooting_active=0
    road_active=0
    robbery_active=0
    abuse_active=0
    arrest_active=0
    arson_active=0
    assault_active=0
    burglary_active=0
    explosion_active=0
    fighting_close=0
    shooting_close=0
    road_close=0
    robbery_close=0
    abuse_close=0
    arrest_close=0
    arson_close=0
    assault_close=0
    burglary_close=0
    explosion_close=0
    return render(request,"analysis.html",{'fighting_active':fighting_active,'fighting_close':fighting_close,'explosion_active':explosion_active,'explosion_close':explosion_close,
                                           'burglary_active':burglary_active,'burglary_close':burglary_close,'assault_active':assault_active,'assault_close':assault_close,
                                           'shooting_active':shooting_active,'shooting_close':shooting_close,'arson_active':arson_active,'arson_close':arson_close,
                                           'arrest_active':arrest_active,'arrest_close':arrest_close,'abuse_active':abuse_active,'abuse_close':abuse_close,
                                           'robbery_active':robbery_active,'robbery_close':robbery_close,'road_active':road_active,'road_close':road_close})
def about(request):
    cities=[]
    defcrime=[]
    data=[]
    cities.append("Aligarh")
    cities.append("Kanpur") 
    cities.append("Gorakhpur")
    cities.append("Lucknow") 
    cities.append("Lal Bahadur Nagar")
    defcrime.append("Fighting") 
    defcrime.append("Shooting") 
    defcrime.append("RoadAccidents") 
    defcrime.append("Robbery") 
    defcrime.append("Abuse") 
    defcrime.append("Arrest") 
    defcrime.append("Arson") 
    defcrime.append("Assault") 
    defcrime.append("Burglary") 
    defcrime.append("Explosion")
    for i in cities:
        formraw=[]
        for j in defcrime:
            formraw.append(int(Video.objects.filter(location=i,caption=j,description='CLOSED').count()+Video.objects.filter(location=i,caption=j,description='ACTIVE').count()))
        data.append(formraw) 
    return render(request,"about.html",{'data':data})
def contact(request):
    return render(request,"contact.html")
def filter1(request):
    if request.method=='POST':
        location=request.POST.get('location')
        date=request.POST.get('date')
    #     time=request.POST.get('time') 
        video=Video.objects.filter(date=date,location=location)    
    return render(request,"play.html",{'video':video})
def ans(request):
    b="NOT FOUND"
    if request.method=='POST':
        ques=request.POST.get('ques')
        answer=ques
        answer2 = re.sub("[^\w]", " ", answer).split()
        cities=[]
        crimes=[]
        defcrime=[]
        data=[]
        ans3={}
        ans4={}
        aligarh=0
        lucknow=0
        gorakhpur=0
        kanpur=0
        for i in answer2:
            if i.lower()=='lucknow' or i.lower()=='gorakhpur' or i.lower()=='aligarh' or i.lower()=='kanpur':
                cities.append(i.capitalize())
            # if i.lower()=='robbery' or i.lower()=='arrest' or i.lower()=='assault' or i.lower()=='abuse':
            #     crime.append(i.capitalize())
        openai.api_key= "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" 
        ques_new = ques + "." + "Choose one of the following crimes Fighting,Shooting,RoadAccidents,Robbery,Abuse,Arrest,Arson,Assault,Burglary or Explosion are depicted from previous statement?"
        ans9= openai.Completion.create(engine="text-davinci-003",prompt=ques_new,max_tokens=1000)
        z=ans9.choices[0]['text']
        crime=z[2::] 
        crimes.append(crime)
        for i in answer2:
            if i.lower()=='robbery' or i.lower()=='arrest' or i.lower()=='assault' or i.lower()=='abuse':
                crimes.append(i.capitalize())
        for i in cities:
            if i.lower()=="aligarh":
                aligarh=1
            if i.lower()=="lucknow":    
                lucknow=1
            if i.lower()=="gorakhpur":
                gorakhpur=1
            if i.lower()=="kanpur":
                kanpur=1 
        cities=[]   
        if aligarh==1:
            cities.append("Aligarh")
        if kanpur==1:
            cities.append("Kanpur")  
        if gorakhpur==1:
            cities.append("Gorakhpur")   
        if lucknow==1:
            cities.append("Lucknow")
        c=len(cities)  
        d=len(crimes)      
        defcrime.append("Fighting") 
        defcrime.append("Shooting") 
        defcrime.append("RoadAccidents") 
        defcrime.append("Robbery") 
        defcrime.append("Abuse") 
        defcrime.append("Arrest") 
        defcrime.append("Arson") 
        defcrime.append("Assault") 
        defcrime.append("Burglary") 
        defcrime.append("Explosion")
        # if d==0:
        #     crime.append("Fighting") 
        #     crime.append("Shooting") 
        #     crime.append("RoadAccidents") 
        #     crime.append("Robbery") 
        #     crime.append("Abuse") 
        #     crime.append("Arrest") 
        #     crime.append("Arson") 
        #     crime.append("Assault") 
        #     crime.append("Burglary") 
        #     crime.append("Explosion")  
        # d=len(crime) 
        # if c==0:
        #     cities.append("Aligarh") 
        #     cities.append("Kanpur") 
        #     cities.append("Gorakhpur") 
        #     cities.append("Lucknow")  
        for i in cities:
            formraw=[]
            for j in defcrime:
                formraw.append(int(Video.objects.filter(location=i,caption=j,description='CLOSED').count()+Video.objects.filter(location=i,caption=j,description='ACTIVE').count()))
            data.append(formraw)    
        for i in cities:
            form={}
            for j in crimes:
                sum=Video.objects.filter(location=i,caption=j,description='CLOSED').count()+Video.objects.filter(location=i,caption=j,description='ACTIVE').count() 
                form[j]=sum
            ans3[i]=form  
        for i in cities:
            form2={}
            for j in crimes:
                sum2=Video.objects.filter(location=i,caption=j,description='CLOSED').count() 
                form2[j]=sum2
            ans4[i]=form2  
        openai.api_key= "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" 
        ques_new1 = ques 
        ans= openai.Completion.create(engine="text-davinci-003",prompt=ques_new1,max_tokens=1000)
        b=ans.choices[0]['text']
    return render(request,"about.html",{'aligarh':aligarh,'kanpur':kanpur,'gorakhpur':gorakhpur,'lucknow':lucknow,'cities': cities , 'crime':crime , 'ans3':ans3 ,'ans4':ans4,'data':data,'defcrime':defcrime,'c':c,'b':b,'ques':ques,'crimes':crimes})
def filter3(request):
    if request.method=='POST':
        location=request.POST.get('location')
        video=Video.objects.filter(location=location,status='verify') 
        fighting_active=0
        shooting_active=0
        road_active=0
        robbery_active=0
        abuse_active=0
        arrest_active=0
        arson_active=0
        assault_active=0
        burglary_active=0
        explosion_active=0
        fighting_close=0
        shooting_close=0
        road_close=0
        robbery_close=0
        abuse_close=0
        arrest_close=0
        arson_close=0
        assault_close=0
        burglary_close=0
        explosion_close=0
        for x in video:
            if x.status=='verify' and x.caption=='Fighting' and x.description=='ACTIVE':
                fighting_active=fighting_active+1    
            elif x.status=='verify' and x.caption=='Fighting' and x.description=='CLOSED':
                fighting_close=fighting_close+1 
            elif x.status=='verify' and x.caption=='Shooting' and x.description=='ACTIVE':
                shooting_active=shooting_active+1    
            elif x.status=='verify' and x.caption=='Shooting' and x.description=='CLOSED':
                shooting_close=shooting_close+1 
            elif x.status=='verify' and x.caption=='RoadAccidents' and x.description=='ACTIVE':
                road_active=road_active+1    
            elif x.status=='verify' and x.caption=='RoadAccidents' and x.description=='CLOSED':
                road_close=road_close+1 
            elif x.status=='verify' and x.caption=='Robbery' and x.description=='ACTIVE':
                robbery_active=robbery_active+1    
            elif x.status=='verify' and x.caption=='Robbery' and x.description=='CLOSED':
                robbery_close=robbery_close+1 
            elif x.status=='verify' and x.caption=='Abuse' and x.description=='ACTIVE':
                abuse_active=abuse_active+1    
            elif x.status=='verify' and x.caption=='Abuse' and x.description=='CLOSED':
                abuse_close=abuse_close+1 
            elif x.status=='verify' and x.caption=='Arrest' and x.description=='ACTIVE':
                arrest_active=arrest_active+1    
            elif x.status=='verify' and x.caption=='Arrest' and x.description=='CLOSED':
                arrest_close=arrest_close+1 
            elif x.status=='verify' and x.caption=='Arson' and x.description=='ACTIVE':
                arson_active=arson_active+1    
            elif x.status=='verify' and x.caption=='Arson' and x.description=='CLOSED':
                arson_close=arson_close+1 
            elif x.status=='verify' and x.caption=='Assault' and x.description=='ACTIVE':
                assault_active=assault_active+1    
            elif x.status=='verify' and x.caption=='Assault' and x.description=='CLOSED':
                assault_close=assault_close+1 
            elif x.status=='verify' and x.caption=='Burglary' and x.description=='ACTIVE':
                burglary_active=burglary_active+1    
            elif x.status=='verify' and x.caption=='Burglary' and x.description=='CLOSED':
                burglary_close=burglary_close+1 
            elif x.status=='verify' and x.caption=='Explosion' and x.description=='ACTIVE':
                explosion_active=explosion_active+1    
            elif x.status=='verify' and x.caption=='Explosion' and x.description=='CLOSED':
                explosion_close=explosion_close+1     
    return render(request,"analysis.html",{'fighting_active':fighting_active,'fighting_close':fighting_close,'explosion_active':explosion_active,'explosion_close':explosion_close,
                                           'burglary_active':burglary_active,'burglary_close':burglary_close,'assault_active':assault_active,'assault_close':assault_close,
                                           'shooting_active':shooting_active,'shooting_close':shooting_close,'arson_active':arson_active,'arson_close':arson_close,
                                           'arrest_active':arrest_active,'arrest_close':arrest_close,'abuse_active':abuse_active,'abuse_close':abuse_close,
                                           'robbery_active':robbery_active,'robbery_close':robbery_close,'road_active':road_active,'road_close':road_close})
def filter2(request):
    if request.method=='POST':
        fir=request.POST.get('fir')
        if fir=='Default':
            video=Video.objects.filter(status='verify')
        else:
            video=Video.objects.filter(fir=fir,description='ACTIVE')    
    return render(request,"play2.html",{'video':video})
def register(request):
    # if request.method=='POST':
    #     name=request.POST.get('name') 
    #     token=request.POST.get('token') 
    #     signup=Signup.objects.all()
    #     return HttpResponse("kkkkkk")
        # for x in signup:
        #     if name== x.name:
        #         return render(request,"play.html")
    
        # return HttpResponse("INCORRECT INFO")  
  
    # if request.method=='POST':
    #     name=request.POST.get('name') 
    #     token=request.POST.get('token') 
    #     signup=Signup.authenticate(name=name,token=token) 
    #     if signup is not None:
    #         # login(request,signup)  
    #         return redirect('video')
    #     else:
    #         return HttpResponse("INCORRECT INFO")
    # return render(request,"play.html")
    # return redirect('video')  
     return render(request,"play.html")