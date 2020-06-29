import cv2
from tkinter import *
from tkinter import messagebox
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle


window=Tk()
r=IntVar()
e=Entry(window,width=35)
e.pack()
erins=Label(window,text='Enter Folder Name').pack()

directory='C:/Python36/pic/'
images=[]
stId = []
encodeListKnown=None

def create():
    path='C:/Python36/pic/'+e.get()+'/'
    print(path)
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Error')
    cap= cv2.VideoCapture(0);
    i=1;
    j=1;
    while(i<=200):
        ret,frame = cap.read()
#        cv2.imshow('frame',frame)
        if ret == False:
            break
        if(i%10)==0:
            print(path+str(j/10))
            cv2.imwrite(os.path.join(path+str(j/10)+'.jpg'),frame)
        i+=1
        j+=1
    cap.release()
    
#taking images from every sub-directory of the givrn directory assigning the labels
for path,subdirnames,filenames in os.walk(directory):
    for filename in filenames:
        if filename.startswith("."):
            print("Skipping system file")
            continue

        stId.append(os.path.basename(path))#fetching subdirectory names
        img_path=os.path.join(path,filename)#fetching image path
        #print("img_path:",img_path)
        #print("id:",stId)
        test_img=cv2.imread(img_path)#loading each image one by one
        if test_img is None:
            print("Image not loaded properly")
            continue
        images.append(test_img)

print(len(stId))
for id in stId:
    print(id)

#marking attendence in excel
def markAttendance(name):
    with open('aa.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
 

  

#for training data   
def one():
    #extracting the encodings from every images
    def findEncodings(images,stId):
        encodeList = []
        for img,id in zip(images,stId):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            if len(encode) > 0:
                    encoding=encode[0]
                    encodeList.append(encoding)
            else:
                    print("Face Not Found")
                    print(id)
                    quit()
        return encodeList
    encodeListKnown = findEncodings(images,stId)

    #saving the encoding so that for subsequent runs there is no need for extracting encoding every time
    with open('trainingData.dat','wb') as f:
        pickle.dump(encodeListKnown,f)
    print('training Complete')

#for testing data
def two():
    with open('trainingData.dat','rb') as f:
        encodeListKnown = pickle.load(f)
    print (encodeListKnown)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)
 
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            #print(matches)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            #print(matchIndex)
            
            if faceDis[matchIndex]< 0.50:
                name = stId[matchIndex].upper()
                print(name)
                markAttendance(name)
            else:
                name = 'Unknown'
                print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
       
        cv2.imshow('web',img)
        cv2.waitKey(1)


addNewMem=Button(window,text="Add New Member",command=create).pack()
train=Button(window,text="Train",command=one).pack()
test=Button(window,text="Test",command=two).pack()


window.minsize(1280,800)
window.maxsize(1280,800)
window.geometry("1280x750+150+150")
window.mainloop()
