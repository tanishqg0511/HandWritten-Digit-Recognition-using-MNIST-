import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

Boundary=5
WINDOWSIZEX=850
WINDOWSIZEY=720
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
IMAGESAVE=False
img_cnt=1
MODEL=load_model("bestmodel.h5")
isWriting=False
predict=False   
LABELS={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}
pygame.init()
FONT=pygame.font.Font()
DISPLAYSURFACE=pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))
pygame.display.set_caption("DIGIT BOARD")
number_xcord=[]
number_ycord=[]
while True:
    predict=False
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and isWriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURFACE,WHITE,(xcord,ycord),4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type==MOUSEBUTTONDOWN:
            isWriting=True

        if event.type==MOUSEBUTTONUP:
            isWriting=False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)
            rect_min_x,rect_max_x=max(number_xcord[0]-Boundary,0),min(WINDOWSIZEX,number_xcord[-1]+Boundary)
            rect_min_y,rect_max_y=max(number_ycord[0]-Boundary,0),min(WINDOWSIZEX,number_ycord[-1]+Boundary)
            number_xcord=[]
            number_ycord=[]
            img_arr=np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            predict=True

        if IMAGESAVE:
            cv2.imwrite("image.png")
            img_cnt+=1

        if predict:
            image=cv2.resize(img_arr,(28,28))
            image=np.pad(image,(10,10),'constant',constant_values=0)
            image=cv2.resize(image,(28,28))/255

            label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
            textSurface=FONT.render(label,True,RED,WHITE)
            textRectObj=textSurface.get_rect()
            textRectObj.left,textRectObj.bottom=rect_min_x,rect_max_y

            DISPLAYSURFACE.blit(textSurface,textRectObj)

        if event.type==KEYDOWN:
            if event.unicode=="n":
                DISPLAYSURFACE.fill(BLACK)
    
    pygame.display.update()