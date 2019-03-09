import matplotlib.pyplot as plt
import numpy as np
import cv2

#open the webcam

objDetectCamera=cv2.VideoCapture(0)
HEIGHT=500
Row_Frames=[]

while(True):

    #read a new frame
    _, frame=objDetectCamera.read()

    #flip the frame
    frame=cv2.flip(frame,1)

    #rescaling camera output
    asspect=frame.shape[1]/float(frame.shape[0])
    res=int(asspect*HEIGHT) #landscape crientation - wide image
    frame=cv2.resize(frame,(res,HEIGHT))

    #add rectrangle
    cv2.rectangle(frame,(200,75),(75,200),(0,0,255),2)

    #show the frame
    cv2.imshow('finding garbage ',frame)

    #quite camara is 'q' key is pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('B'):
        #save the frames
        Row_Frames.append(frame)

        #preview the frame
        plt.imshow(frame)
        plt.show()

objDetectCamera.release()
cv2.destroyAllWindows()

