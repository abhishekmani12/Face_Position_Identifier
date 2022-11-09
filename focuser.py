
import mediapipe as mp
import cv2
import numpy as np

baseobj=mp.solutions.face_mesh
base_model=baseobj.FaceMesh(min_detection_confidence=0.4, min_tracking_confidence=0.4)

track="focused"
count=0
cap =cv2.VideoCapture(0)
while cap.isOpened():
    s, img=cap.read()
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #fm requires rgb input
    landmarks=base_model.process(img) #get keypoint landmarks mesh for a face
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    #Extracting eye edge(left - 33, right - 263), nose(1), mouth edge(left - 61, right - 291), chin(199) keypoint landmarks

    twoD=[] #x,y
    threeD=[] #axis
    
    h, w, c=img.shape
    
    if landmarks.multi_face_landmarks:
        for dat in landmarks.multi_face_landmarks:
            for i, cood in enumerate(dat.landmark):
                    if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:

                        x=int(cood.x*w) #multiplying width to x cood and height to y cood
                        y=int(cood.y*h)
                        z=cood.z


                        twoD.append([x,y])
                    

                        threeD.append([x,y,z])
                    

            twoD=np.array(twoD, dtype=np.float64)
            threeD=np.array(threeD, dtype=np.float64)
                    
                    
            focalpoint=1*w #fx, fy
            skew=0 #gamma
            u_cood=h/2
            v_cood=w/2

            #camera matrix

            cam_mat=np.array([
                                [focalpoint, 0, u_cood],
                                [0, focalpoint, v_cood],
                                [0, 0, 1]
                            ])

            #distance matrix
            dist_mat=np.zeros((4,1), dtype=np.float64)

            #pnp - convert 3d point in obj cood frame to 2d camera cood frame by getting rotation and translation vectors
            s, rot_v, trans_v=cv2.solvePnP(threeD, twoD, cam_mat, dist_mat)

            rot_mat, _ = cv2.Rodrigues(rot_v) # convert to matrix to get rot angle

            angle, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_mat) #extract angles

            xdegree=angle[0]*360
            ydegree=angle[1]*360

            if ydegree < -10 or ydegree > 10 or xdegree < -4:
                current="Not focused"
            else:
                current="focused"
                
            
            if track == "Not focused" and current == "focused":
                    count+=1
                    print("OFF Focus WARNING: ",count)
                    
            track=current       
            
                        
            #cv2.putText(img, current, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('focus', img)
                    
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break
                
cap.release()
cv2.destroyAllWindows()