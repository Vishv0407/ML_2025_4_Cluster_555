1. Abnormal track IDs are given in respective "AbnormalTracks_X.txt" file. 
   1.1. If you don't find track ID in respective folder "X" for 
        which entry is given in text file then try to find it by its
        first track ID.
   	Ex: For 12, 666.csv is given in text file but it is not there in its respective folder. 
          Then you can use a CSV file starting from 
          666, such as 666_761_797_807_872_923_929_950_1056_1062_1083_.csv 	
2. You can use given image for visualising a track. 
3. You will only use these features from .csv file "frameNo,left,top,w,h". 
   3.1 frame number can be used as time step.  
   3.2 (left,top)- top left corner of bounding box. 
   3.3 (w,h)- Width and Height of bounding box.  
