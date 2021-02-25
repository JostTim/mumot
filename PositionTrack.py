# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:07:19 2020

@author: Timothe
"""

import cv2
from skimage import measure

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from statistics import mean
import math

import os, sys
import logging
import pickle

import pyprind

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__filename__"))))
print(os.path.dirname(os.path.dirname(os.path.abspath("__name__"))))
from LibUtils import network, image, database_IO, geometry, strings, signal
#from LibrairieVSDAna import ReadVSDfile


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Trackers_find':
            return Trackers_find
        if name == 'TrackerInstance':
            return TrackerInstance
        if name == 'TrackerMesh':
            return TrackerMesh
        return super().find_class(module, name)

class Trackers_find():#Lightweighted version of the class that was used for the first version of trajectory tracking

    def __init__(self,binImage,areamin,areamax,**kwargs):
        """
        Trackers_find extracts the elements : 'coords','orientation','area','indexs','surface','bbox','eccentricity','major_axis_length', 'minor_axis_length','image','moments_hu'
        from the shapes whos areas are between areamin and areamax, from a binarized image.
        """
        debug = kwargs.get("debug", False)
        labels = measure.label(binImage, background=1)
        props = measure.regionprops(labels,coordinates='rc')

        #coords = np.empty((0,2))



        self.items = database_IO.df_empty(columns=['coords','orientation','area','indexs','surface','bbox','eccentricity','major_axis_length', 'minor_axis_length','image','moments_hu'], dtypes=[object,float,int,int,object,object,float,float,float,object,object])

        self.area = []
        self.index = []

        if debug :
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(binImage)

        for index, label in enumerate(props):
            if debug and label.area > 100 and  label.area < 20000 and binImage[label.coords[0,0],label.coords[0,1]] != 0 :
                print(f"Label at : {label.centroid}, area {label.area}, eccentricity {label.eccentricity}, MajAL {label.major_axis_length} , MinAL {label.minor_axis_length} ")

                plt.plot(label.coords[:,1],label.coords[:,0],linewidth = 1)
                plt.text(label.coords[0,1],label.coords[0,0],str(label.area))
            # if label.area > areamin and label.area < areamax :
            #     if binImage[label.coords[0,0],label.coords[0,1]] != 0:
            #         if label.eccentricity < 0.80 and label.eccentricity > 0.15 :
            #             if label.major_axis_length < 65 and label.major_axis_length > 25 :
            #                 if label.minor_axis_length < 45 and label.minor_axis_length > 25 :
            if label.area > areamin and label.area < areamax :
                if binImage[label.coords[0,0],label.coords[0,1]] != 0:
                #     if label.eccentricity < 0.75 and label.eccentricity > 0.45 :
                #         if label.major_axis_length < 65 and label.major_axis_length > 45 :
                #             if label.minor_axis_length < 55 and label.minor_axis_length > 28 :
                    y0, x0 = label.centroid

                    #row =  pd.DataFrame({"A": range(3)})

                    #tracker.coords[found] = [x0, y0]

                    #self.coords = np.append(self.coords,np.array([[x0, y0]]),axis = 0)
                    #tracker.orientation[found] = label.orientation
                    #self.area.append(label.area)
                    #self.index.append(index)

                    self.items = self.items.append({'coords': [x0, y0], 'orientation': label.orientation, 'area': label.area, 'indexs': index, 'surface' : label.coords, 'bbox' : label.bbox, 'eccentricity' : label.eccentricity, 'major_axis_length' : label.major_axis_length, 'minor_axis_length' : label.minor_axis_length, 'image': label.filled_image ,'moments_hu' : label.moments_hu}, ignore_index=True)
        if debug :
            plt.show()


# class Trackers_find():

#     def __init__(self,binImage,areamin,areamax,**kwargs):
#         """
#         Trackers_find extracts the elements : 'coords','orientation','area','indexs','surface','bbox','eccentricity','major_axis_length', 'minor_axis_length','image','moments_hu'
#         from the shapes whos areas are between areamin and areamax, from a binarized image.
#         """
#         debug = kwargs.get("debug", False)
#         labels = measure.label(binImage, background=1)
#         props = measure.regionprops(labels,coordinates='rc')

#         #coords = np.empty((0,2))



#         self.items = database_IO.df_empty(columns=['coords','orientation','area','indexs','surface','bbox','eccentricity','major_axis_length', 'minor_axis_length','image','moments_hu'], dtypes=[object,float,int,int,object,object,float,float,float,object,object])

#         self.area = []
#         self.index = []

#         if debug :
#             import matplotlib.pyplot as plt
#             plt.figure()
#             plt.imshow(binImage)

#         for index, label in enumerate(props):
#             if debug and label.area > 100 and  label.area < 20000 and binImage[label.coords[0,0],label.coords[0,1]] != 0 :
#                 print(f"Label at : {label.centroid}, area {label.area}, eccentricity {label.eccentricity}, MajAL {label.major_axis_length} , MinAL {label.minor_axis_length} ")

#                 plt.plot(label.coords[:,1],label.coords[:,0],linewidth = 1)
#                 plt.text(label.coords[0,1],label.coords[0,0],str(label.area))
#             # if label.area > areamin and label.area < areamax :
#             #     if binImage[label.coords[0,0],label.coords[0,1]] != 0:
#             #         if label.eccentricity < 0.80 and label.eccentricity > 0.15 :
#             #             if label.major_axis_length < 65 and label.major_axis_length > 25 :
#             #                 if label.minor_axis_length < 45 and label.minor_axis_length > 25 :
#             if label.area > areamin and label.area < areamax :
#                 if binImage[label.coords[0,0],label.coords[0,1]] != 0:
#                 #     if label.eccentricity < 0.75 and label.eccentricity > 0.45 :
#                 #         if label.major_axis_length < 65 and label.major_axis_length > 45 :
#                 #             if label.minor_axis_length < 55 and label.minor_axis_length > 28 :
#                     y0, x0 = label.centroid

#                     #row =  pd.DataFrame({"A": range(3)})

#                     #tracker.coords[found] = [x0, y0]

#                     #self.coords = np.append(self.coords,np.array([[x0, y0]]),axis = 0)
#                     #tracker.orientation[found] = label.orientation
#                     #self.area.append(label.area)
#                     #self.index.append(index)

#                     self.items = self.items.append({'coords': [x0, y0], 'orientation': label.orientation, 'area': label.area, 'indexs': index, 'surface' : label.coords, 'bbox' : label.bbox, 'eccentricity' : label.eccentricity, 'major_axis_length' : label.major_axis_length, 'minor_axis_length' : label.minor_axis_length, 'image': label.filled_image ,'moments_hu' : label.moments_hu}, ignore_index=True)
#         if debug :
#             plt.show()

#     def RemoveExtremaPos(self, INMindist, INMaxdist):

#         if self.items.shape[0] > 2:
#             #print("entred extrema pos--------------------")
#             couplesList = []
#             for I in range(self.items.shape[0]):
#                 for J in range(self.items.shape[0]):
#                     if I == J :
#                         continue
#                     dist = geometry.Distance(self.items.coords[I][0], self.items.coords[I][1], self.items.coords[J][0], self.items.coords[J][1])
#                     if dist >= INMindist and dist <= INMaxdist :
#                         if J > I :

#                             tempAppend = [I,J]
#                         else :
#                             tempAppend = [J,I]
#                         stop = 0
#                         for T in range(len(couplesList)):
#                             if couplesList[T] == tempAppend :
#                                 stop = 1
#                         if stop :
#                             break
#                         else :
#                             #print(f"Distance {dist} for couple {tempAppend}")
#                             couplesList.append(tempAppend)

#             if len(couplesList) > 0:
#                 #print(couplesList)
#                 temp = self.items.copy()
#                 temp = temp.drop(np.arange(0, self.items.shape[0],1).tolist())
#                 #display(self.items)
#                 treated = []
#                 loc = 0
#                 for I in range(len(couplesList)):
#                     #print(f"couple {couplesList[I]}")

#                     if self.items.coords[couplesList[I][0]][1] < self.items.coords[couplesList[I][1]][1] :
#                         if couplesList[I][0] not in treated :
#                             #print(f"{couplesList[I][0]} = {self.items.coords[couplesList[I][0]][1]} is front")
#                             temp.loc[loc] = self.items.loc[couplesList[I][0]]
#                             loc = loc + 1
#                             treated.append(couplesList[I][0])
#                         else :
#                             #print(f"{couplesList[I][1]} = {self.items.coords[couplesList[I][1]][1]} is already in dataframe")
#                             pass
#                         if couplesList[I][1] not in treated :
#                             #print(f"{couplesList[I][1]} = {self.items.coords[couplesList[I][1]][1]} is back")
#                             temp.loc[loc] = self.items.loc[couplesList[I][1]]
#                             loc = loc + 1
#                             treated.append(couplesList[I][1])
#                         else :
#                             #print(f"{couplesList[I][1]} = {self.items.coords[couplesList[I][1]][1]} is already in dataframe")
#                             pass
#                     else :
#                         if couplesList[I][1] not in treated :
#                             #print(f"{couplesList[I][1]} = {self.items.coords[couplesList[I][1]][1]} is front")
#                             temp.loc[loc] = self.items.loc[couplesList[I][1]]
#                             loc = loc + 1
#                             treated.append(couplesList[I][1])
#                         else :
#                             #print(f"{couplesList[I][1]} = {self.items.coords[couplesList[I][1]][1]} is already in dataframe")
#                             pass
#                         if couplesList[I][0] not in treated :
#                             #print(f"{couplesList[I][0]} = {self.items.coords[couplesList[I][0]][1]} is back")
#                             temp.loc[loc] = self.items.loc[couplesList[I][0]]
#                             loc = loc + 1
#                             treated.append(couplesList[I][0])
#                         else :
#                             #print(f"{couplesList[I][1]} = {self.items.coords[couplesList[I][1]][1]} is already in dataframe")
#                             pass
#                 self.items = temp

#     def GetContour(self,binImage):

#         TempContours = measure.find_contours(binImage, 0.8)
#         #print(f"found {np.shape(self.coords)[0]} trackers")
#         self.contours = np.full((np.shape(self.coords)[0],2),np.nan,dtype=object)

#         for I in range(np.shape(self.coords)[0]):
#             mindist = 100000
#             saveIndex = None
#             for n, contour in enumerate(TempContours):
#                 x_centroid = np.mean(contour[:, 1])
#                 y_centroid = np.mean(contour[:, 0])
#                 dist = geometry.Distance(x_centroid,y_centroid,self.coords[I,0],self.coords[I,1])
#                 if mindist > dist :
#                     mindist = dist
#                     saveIndex = n

#             #ARR_cont = measure.subdivide_polygon(contour, degree=2, preserve_ends=True)
#             ARR_cont = measure.approximate_polygon(TempContours[saveIndex], tolerance=0.5)

#             self.contours[I,:] = [ARR_cont[:,1].tolist(),ARR_cont[:,0].tolist()]

# class Trackers_clean():
#     """
#     Deprecated - May produce very unexpected results
#     Was originally built to make calculations from frame to frame and from tracker to tracker to correct the trajectories from the raw data obtained with Trackers_find.
#     Poorly designed, works frame by frame, complicated and inefficient. Now Use TrackerInstance and it's wrapper that has proven very reliable.
#     """

#     def __init__(self):
#         self.FrameDict = {}
#         self.IndexDict = {}
#         self.frame = 0
#         self.Init = False

#     def AddFrame(self,TrackerObjDataframe):
#         if TrackerObjDataframe.shape[0] >= 2:
#             self.FrameDict.update([(f"{self.frame}" , TrackerObjDataframe)])
#             self.IndexDict.update([(f"{self.frame}" , int(self.frame))])

#         else :
#             temp = TrackerObjDataframe.copy()
#             temp = temp.drop(np.arange(0, TrackerObjDataframe.shape[0],1).tolist())
#             self.FrameDict.update([(f"{self.frame}" , temp)])
#             self.IndexDict.update([(f"{self.frame}" , int(self.frame))])

#         self.frame = self.frame +1
#     def MakeSequence(self):
#         self.Sequence = pd.concat(self.FrameDict.values(), axis=0, keys=self.IndexDict.values(),names = ['Frame','Trkid'])

#     def CleanSequence(self,min_consecutive,maxdist,secondpass=False):

#         prepos = []
#         catch = 0
#         uncatch = 0
#         serie = []

#         self.Sequence.sort_index(level = 0, axis = 0 , inplace = True, ascending = True)

#         for frame, items in self.Sequence.groupby(level=0):

#             if items.shape[0] == 1:
#                 self.Sequence.drop(labels = frame, axis = 0, level = 0, inplace=True)
#                 catch = 0
#                 uncatch = uncatch + 1
#                 prepos = []

#             elif items.shape[0] == 2:

#                 pos = [   mean( [ self.Sequence.loc[frame,0]["coords"][0],self.Sequence.loc[frame,1]["coords"][0] ] ) , mean( [ self.Sequence.loc[frame,0]["coords"][1],self.Sequence.loc[frame,1]["coords"][1] ] ) ]

#                 if len(prepos) :

#                     if geometry.Distance(prepos[0], prepos[1], pos[0], pos[1]) < maxdist :
#                         catch = catch + 1
#                         serie.append(frame)

#                     else :
#                         if catch > 0 and catch < min_consecutive :
#                             for delete in serie:
#                                 self.Sequence.drop(labels = delete, axis = 0, level = 0, inplace=True)
#                             catch = 1
#                             serie = [frame]
#                         else :
#                             catch = 1
#                             serie = [frame]
#                 else :
#                     catch = catch + 1
#                     serie.append(frame)

#                 prepos = pos

#             elif items.shape[0] > 2:

#                 if catch > min_consecutive :

#                     dists = []
#                     locs = []
#                     for index in range(items.shape[0]):
#                         for index2 in range(items.shape[0]):
#                             if index >= index2 :
#                                 continue
#                             pos = [   mean( [ self.Sequence.loc[frame,index]["coords"][0],self.Sequence.loc[frame,index2]["coords"][0] ] ) , mean( [ self.Sequence.loc[frame,index]["coords"][1],self.Sequence.loc[frame,index2]["coords"][1] ] ) ]
#                             dists.append(geometry.Distance(prepos[0], prepos[1], pos[0], pos[1]))
#                             locs.append([index,index2])

#                     select = np.argmin(np.asarray(dists))

#                     for index in range(items.shape[0]):
#                         if index not in locs[select]:
#                             self.Sequence.drop((frame,index), inplace=True)

#                     newmultindex = self.Sequence.loc[frame]

#                     i = 0
#                     for index, row in newmultindex.iterrows():

#                         self.Sequence.drop((frame,index), inplace=True)

#                         self.Sequence.loc[(frame,i), ['coords','orientation','area','indexs']] = row.tolist()
#                         i = i + 1

#                     pos = [   mean( [ self.Sequence.loc[frame,0]["coords"][0], self.Sequence.loc[frame,1]["coords"][0] ] ) , mean( [ self.Sequence.loc[frame,0]["coords"][1], self.Sequence.loc[frame,1]["coords"][1] ] ) ]
#                     if len(prepos) :

#                         if geometry.Distance(prepos[0], prepos[1], pos[0], pos[1]) < maxdist :
#                             catch = catch + 1
#                             serie.append(frame)

#                         else :
#                             if catch > 0 and catch < min_consecutive :
#                                 for delete in serie:
#                                     self.Sequence.drop(labels = delete, axis = 0, level = 0, inplace=True)
#                                 catch = 1
#                                 serie = [frame]
#                             else :
#                                 catch = 1
#                                 serie = [frame]
#                     else :
#                         catch = catch + 1
#                         serie.append(frame)

#                     prepos = pos

#         self.Sequence.sort_index(level = 0, axis = 0 , inplace = True, ascending = True)


#         if secondpass :

#             prepos = []
#             catch = 0
#             serie = []


#             FramesValues = np.flip( np.unique( self.Sequence.index.get_level_values(0).tolist() ) , axis = 0 ).tolist()

#             for frame in FramesValues:
#                 items = self.Sequence.loc[frame,:]

#                 if items.shape[0] == 1:
#                     self.Sequence.drop(labels = frame, axis = 0, level = 0, inplace=True)
#                     catch = 0
#                     uncatch = uncatch + 1
#                     prepos = []

#                 elif items.shape[0] == 2:

#                     pos = [   mean( [ self.Sequence.loc[frame,0]["coords"][0],self.Sequence.loc[frame,1]["coords"][0] ] ) , mean( [ self.Sequence.loc[frame,0]["coords"][1],self.Sequence.loc[frame,1]["coords"][1] ] ) ]

#                     if len(prepos) :

#                         if geometry.Distance(prepos[0], prepos[1], pos[0], pos[1]) < maxdist :
#                             catch = catch + 1
#                             serie.append(frame)

#                         else :
#                             if catch > 0 and catch < min_consecutive :
#                                 for delete in serie:
#                                     self.Sequence.drop(labels = delete, axis = 0, level = 0, inplace=True)
#                                 catch = 1
#                                 serie = [frame]
#                             else :
#                                 catch = 1
#                                 serie = [frame]
#                     else :
#                         catch = catch + 1
#                         serie.append(frame)

#                     prepos = pos


#                 elif items.shape[0] > 2:

#                     if catch > min_consecutive :

#                         dists = []
#                         locs = []
#                         for index in range(items.shape[0]):
#                             for index2 in range(items.shape[0]):
#                                 if index >= index2 :
#                                     continue
#                                 pos = [   mean( [ self.Sequence.loc[frame,index]["coords"][0],self.Sequence.loc[frame,index2]["coords"][0] ] ) , mean( [ self.Sequence.loc[frame,index]["coords"][1],self.Sequence.loc[frame,index2]["coords"][1] ] ) ]
#                                 dists.append(geometry.Distance(prepos[0], prepos[1], pos[0], pos[1]))
#                                 locs.append([index,index2])

#                         select = np.argmin(np.asarray(dists))

#                         for index in range(items.shape[0]):
#                             if index not in locs[select]:
#                                 self.Sequence.drop((frame,index), inplace=True)

#                         newmultindex = self.Sequence.loc[frame]

#                         i = 0
#                         for index, row in newmultindex.iterrows():

#                             self.Sequence.drop((frame,index), inplace=True)

#                             self.Sequence.loc[(frame,i), ['coords','orientation','area','indexs']] = row.tolist()
#                             i = i + 1

#                         pos = [   mean( [ self.Sequence.loc[frame,0]["coords"][0], self.Sequence.loc[frame,1]["coords"][0] ] ) , mean( [ self.Sequence.loc[frame,0]["coords"][1], self.Sequence.loc[frame,1]["coords"][1] ] ) ]

#                         if len(prepos) :

#                             if geometry.Distance(prepos[0], prepos[1], pos[0], pos[1]) < maxdist :
#                                 catch = catch + 1
#                                 serie.append(frame)

#                             else :
#                                 if catch > 0 and catch < min_consecutive :
#                                     for delete in serie:
#                                         self.Sequence.drop(labels = delete, axis = 0, level = 0, inplace=True)
#                                     catch = 1
#                                     serie = [frame]
#                                 else :
#                                     catch = 1
#                                     serie = [frame]
#                         else :
#                             catch = catch + 1
#                             serie.append(frame)

#                         prepos = pos

#         self.Sequence.sort_index(level = 0, axis = 0 , inplace = True, ascending = True)



#     def MakeMerge(self):
#         self.FrameDict = {}
#         lastpos = []
#         lastframe = None
#         capture = 0
#         lastspeed = None

#         #Iter = np.flip( np.unique( self.Sequence.index.get_level_values(0).tolist() ) , axis = 0 ).tolist()

#         for frame, items in self.Sequence.groupby(level=0):

#             Xmean = mean( [ self.Sequence.loc[frame,0]["coords"][0], self.Sequence.loc[frame,1]["coords"][0] ] )
#             Ymean = mean( [ self.Sequence.loc[frame,0]["coords"][1], self.Sequence.loc[frame,1]["coords"][1] ] )

#             if capture == 0 :
#                 speed = np.nan
#                 accel = np.nan
#             elif capture == 1 :
#                 speed = geometry.Distance( lastpos[0], lastpos[1], Xmean, Ymean )
#                 accel = np.nan
#                 lastspeed = speed
#             elif capture >= 2:
#                 speed = geometry.Distance( lastpos[0], lastpos[1], Xmean, Ymean )
#                 accel = speed - lastspeed
#                 lastspeed = speed


#             if lastframe is not None :
#                 if frame == lastframe + 1:
#                     capture = capture + 1
#                 else :
#                     capture = 1

#             lastframe = frame
#             lastpos = [Xmean,Ymean]

#             a = np.array([self.Sequence.loc[frame,0]["coords"][0],  self.Sequence.loc[frame,0]["coords"][1]])
#             b = np.array([self.Sequence.loc[frame,1]["coords"][0],  self.Sequence.loc[frame,1]["coords"][1]])
#             c = np.array([self.Sequence.loc[frame,0]["coords"][0]+10,  self.Sequence.loc[frame,0]["coords"][1]])

#             ba = a - b
#             bc = c - b

#             cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#             angle = np.degrees(np.arccos(cosine_angle))

#             tempdict = {'coordx': Xmean ,'coordy': Ymean , 'angle':angle, 'speed':speed, 'accel':accel}
#             #tempframe = pd.DataFrame( np.asarray( [[Xmean,Ymean] ,angle ,speed ,accel] ) , columns = ['coords', 'angle', 'speed', 'accel'], dtypes = [list,float,float,float] )
#             #tempframe = pd.concat(tempdict.values(), axis=1, keys=tempdict.keys())
#             tempframe = pd.DataFrame(tempdict,index=[0]) #, dtype = ['O','f','f','f']
#             self.FrameDict.update( [ ( int(frame), tempframe ) ] )
#         try :
#             self.Merge = pd.concat(self.FrameDict.values(), axis=0, keys=self.FrameDict.keys(), names = ['Frames']).droplevel(1,axis=0)
#             return True
#         except :
#             print("video with no mouse inside")
#             return False

# def Trackers_Video(path,**kwargs):
#     """
#     Deprecated - May produce very unexpected results
#     Original wrapper around a video path to call Trackers_Clean
#     """

#     if path == "missing"   :
#         return None, None

#     if "output" in kwargs :
#         output = kwargs.get("output")
#         print(output)
#         if os.path.isfile(output):
#             print("File exists")
#             overwrite = kwargs.get("overwrite" ,False)
#             if not overwrite :
#                 return
#     else :
#         output = None

#     thresh = kwargs.get("thresh" , 50)

#     test = kwargs.get("test", False)
#     testval = False

#     claheob = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
#     HandleTOP = cv2.VideoCapture(path ,cv2.IMREAD_GRAYSCALE)
#     NumberOfFrames = int(HandleTOP.get(cv2.CAP_PROP_FRAME_COUNT))
#     bar = pyprind.ProgBar(NumberOfFrames, track_time=True, title=f'Tracking video {path}',bar_char='█')

#     videoTrack = Trackers_clean()
#     for frameid in range(NumberOfFrames):
#         bar.update()
#         _ , TempIMG = HandleTOP.read()
#         TempIMG = np.rot90(TempIMG,1)

#         if test and frameid >= 400:
#             import matplotlib.pyplot as plt

#             image.QuickHist(TempIMG)
#             plt.figure()
#             plt.imshow(TempIMG,cmap = "gray")
#             plt.show()
#             testval = True


#         BinIMG = image.IM_binarize(TempIMG,thresh,clahe = claheob)
#         if frameid == 170:
#             firstFrame = BinIMG

#         if testval :
#             # Trk = Trackers_find(BinIMG,500,2000, debug = True)
#             Trk = Trackers_find(BinIMG,1200,2300, debug = True)
#             testval = input("Good or not")
#             if testval != "OK":
#                 sys.exit()
#             else :
#                 testval = False
#                 test = False
#         else :
#             Trk = Trackers_find(BinIMG,650,1700, debug = False)

#         Trk.RemoveExtremaPos(40 , 200)

#         videoTrack.AddFrame(Trk.items)

#     del Trk

#     videoTrack.MakeSequence()
#     videoTrack.CleanSequence(7,10,True)
#     result = videoTrack.MakeMerge()

#     if result:
#         print(f"saving {output}")
#         if output is not None :
#             videoTrack.Merge.to_pickle(output)
#         else :
#             return videoTrack.Merge, firstFrame
#     else :
#         return None, firstFrame

######################################

class TrackerInstance():

    def __init__(self, identification, main_attributes = None, **kwargs):
        """
        Defines a tracker model instance, with a specific identity, refered by it's id (string), containing a variable number of items in a list : self.attributes
        These attributes are dataframes obtained with Trackers_find and contains  'coords','orientation','area','indexs','surface','bbox','eccentricity','major_axis_length', 'minor_axis_length','image','moments_hu'
        for the blob that was identified by a user as a tracker.

        Args:
            identification (str): Model Name.
            main_attributes (Optional[list]): Attributes list of another model to reload on a fresh instance. Defaults to None.
            **thresh (int): the threshold applied to the given images here and inside the function AddModelVisualize, to binarize them. Must match the threshold set to the TrackerMesh class for better results
            **SDcoeff (int): integer : the coefficient multiplied by the Standard Deviation of all the tracker models areas (SD recalculated each time a new model is added) This doefficient will be used to determine when a comparison is made with the method below, if the area of the item is between mean area - coeef * SD and men area + coeff + SD
            **viz (bool): if true, the initialisation will use the array given by the array kwarg to extract the first tracker and put it in attributes list
            **array (np.ndarray): 2 or 3 dimensions array to extract tracker( 3rd dimension is time )
        """

        self.treshold = kwargs.get('thresh',50)
        self.SDcoeff = kwargs.get("SDcoeff",4)
        self.id = identification
        if main_attributes is not None :
            self.attributes = main_attributes
            self.Compute_Mean()
            self.Compute_SD()
        else :
            self.binimage = []
            self.attributes = []

        if kwargs.get("viz",False):
            self.Add_Model_Visualize(kwargs.get("array"),plot = True)



    def Compute_SD(self):

        area = []
        eccentricity = []
        if len(self.attributes) >= 3 :
            for index in range(0,len(self.attributes)):
                area.append(self.attributes[index]['area'])
                eccentricity.append(self.attributes[index]['eccentricity'])

            self.areaSD = np.std( np.asarray(area)  )
            self.eccentricitySD = np.std( np.asarray(eccentricity)  )
            print(self.areaSD, self.areaSD * self.SDcoeff)

        elif 1 <= len(self.attributes) < 3 :

            self.areaSD = 100
            self.eccentricitySD = 0.15

        else :
            raise ValueError("Can't compute standard deviation with no model data, use Add_Model first")

    def Compute_Mean(self):

        area = []
        eccentricity = []
        if len(self.attributes) >= 3 :
            for index in range(0,len(self.attributes)):

                area.append(self.attributes[index]['area'])
                eccentricity.append(self.attributes[index]['eccentricity'])
            self.area = np.mean( np.asarray(area) )
            self.eccentricity = np.mean( np.asarray(eccentricity) )

        elif 1 <= len(self.attributes) < 3 :
            self.area = self.attributes[0]['area']
            self.eccentricity = self.attributes[0]['eccentricity']

        else :
            raise ValueError("Can't compute mean with no model data, use Add_Model first")

    def Add_Model(self,tracker_attributes,calcSD = True):

        self.attributes.append(tracker_attributes)
        self.Compute_Mean()
        if calcSD :
            self.Compute_SD()

    def Add_Model_Visualize(self,array,**kwargs):

        calcSD = kwargs.get("calcSD", True)

        args = {"mode" : "coords", "title" : f"select {self.id} tracker coordinates"}
        if len(self.attributes) >= 1 :
            trackimg = self.attributes[0]['image'].astype(np.uint8)
            trackimg[trackimg > 0] = 255
            trackimg = np.pad(trackimg,2,"constant",constant_values = 0)
            args.update( {"tracker_img" : trackimg} )
        values = image.VideoDialog(array,**args)

        if type(values) is dict:

            x = values.get('x',None)
            y = values.get('y',None)
            frame = values.get('frame',None)
            returnvalue = values.get('retbool',None)
            trckfound = values.get('trackerfound',None)
            skip = values.get('skip',False)

            if returnvalue == 0 :
                sys.exit()

            if trckfound == False :
                return False

            if skip == True :
                return None

            if len(array.shape) > 2 :
                img = image.IM_binarize(array[:,:,frame],self.treshold)
            else:
                img = image.IM_binarize(array,self.treshold)

            regions = Trackers_find(img,20,5000)

            distances = []
            for index, lines in regions.items.iterrows() :
                distances.append( geometry.Distance(lines['coords'][0], lines['coords'][1],x,y) )
            if len(distances) > 0 :
                selectindex = np.argmin(np.asarray(distances))
                line = regions.items.iloc[selectindex]
                if kwargs.get("plot",False):
                    self.PlotSelectedTracker(line)
                #print(line)
                self.Add_Model(line,calcSD)
                return True
            else :
                raise ValueError("No tracker found at these coordinates")
        else :
            print(values)

            #print("Single image is not yet supported reliably to 3Dwidget, change if necessary")

    def PlotModels(self):

        for line in self.attributes:
            self.PlotSelectedTracker(line)

    def PlotSelectedTracker(self,line):

        plt.figure(figsize = (2,2))
        plt.imshow(line['image'])
        plt.show()

    def AreaFilter(self, line):

        if int( self.area + (self.SDcoeff*self.areaSD)) > line['area'] and int(self.area - (self.SDcoeff*self.areaSD)) < line['area'] :
            return True
        else :
            return False

    def Compare(self, frame_attributes, **kwargs):

        filterArea = kwargs.get("FilterArea", True)
        scores = []

        if filterArea and not self.AreaFilter(frame_attributes) :
            return None

        for index in range(0,len(self.attributes)):
            scores.append( MatchContours( self.attributes[index]['image'] , frame_attributes['image'] ) )

        return np.asarray(scores)

def ExtractContour(binimg) :
    binimg = image.IM_easypad(binimg, 2 ,mode = 'constant')

    if binimg.dtype == np.bool :
        binimg = binimg.astype(np.uint8)
        binimg[binimg > 0] = 255

    _ , TempContours, _ = cv2.findContours(binimg, 2, 1)
    return  TempContours[0]

def MatchContours(binimg1,binimg2):
    cont1 = ExtractContour(binimg1)
    cont2 = ExtractContour(binimg2)
    ret = cv2.matchShapes(cont1,cont2,1,0.0)
    #print(f"Matching value is : {ret}")
    return ret


class TrackerMesh():

    def __init__(self,models,array,**kwargs):

        #self.engine = network.OpenSQL()
        #self.root = network.find_favoritesRootFolder(**kwargs)

        self.AUTOMODE = kwargs.get("auto",False)
        self.STOP_AND_SAVE = False

        if 'thresh' in kwargs.keys() :
            self.threshold = kwargs.get('thresh')
            del kwargs['thresh']
        else :
            retval = image.VideoDialog(array,mode = "binarise")
            self.threshold = retval['theshold']

        if 'SDcoeff' in kwargs.keys() :
            self.SDcoeff = kwargs.get('SDcoeff')
            del kwargs['SDcoeff']
        else :
            self.SDcoeff = 4

        self.criterion = kwargs.get("criterion", 0.06)

        self.ModelInstances = {}

        if type(models) is list :
            for name in models :
                self.ModelInstances.update({ name :  TrackerInstance(name, thresh = self.threshold, array = array, viz = True, SDcoeff = self.SDcoeff,**kwargs) })

        elif type(models) is dict :
            self.ModelInstances = models

        else :
            raise ValueError("Cannot use this type of data as models or models names. Accepted types : dictionnary of list or list of strings")

        self.MakeResultStruct(array,**kwargs)

        self.CompareResultSize()

        self.EnsureMinimumModelsAvailable()

        self.Completion = False

    def Set_SDcoeff(self,SDcoeff):
        self.SDcoeff = SDcoeff
        for name in self.ModelInstances.keys():
            self.ModelInstances[name].SDcoeff = self.SDcoeff

    def Set_Automode(self,auto):
        self.AUTOMODE = auto
        self.STOP_AND_SAVE = False

    def MakeResultStruct(self,array,**kwargs):

        self.array = array

        results =  kwargs.get('results',None)
        statiknames = kwargs.get('statiknames',None)

        self.Results = {}
        if results is not None:
            self.Results = results
            self.CompareResultSize()
        else :
            for name in self.ModelInstances.keys():
                self.Results.update({ name : [None] * np.shape(self.array)[2]})
            self.Results.update({ "statiks" : {}})
            if statiknames is not None :
                for name in statiknames :
                    self.Results["statiks"].update({statiknames :  [None] * np.shape(self.array)[2]})

        self.Content = [None] * np.shape(self.array)[2]

        self.Scores = {}
        for name in self.ModelInstances.keys():
            self.Scores.update( { name : [None] * np.shape(self.array)[2] } )

    def LoadContent(self,Content):

        if len(Content) == self.array.shape[2]:
            self.Content = Content

    def CompareResultSize(self):

        if len(self.Results[list(self.ModelInstances.keys())[0]]) != np.shape(self.array)[2] :
            raise ValueError("Results are not of same length than video array. Cannot proceed")

    def LoadResults(self, Results):

        self.Results = Results

    def LoadNewVideo(self, array,**kwargs):

        self.array = array
        self.MakeResultStruct(**kwargs)
        self.CompareResultSize()

    def ExtractShapes(self):

        sizes = []

        for trackername in self.ModelInstances.keys():
            minsize = self.ModelInstances[trackername].area - (self.ModelInstances[trackername].SDcoeff*self.ModelInstances[trackername].areaSD)
            maxsize = self.ModelInstances[trackername].area + (self.ModelInstances[trackername].SDcoeff*self.ModelInstances[trackername].areaSD)
            sizes.append([minsize,maxsize])

        sizes = np.asarray(sizes).reshape((-1, 2))
        minsize = np.min(sizes[:,0])
        maxsize = np.max(sizes[:,1])

        bar = pyprind.ProgBar( np.shape(self.array)[2],bar_char='░', title=f'Extracting shapes from video of length {np.shape(self.array)[2]}' )

        self.Content = [None] * np.shape(self.array)[2]

        for frame_index in range (0, np.shape(self.array)[2] ):
            bar.update()
            if self.Content[frame_index] is None :
                binimg = image.IM_binarize(self.array[:,:,frame_index],self.threshold)
                shapes = Trackers_find(binimg,minsize,maxsize)
                self.Content[frame_index] = shapes

    def CompareShape(self,shapes,model,**kwargs):

        criterion = self.criterion

        score = []
        scorindex = []
        savebadscores = []
        noShapeHasArea = True

        for index, line in shapes.items.iterrows() :
            compscor = model.Compare(line)
            if compscor is not None :
                noShapeHasArea = False
                minscore = np.min(compscor)
                if  minscore < criterion :
                    score.append(minscore)
                    scorindex.append(index)
                else:
                    savebadscores.append(np.min(compscor))
        if noShapeHasArea :
            return None
        if len(score) > 0:
            selectedindex =  scorindex[np.argmin(np.asarray(score))]
            return [shapes.items.at[selectedindex,'coords'][0] , shapes.items.at[selectedindex,'coords'][1]]
        else :
            return [np.nan, np.nan, np.min(savebadscores)]

    def CureLeastConfident(self,**kwargs):

        #addm = kwargs.get("addmodel",True)
        infos = kwargs.get("infos",True)

        logger = logging.getLogger("TrackerMesh.CureLeastConfident")
        logger.setLevel(logging.INFO)

        for name in self.ModelInstances.keys():

            if infos :
                tempres = np.asarray(self.Results[name]).reshape((-1, 2))
                unique, counts = np.unique(tempres[:,0], return_counts=True)
                outp = dict(zip(unique, counts))
                val1 = outp.get(-1,0)
                val = ( ( tempres.shape[0] - val1 ) / tempres.shape[0] ) * 100

                print(f"{val:.1f}% complete for tracker {name}")

            if val > 80:
                self.UnsupervisedCure(name,**kwargs)

            if not all(v is None for v in self.Scores[name]):
                LeastScoreFrameIndex = np.nanargmax(np.asarray(self.Scores[name],dtype=np.float32))
                if infos :
                    print(f"Tracker {name}, frame causing issue : {LeastScoreFrameIndex}")
                    self.PlotResults(frame = LeastScoreFrameIndex)

                if self.AUTOMODE :
                    self.STOP_AND_SAVE = True
                    logger.warning(f"WARNING : Stopped at Tracker#{name}, frame causing issue : frame#{LeastScoreFrameIndex}")
                    continue
                found = self.ModelInstances[name].Add_Model_Visualize(self.array[:,:,LeastScoreFrameIndex] ,plot = True)

                if found == False :
                    returnval = self.DetectContingence(self.Results[name])
                    sliceindexes = self.CheckNanPadContingency(self.Results[name],LeastScoreFrameIndex,*returnval)
                    if sliceindexes is None :
                        self.Results[name][LeastScoreFrameIndex] = [np.nan,np.nan]
                        self.Scores[name][LeastScoreFrameIndex] = None
                    else :
                        for i in range(sliceindexes[0],sliceindexes[1]):
                            self.Results[name][i] = [np.nan,np.nan]
                            self.Scores[name][i] = None

    def UnsupervisedCure(self,trackername,**kwargs):

        logger = logging.getLogger("TrackerMesh.UnsupervisedCure")
        logger.setLevel(logging.INFO)

        resar = np.asarray(self.Results[trackername]).reshape(-1,2)

        dists = []
        for frame_index in range(1,resar.shape[0]):
            dists.append( geometry.Distance( resar[frame_index,0] , resar[frame_index,1]  , resar[frame_index-1,0] , resar[frame_index-1,1] ) )

        binlist = signal.BinarizeList(dists,10)
        slices, values = signal.DetectContingence(binlist)

        maxf = 0 # maxi as max frame size index, to reference the slice with largest size
        maxi = None # maxf as max frame size, updated each tome a largest slice is found
        for i in range(len(slices)):
            if values[i]==0: #only selecting slices of "found" coordinates
                if slices[i][1] - slices[i][0] > maxf :
                    maxf = slices[i][1] - slices[i][0]
                    maxi = i

        #maxtrusted_distance = kwargs.get("maxdist",20)
        #maxtrusted_score = kwargs.get("maxscore",0.1)

        if slices[maxi][1]-slices[maxi][0] < 40: #parameter here to modify if one wants to adjust the minimum trusted size on a single slice to initialize autocompletion from it's borders
            self.STOP_AND_SAVE = True
            logger.error(f"ERROR : slice at {slices[maxi][0]} to {slices[maxi][1]} : the longest slice is shorter than minimum value to initiate autocompletion")
            return

        for DIRECTION in [1,-1]:
            if DIRECTION == 1 :
                start = slices[maxi][1]-1
                stop = len(self.Results[trackername])
            if DIRECTION == -1 :
                start = slices[maxi][0]+1
                stop = 0

            for frame_index in range(start, stop, DIRECTION):

                if not math.isnan(self.Results[trackername][frame_index][0]) :

                    shapes = self.Content[frame_index]

                    coords = []
                    avgscore = []

                    for index, lines in shapes.items.iterrows() :

                        if self.ModelInstances[trackername].AreaFilter(lines) :

                            diss = geometry.Distance(lines['coords'][0], lines['coords'][1], self.Results[trackername][frame_index-DIRECTION][0], self.Results[trackername][frame_index-DIRECTION][1] )
                            compscor = self.ModelInstances[trackername].Compare( lines )
                            if compscor is not None :
                                avgscore.append((np.min(compscor) * 50) + diss ** 3)
                            else :
                                avgscore.append(diss ** 3 + 200)
                            coords.append( [lines['coords'][0], lines['coords'][1]] )

                    if len(avgscore) > 0 :
                        chosenindex = np.argmin(avgscore)
                        #print(len(coords),len(avgscore),coords,avgscore,chosenindex)
                        #print(avgscore[chosenindex])

                        if geometry.Distance( coords[chosenindex][0], coords[chosenindex][1], self.Results[trackername][frame_index-DIRECTION][0], self.Results[trackername][frame_index-DIRECTION][1] ) < 20 : #maxspeed ( distance from a point to the next frame to frame)
                            self.Results[trackername][frame_index] = coords[chosenindex]
                            self.Scores[trackername][frame_index] = None
                        else :
                            self.Results[trackername][frame_index] = [np.nan,np.nan]
                            self.Scores[trackername][frame_index] = None
                            break
                    else :
                        self.Results[trackername][frame_index] = [np.nan,np.nan]
                        self.Scores[trackername][frame_index] = None
                        break




    def Track(self,**kwargs):

        if all(v is None for v in self.Content) or kwargs.get("force",False):
            self.ExtractShapes()

        bar = pyprind.ProgBar(np.shape(self.array)[2],bar_char='░', title=f'Shape matching for video of length {np.shape(self.array)[2]}')

        for frame_index in range (0, np.shape(self.array)[2] ):
            bar.update()
            for trackername in self.Results.keys():
                if trackername != "statiks":
                    if self.Results[trackername][frame_index] is None :
                        flag = True
                    else :
                        if self.Results[trackername][frame_index][0] == -1:
                            flag = True
                        else :
                            flag = False
                    if flag :
                        resdata = self.CompareShape(self.Content[frame_index] , self.ModelInstances[trackername],**kwargs)
                        if resdata is None :
                            self.Results[trackername][frame_index] = [np.nan,np.nan] # return none if no area is in the selected range : mouse not in video
                            self.Scores[trackername][frame_index] = None
                        else :
                            if np.isnan(resdata[0]): # return [np.nan,np.nan] if no match with sufficient score. To be retried with new models added
                                self.Results[trackername][frame_index] = [-1,-1]
                                self.Scores[trackername][frame_index] = resdata[2]
                            else :
                                self.Results[trackername][frame_index] = resdata
                                self.Scores[trackername][frame_index] = None

        for trackername in self.ModelInstances.keys():

            self.CheckUniqueBlobAttribution(trackername)
            returnval = self.DetectContingence(self.Results[trackername])
            slices, values = self.CheckIsolates(*returnval)
            for i in range(len(slices)):
                if values[i] == 1 or values[i] == -1 :
                    for u in range(slices[i][0],slices[i][1]):
                        self.Results[trackername][u] = [np.nan,np.nan]
                        self.Scores[trackername][u]  = None


    def EnsureMinimumModelsAvailable(self):
        for trackername in self.ModelInstances.keys():
            while True :
                if len(self.ModelInstances[trackername].attributes) < 3 :
                    self.ModelInstances[trackername].Add_Model_Visualize(self.array ,plot = False)
                else :
                    break

    def DetectContingence(self,List):
        Listcopy = List.copy()
        for i in range(len(Listcopy)):
            if Listcopy[i] is None:
                continue
            else :
                if Listcopy[i][0] == -1 :
                    Listcopy[i] = -1
                elif np.isnan(Listcopy[i][0]):
                    Listcopy[i] = np.nan
                else :
                    Listcopy[i] = 1

        ranges = [i+1  for i in range(len(Listcopy[1:])) if not ( ( Listcopy[i] == Listcopy[i+1] ) or ( math.isnan(Listcopy[i]) and math.isnan(Listcopy[i+1]) ) ) ]

        #ranges = [ i+1  for i in range(len(Listcopy[1:])) if Listcopy[i] != Listcopy[i+1] ]
        ranges.append(len(Listcopy))
        ranges.insert(0, 0)

        slices = []
        values = []
        for i in range(len(ranges)-1):
            slices.append([ranges[i], ranges[i+ 1]])
            if Listcopy[ranges[i]] is None :
                values.append(None)
            else :
                values.append(Listcopy[ranges[i]])

        return slices, values

    def CheckNanPadContingency(self,Listcopy,MainIndex,slices,values):

        if MainIndex > len(Listcopy)-1:
            raise ValueError(f"Cannot supply an index greater than the list size for contingency detection, index { MainIndex }: max possible index : { len(Listcopy)-1 }")
        #print(slices)
        #print(values)
        #print(len(slices))
        #print(len(values))

        for i in range(len(slices)):
            if slices[i][0] <= MainIndex < slices[i][1]:
                slicendex = i
                break

        if slicendex >= len(slices)-1 or slicendex <= 0 :
            return None

        if slices[slicendex + 1][1] == len(Listcopy) and slices[slicendex - 1][0] == 0 :
            return None

        #print(values[slicendex - 1] , values[slicendex + 1])
        if ( slices[slicendex + 1][1] == len(Listcopy) or slices[slicendex - 1][0] == 0 ):#"flanking a border"
            if not values[slicendex - 1] is None and not values[slicendex + 1] is None :
                if np.isnan(values[slicendex - 1]) and np.isnan(values[slicendex + 1]):

            #"flanking a border"                                                                   #"flanking a NanPad"
            #print(values[slicendex - 1], values[slicendex + 1])
                    return slices[slicendex]
            #print(values[slicendex])
        else :
            return None

    def CheckUniqueBlobAttribution(self,trackername):

        for frame_index in range (0, np.shape(self.array)[2] ):
            count = 0
            coords = []
            if self.Results[trackername][frame_index] is not None :
                if self.Results[trackername][frame_index][0] == -1 :
                    for index, lines in self.Content[frame_index].items.iterrows() :
                        if self.ModelInstances[trackername].AreaFilter(lines) :
                            if count == 0:
                                count = count + 1
                                coords = lines["coords"]
                            else :
                                break
                    if count == 1 :
                        for secondname in self.ModelInstances.keys():
                            if geometry.Distance( self.Results[secondname][frame_index][0], self.Results[secondname][frame_index][1], coords[0], coords[1] ) < 2:
                                self.Results[trackername][frame_index] = [np.nan, np.nan]
                                self.Scores[trackername][frame_index] = None
                    else :
                        self.Results[trackername][frame_index] = [np.nan, np.nan]
                        self.Scores[trackername][frame_index] = None
                        continue


    def CheckIsolates(self,slices,values):

        isolateslices = []
        isolatevalues = []
        for i in range(1,len(slices)-1):
            if not values[i - 1] is None and not values[i + 1] is None :
                if np.isnan(values[i - 1]) and np.isnan(values[i + 1]) and slices[i][1]-slices[i][0] < 20 : #change parameter 20 here to set maximum coordinates island between nas to automatically remove
                    isolateslices.append(slices[i])
                    if values[i] is None :
                        isolatevalues.append(None)
                    else :
                        isolatevalues.append(values[i])
        return isolateslices, isolatevalues

     ##TODO reshape deshape coordinates data : np.asarray(coordf).reshape((-1, 2))

    def CureLoop(self,**kwargs):

        logger = logging.getLogger("TrackerMesh.CureLoop")
        logger.setLevel(logging.INFO)

        from IPython.display import clear_output
        if "criterion" in kwargs :
            self.criterion = kwargs.get("criterion")

        while True :

            self.Track(criterion = self.criterion)
            clear_output(wait=False)
            self.CureLeastConfident(**kwargs)
            self.PlotResults(**kwargs)
            if self.STOP_AND_SAVE :
                print("unfinished, saving to cure later")
                logger.warning(f"WARNING : Unfinished tracking /!\/!\/!\  /!\/!\/!\ ")
                break

            if self.CheckTracking():
                print("tracking finished")
                logger.info(f"INFO : Completed")
                break

    def CheckTracking(self):

        for trackername in self.ModelInstances.keys():
            for frame_index in range (0, np.shape(self.array)[2] ):
                if self.Results[trackername][frame_index] is not None :
                    if self.Results[trackername][frame_index][0] == -1 :
                        self.Completion = False
                        return self.Completion
                else :
                    self.Completion = False
                    return self.Completion
        self.Completion = True
        return self.Completion

    def PlotResults(self,**kwargs):

        figsize = kwargs.get("figsize", (10,10))
        frame = kwargs.get("frame", 0)

        plt.figure(figsize=figsize)
        if len(self.array.shape) > 2 :
            plt.imshow(self.array[:,:,frame],cmap = "gray")
        else:
            plt.imshow(self.array,cmap = "gray")
        for trackername in self.ModelInstances.keys():
            currentres = np.asarray(self.Results[trackername]).reshape((-1, 2))
            plt.plot(currentres[:,0],currentres[:,1])
        plt.xlim([0,620])
        plt.ylim([1024,0])
        plt.show()

def SaveTrackerMesh(outputpath,MeshInstance,**kwargs):
    save = {"models" : MeshInstance.ModelInstances, "results" : MeshInstance.Results, "threshold": MeshInstance.threshold, "content" : MeshInstance.Content , "criterion" : MeshInstance.criterion, "completion" : MeshInstance.Completion, "SDcoeff" : MeshInstance.SDcoeff}

    memspace = kwargs.get("save_space",False)
    if memspace :
        save.pop("models")
    print(f"Saving to {outputpath}")
    with open(outputpath,'wb') as pickleHandle:
        _ = pickle.dump(save, pickleHandle)


def LoadTrackerMesh(inputpath,**kwargs):

    if type(inputpath) is str :
        with open(inputpath,'rb') as pickleHandle:
            loads = CustomUnpickler(pickleHandle).load()
    else :
        MeshInstance = inputpath
        loads = {"models" : MeshInstance.ModelInstances, "results" : MeshInstance.Results, "threshold": MeshInstance.threshold, "content" : MeshInstance.Content , "criterion" : MeshInstance.criterion, "completion" : MeshInstance.Completion, "SDcoeff" : MeshInstance.SDcoeff}

    result = {"loads" : loads}
    if loads.get("results", None) is not None :

        result.update({"coordinates" : loads["results"]})
    else :
        return None

    loadtype = kwargs.get("loadtype", "results")

    if loadtype == "results" :
        return loads["results"]

    elif loadtype == "toinstance" :
        array = kwargs.get("array",None)
        if array is None :
            raise ValueError(f"you must specify an array kwarg when using loadtype = 'toinstance'")
        MeshInstance = TrackerMesh( loads["models"] , array , thresh = loads["threshold"], SDcoeff = loads["SDcoeff"], criterion = loads["criterion"] )
        return MeshInstance

    elif loadtype == "reload" :
        array = kwargs.get("array",None)
        if array is None :
            raise ValueError(f"you must specify an array kwarg when using loadtype = 'reload'")
        MeshInstance = TrackerMesh( loads["models"] , array , thresh = loads["threshold"], SDcoeff = loads["SDcoeff"], criterion = loads["criterion"] )
        MeshInstance.LoadContent(loads["content"])
        MeshInstance.LoadResults(loads["results"])
        if loads["completion"] :
            result.update({"completion" : True})
        else :
            result.update({"completion" : False})
        return MeshInstance

    else :
        raise ValueError(f"loadtype not understood : {loadtype} - use either 'results', 'toinstance' or 'reload'")

def ExtractModelsFromPickle(inputpath):

    with open(inputpath,'rb') as pickleHandle:
        loads = CustomUnpickler(pickleHandle).load()

    models = loads["models"]

    return models

def ExtractArgsFromPickle(inputpath):

    with open(inputpath,'rb') as pickleHandle:
        loads = CustomUnpickler(pickleHandle).load()
    args = {"thresh": loads["threshold"], "criterion" : loads["criterion"], "SDcoeff" : loads["SDcoeff"]}
    return args

def ExtractArgsFromInstance(MeshInstance):

    args = {"thresh": MeshInstance.threshold, "criterion" : MeshInstance.criterion, "SDcoeff" : MeshInstance.SDcoeff}
    return args

def TestSessionIsExisting(session,trial,steplist):

    for i in range(len(steplist["session"])):
        if steplist["session"][i] == session and steplist["trial"][i] == trial :
            return False
    return True

def Main_Wrapper_ShapeTracking(target,target_meaning,method,**kwargs):
    import logging
    from datetime import datetime

    now = datetime.now()

    fileanalysis_suffix = "ShapeMatch_trajectories"
    version = kwargs.get('version',"V1")

    if target_meaning == "training_set" :
        sessionlist = database_IO.TrainingSetSessions(target)
    elif target_meaning == "sessions" :
        if type(target) is not list :
            target = [target]
        sessionlist = target
    else :
        raise ValueError("target signification not understood, use 'session' or 'training_set'")

    if target_meaning == "training_set" :
        modelfile_name = f'{target_meaning}${target}#{fileanalysis_suffix}#{version}.pickle'
        logsFilename = now.strftime(f"LOGS#{fileanalysis_suffix}#{version}#{target_meaning}${target}#%y%m%d_%H-%M-%S.log")
    else :
        modelfile_name = f'{target_meaning}${sessionlist[0]}${sessionlist[-1]}#{fileanalysis_suffix}#{version}.pickle'
        logsFilename = now.strftime(f"LOGS#{fileanalysis_suffix}#{version}#{target_meaning}${sessionlist[0]}${sessionlist[-1]}#%y%m%d_%H-%M-%S.log")


    logsBasename = r"\\157.136.60.11\EqShulz\Timothe\DataProcessing\Logs"
    logging.basicConfig(filename=os.path.join(logsBasename,logsFilename),level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt = '%d/%m/%Y %H:%M:%S %p --')
    logger = logging.getLogger("Main_Wrapper_ShapeTracking")
    logger.setLevel(logging.INFO)
    logging.info("NEW PROGRAMM CALL AT DATE :" + now.strftime("%Y%m%d AND HOUR %H:%M:%S") + "\n")


    if method == "from_models" :

        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        rootfolder = os.path.join(network.find_favoritesRootFolder(),"DataProcessing\TrackerModels")
        MostRecentFile = filedialog.askopenfilename(initialdir=rootfolder,title=f"Please select a file for {target} {target_meaning}",filetypes = (("pickle","*.pickle"),("all files","*.*")) )
        #file_path = filedialog.askdirectory(parent=root,initialdir=rootfolder,title=f"Please select a directory containing {fileanalysis_suffix} .pickles")

        #MostRecentFile = strings.GetMostRecentFile( strings.GetVersionnedPickles( file_path,fileanalysis_suffix,version ) )
        models = ExtractModelsFromPickle(MostRecentFile)
        args = ExtractArgsFromPickle(MostRecentFile)
        args.update(kwargs)

    if method == "new" :

        import random
        steplist = {"session" : [],"trial": [],"frame" : []}

        for steps in range(kwargs.get("additions",15)):
            while True :
                session = random.choice(sessionlist)
                SessionDataFrame = database_IO.SessionDataframe(session,castErrors = False,method = "new")
                if SessionDataFrame is not None:
                    exclude_list = strings.LoadConfigFile(SessionDataFrame.dirs["batch"],"Trajectories","Excluded_sessions")
                    if session not in exclude_list:
                        break
            while True :
                trial = random.randint(0,SessionDataFrame.shape[0])

                if SessionDataFrame.at[int(trial),"BEHpath"] != "missing" and TestSessionIsExisting(session,trial,steplist):
                    break

            if len(steplist["session"]) == 0:
                array = image.GetImage(SessionDataFrame.at[int(trial),"BEHpath"], rot = 1, all = True)
                temp_mesh = TrackerMesh( kwargs.get("modelnames",['front','back']) ,array, **kwargs.get("args",{"criterion" : 0.07, "thresh" : 60, "SDcoeff" : 15}))
                models = temp_mesh.ModelInstances
                args = ExtractArgsFromInstance(temp_mesh)
            else :
                for name in models.keys():
                    framelist = []
                    while True :
                        #frame = random.randint(10,image.GetVideoLen(SessionDataFrame.at[int(trial),"BEHpath"]) - 10)
                        frame = random.choice([i for i in range(10,image.GetVideoLen(SessionDataFrame.at[int(trial),"BEHpath"])-10) if i not in framelist])
                        array = image.GetImage(SessionDataFrame.at[int(trial),"BEHpath"], rot = 1,pos = frame, span = 10)
                        retval = models[name].Add_Model_Visualize( array, plot = True )
                        if retval is None :
                            break
                        else :
                            if retval is False :
                                framelist = framelist + list(range(frame-10,frame+10))
                            else :
                                break

            steplist["session"].append(session)
            steplist["trial"].append(trial)

        temp_mesh = TrackerMesh( models ,array, **args)

        SaveTrackerMesh(os.path.join(network.find_favoritesRootFolder(), 'DataProcessing\TrackerModels', modelfile_name ),temp_mesh,save_space = False)

    kwargs.pop("version")
    for session in sessionlist :

        SessionDataFrame = database_IO.SessionDataframe(session,method = "new")
        logger.info(f"INFO : Session s#{session} starting : location of videos : {SessionDataFrame.at[0,'PIPELINEpath']}\n")
        exclude_list = strings.LoadConfigFile(SessionDataFrame.dirs["batch"],"Trajectories","Excluded_sessions")
        if session in exclude_list:
            continue
        for trial, I in enumerate(SessionDataFrame["BEHpath"]):
            if SessionDataFrame.at[int(trial),"BEHpath"] != "missing" :
                output = os.path.join(SessionDataFrame.at[int(trial),"PIPELINEpath"],"Trajectories" , f'{SessionDataFrame.at[int(trial),"FullName"]}#{fileanalysis_suffix}#{version}.pickle')
                if not os.path.isfile(output) or kwargs.get("overwrite",False):
                    try :
                        logger = logging.getLogger("Main_Wrapper_ShapeTracking")
                        array = image.GetImage(SessionDataFrame.at[int(trial),"BEHpath"], rot = 1, all = True)
                        newmesh = TrackerMesh( models ,array, **args)
                            #another way to reload tracker - more versatile but unnecessary here : #newmesh = LoadTrackerMesh(oldmesh, loadtype = "toinstance", array = array)
                        logger.info(f"INFO : Start for video : s#{session} t#{trial}")

                        newmesh.Set_Automode(True)
                        newmesh.CureLoop()
                        SaveTrackerMesh(output,newmesh,save_space = True)
                        logger.info(f"INFO : End\n")
                        models = newmesh.ModelInstances
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

                        Errorstring = f"ERROR : {e} - {exc_type} - in {fname} at line {exc_tb.tb_lineno} : {exc_obj}"

                        save = {"error" : True, "error_detail" : Errorstring}
                        with open(output,'wb') as pickleHandle:
                            _ = pickle.dump(save, pickleHandle)

                        print(e, exc_type, fname, exc_tb.tb_lineno)
                        logger.error(Errorstring + '\n')
        _ = database_IO.GatherToDataframe(SessionDataFrame, fileanalysis_suffix , version, save = True, **kwargs)
        logger.info(f"INFO : Session s#{session} complete\n")

######################################


def GetTrajectoryResults(mesh, names = ["front","back"],**kwargs):

    def DetectContingence(List):

        Listcopy = List.copy()
        for i in range(len(Listcopy)):
            if Listcopy[i] is None:
                continue
            else :
                if Listcopy[i][0] == -1 :
                    Listcopy[i] = -1
                elif np.isnan(Listcopy[i][0]):
                    Listcopy[i] = np.nan
                else :
                    Listcopy[i] = 1

        ranges = [i+1  for i in range(len(Listcopy[1:])) if not ( ( Listcopy[i] == Listcopy[i+1] ) or ( math.isnan(Listcopy[i]) and math.isnan(Listcopy[i+1]) ) ) ]

        #ranges = [ i+1  for i in range(len(Listcopy[1:])) if Listcopy[i] != Listcopy[i+1] ]
        ranges.append(len(Listcopy))
        ranges.insert(0, 0)

        slices = []
        values = []
        for i in range(len(ranges)-1):
            slices.append([ranges[i], ranges[i+ 1]])
            if Listcopy[ranges[i]] is None :
                values.append(None)
            else :
                values.append(Listcopy[ranges[i]])

        return slices, values

    error = kwargs.get('castErrors',False)

    trajes = {}
    for name in names :

        trajes.update( { name : np.asarray(mesh[name]).reshape(-1,2).astype(float) } )

        for i in range(trajes[name].shape[0]):

            if trajes[name][i,0] == -1 or trajes[name][i,1] == -1 :
                if error :
                    print("error  : if -1 still exists in results")
                    return None
                else :
                    trajes[name][i,0] = trajes[name][i,1] = np.nan

        slices, values = DetectContingence(trajes[name].tolist())

        valid = 0
        for i in values :
            if i == 1 :
                valid = valid + 1
        if valid != 1 and error :
            print("There are hole(s) in  the tracked data")
            return None

    catch = 0
    for i in range(trajes[names[0]].shape[0]):
        if np.isnan(trajes[names[0]][i,0]) or (np.isnan(trajes[names[1]][i,0])) or np.isnan(trajes[names[0]][i,1]) or (np.isnan(trajes[names[1]][i,1])):
            if catch == 1:
                catch = 0
                trajes[names[0]][i-1,0]= trajes[names[0]][i-1,1]=  trajes[names[1]][i-1,0]= trajes[names[1]][i-1,1]= np.nan

            trajes[names[0]][i,0]= trajes[names[0]][i,1]= trajes[names[1]][i,0]= trajes[names[1]][i,1]= np.nan
        else :
            if catch == 0 :
                catch = 1
                trajes[names[0]][i,0]= trajes[names[0]][i,1]= trajes[names[1]][i,0]= trajes[names[1]][i,1]= np.nan

    #TODO check that these is no spot where the acceleration of one or the other tracker is very high. Test if this is true for
    # 2 consecutive points(1peak) or for 2 distant points(a shift and then recovery) If yes, show the frame, or correct if the shift is short enough

    return trajes


if __name__ == "__main__":



    sys.exit()
    rootfolder = network.find_favoritesRootFolder()
    #sessions = [1650,1651,1647,1649,1646,1648,1643,1645,1642,1644,1639,1641,1638,1640,1637,1629,1630,1631,1632,1636,1628,1627,1626]
    #sessions = [1556,1557,1559,1560,1561,1562,1563,1564] #sessions of 14 02 20
    #sessions = [1546,1547,1548,1549] #sessions of 13_1 (fiberless)
    #sessions = [1553,1555] #sessions of 13_2 (with fiber) (21 & 22 : not working)
    sessions = [1548,1547]

    untrusted = False

    treshold = 55



    for sess in sessions :
        SessionDataFrame = database_IO.SessionDataframe( sess , rootfolder)
        for index, I in enumerate(SessionDataFrame["BEHpath"]):
            print(f"Video:{index} of {SessionDataFrame.shape[0]}")
            if untrusted and index == 2 :
                Trackers_Video(SessionDataFrame.at[int(index),"BEHpath"], output = os.path.join(SessionDataFrame.at[int(index),"PIPELINEpath"], "Trajectories" ,f'{SessionDataFrame.at[int(index),"FullName"]}_merge.pickle' ),thresh = treshold, test = True, overwrite = True)
                untrusted = False
            else :
                Trackers_Video(SessionDataFrame.at[int(index),"BEHpath"], output = os.path.join(SessionDataFrame.at[int(index),"PIPELINEpath"], "Trajectories" ,f'{SessionDataFrame.at[int(index),"FullName"]}_merge.pickle' ),thresh = treshold, test = False, overwrite = True)



