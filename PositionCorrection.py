# -*- coding: utf-8 -*-

import os, sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal as sig
from scipy.ndimage import rotate

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

sys.path.append(uppath(__file__, 2))

from LibUtils import signal

from LibUtils import geometry

def ConvertValues_front_nose( TrackerLine , coordsNose , coordsNeck ):
    NoseNeckLine = geometry.ULine( geometry.UPoint( coordsNose["x"],coordsNose["y"] ) , geometry.UPoint( coordsNeck["x"],coordsNeck["y"] ) )
    FrontNoseLine = geometry.ULine(NoseNeckLine.A,TrackerLine.A)
    return FrontNoseLine.length, FrontNoseLine.angle(TrackerLine) , NoseNeckLine.angle(TrackerLine)
            #distance from front tracker to nose , angle from Front_nose line to trackers line, angle from manually marked line and tracker line

def CalculateNose( TrackerLine , Front_to_Nose_length, Front_to_Nose_angle ):
    """
    Parameters
    ----------
    TrackerLine : TYPE
        DESCRIPTION.
    Front_to_Nose_length : TYPE
        DESCRIPTION.
    Front_to_Nose_angle : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return TrackerLine.A.project( Front_to_Nose_length , TrackerLine.angle() + Front_to_Nose_angle )

def CalculateNeck( array , TrackerLine, NosePoint , Nose_Neck_length, Tracker_suppNoseNeck_angle, Checkwidth, offsetangle = None, **kwargs ):
    
    if offsetangle is None :
        search_flag = True
        offsetangle = 0
    else :
        search_flag = False

    Supp_NoseNeckLine = geometry.ULine(NosePoint,  NosePoint.project( -Nose_Neck_length, TrackerLine.angle() + Tracker_suppNoseNeck_angle + offsetangle )  )

    NeckCheckLine = geometry.ULine( Supp_NoseNeckLine.B.project( -Checkwidth,90+Supp_NoseNeckLine.angle() ) ,  Supp_NoseNeckLine.B.project( Checkwidth,90+Supp_NoseNeckLine.angle() )  )

    if not search_flag :
        return Supp_NoseNeckLine , NeckCheckLine

    while True :
        peaks = NeckPeaks( NeckCheckLine , array, plot = kwargs.get("plot",True) )
        symscore = peaks[0]-peaks[1]
        if abs(symscore) < 3 :
            neck_realwidth = peaks[0]
            break
        else :
            if symscore > 0:
                offsetangle = offsetangle + 0.2

            else :
                offsetangle = offsetangle - 0.2
            Supp_NoseNeckLine = geometry.ULine(NosePoint,  NosePoint.project( -Nose_Neck_length, TrackerLine.angle() + Tracker_suppNoseNeck_angle + offsetangle )  )
            NeckCheckLine = geometry.ULine( Supp_NoseNeckLine.B.project( -Checkwidth,90+Supp_NoseNeckLine.angle() ) ,  Supp_NoseNeckLine.B.project( Checkwidth,90+Supp_NoseNeckLine.angle() )  )

    NeckLine = geometry.ULine( Supp_NoseNeckLine.B.project( -neck_realwidth,90+Supp_NoseNeckLine.angle() ) ,  Supp_NoseNeckLine.B.project( neck_realwidth,90+Supp_NoseNeckLine.angle() )  )

    return Supp_NoseNeckLine, NeckLine , neck_realwidth , offsetangle

def NeckPeaks( NeckCheckLine, array, plot = False ):


    x, y = np.linspace(*NeckCheckLine.X, int(NeckCheckLine.length)), np.linspace(*NeckCheckLine.Y, int(NeckCheckLine.length))
    ValueSlice = signal.MapSpline( array.T, np.vstack( (x,y) ) )
    diffslice = np.abs(np.gradient(signal.Smooth1D(ValueSlice,25)))
    neck_ckeck_width = NeckCheckLine.length/2
    if plot :
        #matplotlib.use('Qt5Agg')
        plt.plot(diffslice)
        plt.xticks([0,neck_ckeck_width/2,neck_ckeck_width,neck_ckeck_width*1.5,2*neck_ckeck_width],(-neck_ckeck_width,-neck_ckeck_width/2,0,neck_ckeck_width/2,neck_ckeck_width))
        plt.show()

    midpoint = int(NeckCheckLine.length/2)

    peaks , vals = sig.find_peaks( diffslice, height = 5 )
    prominence = sig.peak_prominences(diffslice,peaks)[0].tolist()

    mostprom_peaks = []
    for start in [0,midpoint]:
        maxprom = 0
        indx = None
        for pick, promi in zip(peaks,prominence) :

            if start <= pick < start + midpoint :
                if promi > maxprom :
                    promi = maxprom
                    indx = pick
        if start == 0 :
            mostprom_peaks.append(abs(indx-midpoint))
        else :
            mostprom_peaks.append(indx-start)

    return mostprom_peaks

def StabilizeVideo( array, nose_necklines_collec ,padvalue = 600, outputwidth = 500 ):

    newlist = []
    extentvalue = padvalue/2

    for image_idx in range(array.shape[2]):
        nosepoint = nose_necklines_collec[image_idx].A
        if not nosepoint.isnan :

            img = array[:,:,image_idx]
            img = np.pad(img,padvalue, mode='constant')
            centerx , centery = nosepoint.x+padvalue, nosepoint.y+ padvalue
            img = img[ int(centery-extentvalue) :  int(centery+extentvalue) , int(centerx-extentvalue) : int(centerx+extentvalue) ]

            newlist.append(img)

            #im_bg(ax,img)
            #break

    newarray = np.stack(newlist)
    newarray = np.moveaxis(newarray, 0, -1)

    outputlist = []

    #fig, ax = plt.subplots(1,1, figsize = (5,5) )
    anglelist = nose_necklines_collec.angles(removenans = True)
    for image_idx in range(newarray.shape[2]):
        angle = anglelist[image_idx]
        newimg = rotate(newarray[:,:,image_idx],angle,reshape = False,order = 5)
        #im_bg(ax,newimg)
        #print(newimg.shape)
        #plt.plot(xw,yw,'o')
        finalimg = newimg[ int( extentvalue - outputwidth/2 )  : int( extentvalue + outputwidth/2 ) , int( extentvalue - outputwidth/2) : int( extentvalue + outputwidth/2 ) ]
        #im_bg(ax,finalimg)
        outputlist.append(finalimg)

    outputarray = np.stack(outputlist)
    outputarray = np.moveaxis(outputarray, 0, -1)

    return outputarray

def CutStabarray(stabarray):

    cut_arrays = []
    cut_arrays.append( np.flip(stabarray[:,:int(stabarray.shape[1]/2),:], axis = 1) )
    cut_arrays.append( stabarray[:,int(stabarray.shape[1]/2):,:] )

    return cut_arrays