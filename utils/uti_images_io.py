# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    This Module defines functions for reading images from folder, video, or Webcam
    
    Main classes and functions:

    Read:
        class Read_Images_From_Folder
        class Read_Images_From_Video
        class Read_Images_From_Webcam
    Write:
        class Video_Writer
        class Images_Writer
    
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import time
import multiprocessing
import warnings
import glob
import queue
import threading
# […]

# Libs
import numpy as np 
import cv2
# […]

# Own modules
# from {path} import {class}
# […]


# Main functions

class Read_Images_From_Folder(object):
    ''' Read all images in a given folder, call as module.
    By default, all files under the folder are considered as image file.
    '''

    def __init__(self, sFolder_Path):
        self.sFile_Names = sorted(glob.glob(sFolder_Path + "/*"))
        self.iImages_Counter = 0
        self.sCurrent_File_Name = ""

    def Read_Image(self):
        if self.iImages_Counter >= len(self.sFile_Names):
            return None
        self.sCurrent_File_Name = self.sFile_Names[self.iImages_Counter]
        Image = cv2.imread(self.sCurrent_File_Name, cv2.IMREAD_UNCHANGED)
        self.iImages_Counter += 1
        return Image

    def __len__(self):
        return len(self.sFile_Names)

    def Image_Captured(self):
        return self.iImages_Counter < len(self.sFile_Names)

    def Stop(self):
        None


class Read_Images_From_Video(object):
    def __init__(self, sVideo_Path, iSample_Interval=1):
        ''' Read Images from a video in a given folder, call as module.
        Arguments:
            sVideo_Path {string}: the path of the video folder.
            iSample_Interval {int}: sample every kth image.
        '''
        if not os.path.exists(sVideo_Path):
            raise IOError("Video does not exist: " + sVideo_Path)
        assert isinstance(iSample_Interval, int) and iSample_Interval >= 1
        self.iImages_Counter = 0
        self._bIs_Stoped = False
        self._Video_Captured = cv2.VideoCapture(sVideo_Path)
        Success, Image = self._Video_Captured.read()
        self._Next_Image = Image
        self._iSample_Interval = iSample_Interval
        self._iFPS = self.get_FPS()
        if not self._iFPS >= 0.0001:
            import warnings
            warnings.warn("Invalid fps of video: {}".format(sVideo_Path))

    def Image_Captured(self):
        return self._Next_Image is not None

    def Get_Current_Video_Time(self):
        return 1.0 / self._iFPS * self.iImages_Counter

    def Read_Image(self):
        Image = self._Next_Image
        for i in range(self._iSample_Interval):
            if self._Video_Captured.isOpened():
                Success, Frame = self._Video_Captured.read()
                self._Next_Image = Frame
            else:
                self._Next_Image = None
                break
        self.iImages_Counter += 1
        return Image

    def Stop(self):
        self._Video_Captured.release()
        self._bIs_Stoped = True

    def __del__(self):
        if not self._bIs_Stoped:
            self.Stop()

    def get_FPS(self):

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # Get video properties
        if int(major_ver) < 3:
            FPS = self._Video_Captured.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            FPS = self._Video_Captured.get(cv2.CAP_PROP_FPS)
        return FPS


class Read_Images_From_Webcam(object):
    def __init__(self, fMax_Framerate=30.0, iWebcam_Index=0):
        ''' Read images from Webcam, call as module.
        Argument:
            fMax_Framerate {float}: the maximum value of the camera framerate.
            iWebcam_Index {int}: index of the web camera. It should be 0 by default.
        '''
        # Settings
        self._fMax_Framerate = fMax_Framerate


        # Initialize video reader
        self._Video = cv2.VideoCapture(iWebcam_Index)
        self._bIs_Stoped = False

        # Maximal Elements to receive
        iQueue_Size = 3

        # Use a thread to keep on reading images from web camera
        self._Images_Queue = queue.Queue(maxsize=iQueue_Size)
        self._Is_Thread_Alive = multiprocessing.Value('i', 1)
        self._Thread = threading.Thread(
            target=self._Thread_Reading_Webcam_Frames)
        self._Thread.start()

        # Manually control the framerate of the webcam by sleeping
        self._fMin_Duration = 1.0 / self._fMax_Framerate
        self._fPrev_Time = time.time() - 1.0 / fMax_Framerate

    def Read_Image(self):
        fDuration = time.time() - self._fPrev_Time
        if fDuration <= self._fMin_Duration:
            time.sleep(self._fMin_Duration - fDuration)
        self._fPrev_Time = time.time()
        Image = self._Images_Queue.get(timeout=10.0)
        return Image

    def Image_Captured(self):
        return True  # The Webcam always returns a new frame

    def Stop(self):
        self._Is_Thread_Alive.value = False
        self._Video.release()
        self._bIs_Stoped = True

    def __del__(self):
        if not self._bIs_Stoped:
            self.Stop()

    def _Thread_Reading_Webcam_Frames(self):
        while self._Is_Thread_Alive.value:
            Success, Image = self._Video.read()
            if self._Images_Queue.full():  # if queue is full, pop one
                Image_to_Discard = self._Images_Queue.get(timeout=0.001)
            self._Images_Queue.put(Image, timeout=0.001)  # push to queue
        print("Web camera thread is dead.")


class Video_Writer(object):
    def __init__(self, sVideo_Path, fFramerate):
        ''' Read images from web camera, call as module.
        Argument:
            sVideo_Path {string}: The path of the folder.
            fFramerate {intenger}: Frame rate of the recorded video web camera.
        '''

        # -- Settings
        self._sVideo_Path = sVideo_Path
        self._fFramerate = fFramerate

        # -- Variables
        self._iImages_Counter = 0
        # initialize later when the 1st image comes
        self._video_writer = None
        self._Width = None
        self._Height = None

        # -- Create output folder
        sFolder = os.path.dirname(sVideo_Path)
        if not os.path.exists(sFolder):
            os.makedirs(sFolder)
            sVideo_Path

    def write(self, Image):
        self._iImages_Counter += 1
        if self._iImages_Counter == 1:  # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the codec
            self._Width = Image.shape[1]
            self._Height = Image.shape[0]
            self._video_writer = cv2.VideoWriter(
                self._sVideo_Path, fourcc, self._fFramerate, (self._Width, self._Height))
        self._video_writer.write(Image)

    def Stop(self):
        self.__del__()

    def __del__(self):
        if self._iImages_Counter > 0:
            self._video_writer.release()
            print("Complete writing {}fps and {}s video to {}".format(
                self._fFramerate, self._iImages_Counter/self._fFramerate, self._sVideo_Path))

        
class Images_Writer(object):
    def __init__(self, sImages_Path, fFramerate):
        ''' Read images from web camera, call as module.
        Argument:
            fMax_Framerate {float}: the real framerate will be reduced below this value.
            iWebcam_Index {int}: index of the web camera. It should be 0 by default.
        '''

        # -- Settings
        self._sVideo_Path = sVideo_Path
        self._fFramerate = fFramerate

        # -- Variables
        self._iImages_Counter = 0
        # initialize later when the 1st image comes
        self._video_writer = None
        self._Width = None
        self._Height = None

        # -- Create output folder
        sFolder = os.path.dirname(sVideo_Path)
        if not os.path.exists(sFolder):
            os.makedirs(sFolder)
            sVideo_Path

    def Write(self, Image):
        self._iImages_Counter += 1
        if self._iImages_Counter == 1:  # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the codec
            self._Width = Image.shape[1]
            self._Height = Image.shape[0]
            self._video_writer = cv2.VideoWriter(
                self._sVideo_Path, fourcc, self._fFramerate, (self._Width, self._Height))
        self._video_writer.Write(Image)

    def Stop(self):
        self.__del__()

    def __del__(self):
        if self._iImages_Counter > 0:
            self._video_writer.release()
            print("Complete writing {}fps and {}s video to {}".format(
                self._fFramerate, self._iImages_Counter/self._fFramerate, self._sVideo_Path))


 
class Image_Displayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._sWindow_Name = "CV2_Display_Window"
        cv2.namedWindow(self._sWindow_Name, cv2.WINDOW_NORMAL)

    def display(self, Image, Wait_Key_ms=1):
        cv2.imshow(self._sWindow_Name, Image)
        cv2.waitKey(Wait_Key_ms)

    def __del__(self):
        cv2.destroyWindow(self._sWindow_Name)


def test_Read_From_Webcam():
    ''' Test the class Read_From_Webcam '''
    Webcam_Reader = Read_Images_From_Webcam(fMax_Framerate=10)
    local_Image_Displayer = Image_Displayer()
    import itertools
    for i in itertools.count():
        Image = Webcam_Reader.Read_Image()
        if Image is None:
            break
        print(f"Read {i}th image...")
        local_Image_Displayer.display(Image)
    print("Program ends")

def test_Read_From_Video():

    # Get Sources from class Read_From_Video
    sVideo_Path = '/home/zhaj/tf_test/Realtime-Action-Recognition-master/output/01-10-15-16-44-054/'
    iSample_Interval = 1
    Images_From_Video = Read_Images_From_Video(sVideo_Path, iSample_Interval)
    local_Image_Displayer = Image_Displayer()
    import itertools
    for i in itertools.count():
        Image = Read_Images_From_Video.Read_Image(iSample_Interval)
        if Image is None:
            break
        print(f"Read {i}th image...")
        local_Image_Displayer.display(Image)
    print("Program ends")




if __name__ == "__main__":
    test_Read_From_Folder()


