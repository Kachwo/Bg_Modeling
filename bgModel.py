import cv2
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from PIL import Image
import time
from skimage.feature import hog
from skimage import exposure
from numpy import unique
from scipy.stats import entropy as scipy_entropy

MHI_DURATION = 10
DEFAULT_THRESHOLD = 32


def genMHI(inputDir, dataset):  # original work from VideoForensics, modified     

    # Array for storing the generated MHI
    Arr = []
    prev_frame = None
    timestamp = 0
    kernel = np.ones((1, 1), np.uint8)

    # Retrieve the correct dataset video frames
    if dataset == "SBM":
        List = glob.glob(os.path.join(inputDir, 'input', '*.jpg'))
    elif dataset == "SBI":
        List = glob.glob(os.path.join(inputDir, 'input', '*.png'))

    List = sorted(List)

    for i in List:  # Generate the corresponding MHI frame by frame

        frame = cv2.imread(i)
        h, w = frame.shape[:2]
        motion_history = np.zeros((h, w), np.float32)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            ret, fgmask = cv2.threshold(
                gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
            timestamp += 1
            # Update the motion history
            cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

            # Normalize the motion history
            mh = np.uint8(
                np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            dilation = cv2.dilate(mh, kernel, iterations=1)
            Arr.append(dilation)    # Append the current MHI into the array

        prev_frame = frame.copy()

    return Arr  # Return the array which stored all the MHIs


def makeVideo(inputArr, outputFileName, outputDir, MakeColorImg=False): # Function implemented by MHI, used for generating a video from a set of frames (Not used)

    height, width = inputArr[0].shape[0], inputArr[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'x264' doesn't work
    out = cv2.VideoWriter('{}/{}.mp4'.format(outputDir,
                                             outputFileName), fourcc, 30, (width, height))  # 'out' is the video generated, 'outputDir' is the output location
    for k in range(len(inputArr)):
        frame = cv2.cvtColor(inputArr[k], cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()   # Release the completed video


def rgb2gray(rgb):  # Function for converting color image to gray image

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray # Return the gray image


def splitImage(inImg):  # Function for splitting the image during quad tree decomposition

    h, w = inImg.shape[0], inImg.shape[1]
    off2X = int(w/2)
    off2Y = int(h/2)

    img1 = inImg[0:off2Y, 0:off2X]
    img2 = inImg[0:off2Y, off2X:w]
    img3 = inImg[off2Y:h, 0:off2X]
    img4 = inImg[off2Y:h, off2X:w]

    return img1, img2, img3, img4   # Return the 4 equal-size quadrants


def combineImage(q1, q2, q3, q4):   # Function for combining the image during quad tree decomposition

    h = q1.shape[0] + q3.shape[0]
    w = q1.shape[1] + q2.shape[1]
    d = q1.shape[2]
    off2X = int(w/2)
    off2Y = int(h/2)
    img = np.arange(h*w*d).reshape(h, w, d)
    img[0:off2Y, 0:off2X] = q1
    img[0:off2Y, off2X:w] = q2
    img[off2Y:h, 0:off2X] = q3
    img[off2Y:h, off2X:w] = q4

    return img  # Return the combined image


def shannon_entropy(image, base=10):    # original work from the scikit-image team for entropy calculation
    _, counts = unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)


def gen_init(img_list): # Function for selecting the initial background image
    
    # Set a unreachable initial entropy value for comparison
    entropy = 1000000000000000000000000
    # 'index' is for showing the frame number of the initial bg
    index = 0
    temp = 0

    for k in range(len(img_list)):
        fr = mpimg.imread(img_list[k], 1)
        x = shannon_entropy(rgb2gray(fr))   # Before the entropy calculation, we need to convert the image into grayimage first
        if x < entropy:     # If the entropy value of the current frame is less than the default value 'entropy'
            entropy = x
            index = k
        # Show the initialization progress
        temp = temp + 1
        progress = temp / len(img_list) * 100
        print("progress of gen_init : ", int(progress), "%")
    print(index)
    bg = mpimg.imread(img_list[index], 1)   # 'bg' is the selected initial bg image

    return bg   # Return the initial bg image


def q_tree(fr_MHI, fr1, fr2, bg, record_Map, oneTenth):     # Function for quad tree decomposition
    # 'fr_MHI' is the MHI at time t
    # 'fr1' is the original frame at time t
    # 'fr2' is the original frame at time t-1
    # 'bg' is the current bg image
    # 'record_Map' is the corresponding record map
    # 'oneTenth' is the value which equals to onetenth of the total number of video frames

    # Split the image into 4 equal-size quadrant
    a1, a2, a3, a4 = splitImage(fr_MHI)
    b1, b2, b3, b4 = splitImage(fr1)
    c1, c2, c3, c4 = splitImage(fr2)
    d1, d2, d3, d4 = splitImage(bg)
    e1, e2, e3, e4 = splitImage(record_Map)

    # Return original bg if size < 2x2
    if (a1.shape[0] < 2) and (a1.shape[1] < 2):
        return bg
    if (a2.shape[0] < 2) and (a2.shape[1] < 2):
        return bg
    if (a3.shape[0] < 2) and (a3.shape[1] < 2):
        return bg
    if (a4.shape[0] < 2) and (a4.shape[1] < 2):
        return bg

    if np.mean(a1) > 0:     # If this quadrant contains white pixels
        q1 = q_tree(a1, b1, c1, d1, e1, oneTenth)   # This quadrant undergoes another quad tree decomposition
    else:
        if abs(np.mean(b1) - np.mean(c1)) < 1:      # If the difference of intensity between fr1 and fr2 is less than 1
            i = e1[0, 0]        # Get the value of the first entry in the record map
            if i >= 0:          
                i = i + 1       # Increase 'i' by 1
            if i >= oneTenth:   # If 'i' is greater than 'oneTenth', the corresponding part of bg is undated by fr1, and i returns to 0
                q1 = b1
                i = 0
            else:
                q1 = d1     # Otherwise, bg unchange
            e1.fill(i)      # All values in the corresponding record map change to 'i'

        else:
            q1 = d1

    if np.mean(a2) > 0:     # Same as above
        q2 = q_tree(a2, b2, c2, d2, e2, oneTenth)
    else:
        if abs(np.mean(b2) - np.mean(c2)) < 1:
            i = e2[0, 0]
            if i >= 0:
                i = i + 1
            if i >= oneTenth:
                q2 = b2
                i = 0
            else:
                q2 = d2
            e2.fill(i)
        else:
            q2 = d2

    if np.mean(a3) > 0:     # Same as above
        q3 = q_tree(a3, b3, c3, d3, e3, oneTenth)
    else:
        if abs(np.mean(b3) - np.mean(c3)) < 1:
            i = e3[0, 0]
            if i >= 0:
                i = i + 1
            if i >= oneTenth:
                q3 = b3
                i = 0
            else:
                q3 = d3
            e3.fill(i)
        else:
            q3 = d3

    if np.mean(a4) > 0:     # Same as above
        q4 = q_tree(a4, b4, c4, d4, e4, oneTenth)
    else:
        if abs(np.mean(b4) - np.mean(c4)) < 1:
            i = e4[0, 0]
            if i >= 0:
                i = i + 1
            if i >= oneTenth:
                q4 = b4
                i = 0
            else:
                q4 = d4
            e4.fill(i)
        else:
            q4 = d4

    # Combine the 4 quadrant to return the bg 
    frame = combineImage(q1, q2, q3, q4)

    return frame        # Return the current computed bg



print("Enter Dataset Name (SBM / SBI):")    # Select the dataset, Type 'SBM' or 'SBI'
X = input()
print("Enter Category/Video name:") # Type the video category and name, e.g. type 'Basic/Blurred' for SBM, type 'Board' for SBI
Y = input()

start = time.time()     # Start the timer for calculating the computation time
dataset_Dir = '/Users/user/Desktop/PythonCode/' + X + '_dataset/' + Y    # My default dataset video frames location
bg_Dir = '/Users/user/Desktop/PythonCode/' + \
    X + '_dataset/' + Y + '/result'                                                         # My default path for storing the computed bg
if X == "SBM":
    fr_List = glob.glob(os.path.join(dataset_Dir, 'input', '*.jpg'))                        # 'fr_List' is a list of original video frames
elif X == "SBI":
    fr_List = glob.glob(os.path.join(dataset_Dir, 'input', '*.png'))

fr_List = sorted(fr_List)       # 'fr_List' is sorted according to the frame number

if len(fr_List) <= 0:
    raise ValueError("Wrong dataset path. Please enter the correct path.")      # Raise error message if the dataset is not read correctly


mhi_Arr = genMHI(dataset_Dir, X)        # 'mhi_Arr' is the array which stores all the MHIs
for i in range(len(mhi_Arr)):           # Rename the MHIs
    if i < 10:
        name = "mhi00000" + str(i) + ".jpg"
    elif i >= 10 and i < 100:
        name = "mhi0000" + str(i) + ".jpg"
    elif i >= 100 and i < 1000:
        name = "mhi000" + str(i) + ".jpg"
    elif i >= 1000 and i < 10000:
        name = "mhi00" + str(i) + ".jpg"
    mpimg.imsave(dataset_Dir+'/MHI/' + name, mhi_Arr[i], cmap='Greys_r')    # Save the MHIs in the  default path
mhi_List = glob.glob(os.path.join(dataset_Dir, 'MHI', '*.jpg'))             # 'mhi_List' is the list which stores all the renamed MHIs
mhi_List = sorted(mhi_List)                                                 # 'mhi_List' is sorted according to the MHI number

bg = gen_init(fr_List)      # Generate the initial bg
h, w = bg.shape[0], bg.shape[1]     
record_Map = np.zeros((h, w))       #create the record map
temp = 0
oneTenth = int(len(fr_List) / 10)       # Calculate the 'oneTenth' value
print(oneTenth)

for k in range(len(fr_List)-1): 

    fr_MHI = mpimg.imread(mhi_List[k])    # 'fr_MHI' is the MHI at time t
    fr1 = mpimg.imread(fr_List[k+1], 1)   # 'fr1' is the original frame at time t
    fr2 = mpimg.imread(fr_List[k], 1)     # 'fr2' is the original frame at time t-1
    bg = q_tree(fr_MHI, fr1, fr2, bg, record_Map, oneTenth)     # Undergoes quad tree decomposition

    temp = temp + 1
    progress = temp / len(fr_List) * 100
    print("progress : ", int(progress), "%")    # Show the computation progress

final = np.array(bg, dtype='uint8')     # 'final' is the final bg image
mpimg.imsave(bg_Dir + "/bg.jpg", final)     # Save the final bg in the default path, named 'bg.jpg'


end = time.time()   # Stop the timer
totalMin = int((end - start) / 60)
print("Time used: ", totalMin, " min")      # Show the computation time
