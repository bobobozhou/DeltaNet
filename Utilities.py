import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
import VIS.VIS_Volume as VVV
import matplotlib.pyplot as plt


def loadImages(srcFolder):
    # get the volume.nii file list
    fileList = [f for f in listdir(srcFolder) if isfile(join(srcFolder, f)) and 'segmentation' not in f and 'raw' not in f]
    print 'IMAGE FILE LIST: ' + str(fileList)

    # imread the volume.nii from fileList
    sitkImages = dict()
    for f in fileList:
        sitkImages[f] = sitk.ReadImage(join(srcFolder, f))

    X_vol = []
    for key in sorted(sitkImages):
        V = sitk.GetArrayFromImage(sitkImages[key]).astype(dtype=float)
        X_vol.append(np.reshape(V, (1, 128, 128, 64, 1)))

    X_vol_con = np.concatenate(X_vol, axis=0)
    return X_vol_con


def loadGT(srcFolder):
    # get the volume.nii file list
    fileList = [f for f in listdir(srcFolder) if isfile(join(srcFolder, f)) and 'segmentation' not in f and 'raw' not in f]

    # get the corresponding GT file list
    gtList = list()
    for f in fileList:
        filename, ext = splitext(f)
        gtList.append(join(filename + '_segmentation' + ext))
    print 'LABEL FILE LIST: ' + str(gtList)

    # imread the volume_segmentation.nii from gtList
    sitkGT = dict()
    for f in gtList:
        sitkGT[f] = sitk.ReadImage(join(srcFolder, f))

    X_lb = []
    for key in sorted(sitkGT):
        V = sitk.GetArrayFromImage(sitkGT[key]).astype(dtype=float)
        X_lb.append(np.reshape(V, (1, 128, 128, 64, 1)))

    X_lb_con = np.concatenate(X_lb, axis=0)
    return X_lb_con


if __name__ == '__main__':
    srcFolder = 'Dataset1/Train/'
    dat = loadImages(srcFolder)
    gt = loadGT(srcFolder)

    VVV.multi_slice_viewer(dat[0, :, :, :, 0])
    VVV.multi_slice_viewer(gt[0, :, :, :, 0])
    plt.show()
