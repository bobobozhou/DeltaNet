import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import VIS.VIS_Volume as VVV

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    VVV.multi_slice_viewer(ct_scan)

    return ct_scan, origin, spacing

load_itk('Dataset1/Train/0082_A.mhd')
load_itk('Dataset1/Train/0082_A_segmentation.mhd')
plt.show()
