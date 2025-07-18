#
# DM5 Reader for InSitu 4DSTEM
#
#   Version Control:
#   DigitalMicrograph       3.6.1.4719
#   Python                  3.10
#
#   Numpy                   1.23.5
#   H5Py                    3.90
#
#   Primary development and testing in Windows 10 using python 3.10 embedded miniconda environment in DigitalMicrograph
#   3.61.4719. Additional testing performed using python 3.9 and PyCharm under MacOS Ventura 13.1
#
# Description:
#   This script is meant as a basic template to demonstrate how to access and read InSitu director images saved in .dm5
#   format from InSitu 4DSTEM data that were acquired in DigitalMicrograph. The .dm5 file format is written using HDF5
#   libraries and can be read by external applications other than Gatan DigitalMicrograph.
#
#   If this script is run in DigitalMicrograph without any code modification, it will:
#
#       1) Access and extract a user defined timeslice from the InSitu dataset of slice index N.
#       2) Display the selected timeslice as a 4D data array as a DM image object.
#       3) Print the 4D STEM calibrations for dimensions: x, y, h and k, to the results window as text.
#
#   The DM package functions are not accessible outside of DigitalMicrograph. Visualization code for step 2) is skipped
#   if this script is run in any application other than DigitalMicrograph.
#
# Instructions:
#
#       1) After running your InSitu 4DSTEM experiment, save the InSitu director image for your InSitu 4DSTEM
#       dataset as a .dm5 file.
#
#       The default file type for the director image is typically .dm4. If you forget step 1) or wish to run this script
#       on older data, open the director image in DigitalMicrograph version 3.60 or higher and save an additional
#       director image file with .dm5 format instead of .dm4.
#
#       The software will display a prompt stating: "You're trying to save an ImageDocument associated with an InSitu
#       sequence. Do you want to save another copy of the whole InSitu dataset".
#
#       This is typically not necessary. The DEFAULT RECOMMENDATION is to SELECT NO HERE.
#
#       2) !!!! ENSURE YOUR .DM5 FILE IS IN THE SAME FOLDER LOCATION AS THE RAW FILES WHICH YOU WISH TO PROCESS !!!
#
#       3) In function main(), set variable N to the timeslice you wish to extract from InSitu time series raw files.
#       The main() codeblock is between lines 213 - 234 and N is defined in line 225 at time of writing
#       (LJS - 2024.10.01)
#
#       4) Save this script file
#
#       5) Execute this script by pressing the "ctrl" and "enter" keys together, or left click "execute" button in the
#       script window.
#
#   L J Spillane Copyright © 2024 Gatan Inc.
#
#######################################################################################################################

import os
import sys
from math import pi
from tkinter import filedialog as fd

import h5py
import numpy as np
import emd

sys.argv.extend(["-a", " "])


class InSitu4DSTEM_Reader:

    def __init__(self, filePath):
        print("Simple DM5 reader for InSitu 4DSTEM:")
        print("")

        # Save to class member variables
        self.filePath = filePath

        # Additional Initialisation functions
        self.OpenDirectorImageFile_DM5()
        self.GetDocumemtObjectListAttributes()
        self.GetImageSourceListAtrributes()
        self.GetImageListAtributes()

        # Variable defining index of active raw file. Must be initialised as 0
        self.rawFileIndex = 0

    def OpenDirectorImageFile_DM5(self):
        print("Opening: " + str(self.filePath))
        f = h5py.File(self.filePath, "r")

        # Save to class member variables
        self.f = f

    def GetDocumemtObjectListAttributes(self):
        group1 = self.f.get("/DocumentObjectList/[0]")

        sourceIndex = group1.attrs["ImageSource"]
        imageDisplayType = group1.attrs["ImageDisplayType"]

        # Save to class member variables
        self.sourceIndex = sourceIndex
        self.imageDisplayType = imageDisplayType

    def GetImageSourceListAtrributes(self):

        group2Path = "/ImageSourceList/[" + str(self.sourceIndex) + "]"

        group2 = self.f.get(group2Path)

        imageRef = group2.attrs["ImageRef"]
        className = group2.attrs["ClassName"].decode("utf-8")

        # Save to class member variables
        self.imageRef = imageRef
        self.className = className

    def GetImageListAtributes(self):
        group3Path = "/ImageList/[" + str(self.imageRef) + "]"
        group3 = self.f.get(group3Path)

        # get the ImageData tagGroup
        imageData = group3.get("ImageData")

        # get data as numpy array
        data = imageData.get("Data")

        # Save to class member variables
        self.imageData = imageData
        self.data = data

    def GetDimCalibrations(self, dimIndex: int):
        # Function will print origin, scale and units for the defined dimension
        # 4DSTEM data has 4 dimensions, starting at index 0

        # Check N is integer
        if not isinstance(dimIndex, int):
            raise TypeError(f"Expected an integer, got {type(dimIndex).__name__}")

        # get the calibrations of the x-axis
        group = self.imageData.get("Calibrations/Dimension/[" + str(dimIndex) + "]")

        dimOrigin = group.attrs["Origin"]
        dimScale = group.attrs["Scale"]
        dimUnits = group.attrs["Units"].decode("latin-1")

        print("\nDimension: " + str(dimIndex))
        print("Origin [" + str(dimIndex) + "]:" + str(dimOrigin))
        print("Scale [" + str(dimIndex) + "]:" + str(dimScale))
        print("Units [" + str(dimIndex) + "]:" + str(dimUnits))

    def OpenDataInDM(self, dataArray):
        # Function will create DM image object from a numpy array and show this image in the active DM workspace
        # If this script is executed outside of DigtalMicrograph, the visualization code is skipped.

        if "DigitalMicrograph" in sys.modules:
            img = DM.CreateImage(dataArray.copy())
            img.GetTagGroup().SetTagAsBoolean("Meta Data:Data Order Swapped", True)
            img.GetTagGroup().SetTagAsString("Meta Data:Format", "Diffraction image")

            img.ShowImage()
        else:
            print("")
            print(
                "DigitalMicrograph module not present. Array visualization code skipped."
            )

    def GetNthFrame(self, N: int):
        # Function will read out a timeslice from the InSitu dataset and ouput as a number array
        # Any value of N may be chosen from 0 up to the last timeslice in the timeseries
        # The rawfile index is determined automatically depending on the value of N chosen

        # Check N is integer
        if not isinstance(N, int):
            raise TypeError(f"Expected an integer, got {type(N).__name__}")

        directorImageFilePath = self.filePath

        # Find raw file from director image filePath. Assumes director and raw files are in same folder
        rawFilePath = directorImageFilePath[:-4] + ".raw"
        raw_length = os.path.getsize(rawFilePath)

        # Calculate the number of pixels and bytes per frame
        data_type = self.data.dtype
        # pixels_per_frame = np.prod(self.data.shape, dtype="uint")
        pixels_per_frame = (
            self.data.shape[0]
            * self.data.shape[1]
            * self.data.shape[2]
            * self.data.shape[3]
        )

        # bytes_per_frame = self.data.nbytes
        bytes_per_frame = pixels_per_frame * data_type.itemsize

        # Calculate the starting byte position for the Nth frame
        read_start = N * bytes_per_frame

        # make sure to fail gracefully if the read_start is larger than the raw file size
        if read_start >= raw_length:
            return False

        # Determine the correct raw file and offset within that file
        rawFileIndex = read_start // raw_length
        print(read_start / raw_length)
        read_start = read_start % raw_length

        # Adjust raw file path if necessary
        if rawFileIndex != 0:
            rawFilePath = os.path.splitext(rawFilePath)[0] + ".raw_" + str(rawFileIndex)

        print(
            f"Reading frame {N} from raw file: {rawFilePath}, \n "
            f"raw file index: {rawFileIndex},\n "
            f"read start: {read_start},\n "
            f"bytes per frame: {bytes_per_frame},\n "
            f"pixels per frame: {pixels_per_frame},\n "
            f"raw length: {raw_length}\n"
        )

        # Read from raw file
        # Code supports readout from datasets containing multiple raw files
        if read_start + bytes_per_frame > raw_length:
            count_1 = (raw_length - read_start) // data_type.itemsize
            print(f"Reading {count_1} pixels from first part of raw file.")
            # If the
            count_2 = pixels_per_frame - count_1
            print(f"Reading {count_2} pixels from second part of raw file.")

            array_1 = np.fromfile(
                rawFilePath, dtype=data_type, count=count_1, offset=read_start
            )

            print(array_1.shape)

            rawFileIndex += 1
            rawFilePath = os.path.splitext(rawFilePath)[0] + ".raw_" + str(rawFileIndex)

            array_2 = np.fromfile(rawFilePath, dtype=data_type, count=count_2, offset=0)

            print(array_2.shape)

            array = np.concatenate([array_1, array_2])
        else:
            array = np.fromfile(
                rawFilePath, dtype=data_type, count=pixels_per_frame, offset=read_start
            )

        array = array.reshape(self.data.shape)

        return array


def save_frame_as_hdf5(frame_data, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"frame_{index}.h5")
    with h5py.File(filename, "w") as f:
        f.create_dataset("frame", data=frame_data)
        f.attrs["description"] = f"4D-STEM timeslice frame {index}"
        f.attrs["shape"] = frame_data.shape
    print(f"Saved: {filename}")

def save_frame_as_emd(frame_data, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"frame_{index}.emd")

    doc = emd.EMD()
    doc["/data"] = frame_data  # your NumPy array
    doc["/data"].attrs["description"] = f"4D-STEM timeslice frame {index}"

    emd.save(doc, filename)
    print(f"Saved: {filename}")


def main():
    for i in range(5, 14):
        try:
            filePath = fd.askopenfilename(
                filetypes=[("DM5 HDF5 Files", ".dm5")],
                title=f"../data/BTO/InSitu_({i})/",
            )
            if not filePath:
                raise FileNotFoundError(
                    "No file was selected, operation was canceled by the user."
                )
        except FileNotFoundError as e:
            print(e)
            exit(0)

        MyInSitu4DSTEM = InSitu4DSTEM_Reader(filePath)

        N = 15  # number of timeslices (frames) to extract from the time series
        output_dir = os.path.splitext(filePath)[0] + "_frames_h5"

        for j in range(N):
            print(f"Extracting frame {j} from timeslice {N}...")
            frame_data = MyInSitu4DSTEM.GetNthFrame(
                j
            )  # shape: (34, 206, 512, 512) or similar
            if type(frame_data) is not np.ndarray:
                print(f"Frame {j} does not exist in the raw data.")
                continue
            print(frame_data.shape)
            save_frame_as_hdf5(frame_data, output_dir, j)

        # # Visualize DataArray for timeslice N - This will only work if the script is running in DigitalMicrograph
        # MyInSitu4DSTEM.OpenDataInDM(imgData)

        # # Get Calibrations - 4DSTEM has 4 axes
        # dimensions = 4
        # for dimIndex in range(dimensions):
        #     MyInSitu4DSTEM.GetDimCalibrations(dimIndex)


if __name__ == "__main__":
    main()
