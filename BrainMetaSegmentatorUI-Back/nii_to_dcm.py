import os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian

def auto_window_level(data):
    """自动计算窗宽和窗位."""
    max_val = np.percentile(data, 99)
    min_val = np.percentile(data, 1)
    window_center = (max_val + min_val) / 2
    window_width = max_val - min_val
    return window_center, window_width

def convert_nifti_to_dicoms(nifti_path, output_folder, patient_id, patient_name, patient_birth_date):
    # Load the NIfTI file
    nii = nib.load(nifti_path)
    data = nii.get_fdata()

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate unique identifiers for the study and series
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()

    # Compute window center and width
    window_center, window_width = auto_window_level(data)

    for i in range(data.shape[2]):  # Assuming the third dimension is the slice dimension
        slice_data = data[:, :, i]
        normalized_data = np.interp(slice_data, (slice_data.min(), slice_data.max()), (0, 65535))

        # Create and set up DICOM dataset
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.ImplementationClassUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        ds.PatientID = patient_id
        ds.PatientName = patient_name
        ds.PatientBirthDate = patient_birth_date
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.StudyID = "1"
        ds.SeriesNumber = "1"
        ds.InstanceNumber = str(i + 1)
        ds.Modality = 'CT'
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.SOPInstanceUID = generate_uid()

        ds.Rows, ds.Columns = slice_data.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.WindowCenter = str(int(window_center))
        ds.WindowWidth = str(int(window_width))
        ds.PixelData = normalized_data.astype(np.uint16).tobytes()

        # Save the DICOM file
        dicom_filename = os.path.join(output_folder, f"slice_{i+1}.dcm")
        ds.save_as(dicom_filename)
        print(f"Saved DICOM slice: {dicom_filename}")


# Usage
nifti_path = 'E:BaiduNetdiskDownload/BTCV_Challenge_abdomenCT_13organs_RawData/RawData/Training/img/img0001.nii.gz' # Update with the path to your .nii.gz file
output_path = 'E:BaiduNetdiskDownload/BTCV_Challenge_abdomenCT_13organs_RawData/RawData/Training/out'   # Update with desired output path for DICOM file
patient_id = '1234567890'
patient_name = 'John Doe'
patient_birth_date = '19900101'

convert_nifti_to_dicoms(nifti_path, output_path, patient_id, patient_name, patient_birth_date)

