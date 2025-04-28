import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2

def load_clinical_data(clinical_csv_path):
    """Load the clinical data and assign the patient ID label based on case_csPCa."""
    clinical_info = pd.read_csv(clinical_csv_path)
    print(clinical_info.head())
    patient_labels = {}

    for _, row in clinical_info.iterrows():
        patient_id = str(row["patient_id"]).strip()
        is_significant = str(row["case_csPCa"]).strip().upper() == 'YES'
        label = 1 if is_significant else 0
        patient_labels[patient_id] = label

    print(f"Loaded labels for {len(patient_labels)} patients.")
    return patient_labels


def read_and_resize_mri(image_path, new_size=(128, 128)):
    """Read a single MRI image and resize it."""
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # If image is 3D, pick the middle slice
    if len(image_array.shape) > 2:
        mid_slice_idx = image_array.shape[0] // 2
        image_array = image_array[mid_slice_idx]

    resized_image = cv2.resize(image_array, new_size)
    return resized_image


def process_patient_images(patient_folder, new_size=(128, 128)):
    """Process all modalities for a single patient."""
    modalities = ['adc', 'cor', 'hbv', 'sag', 't2w']
    images = []

    patient_id = os.path.basename(patient_folder)

    for modality in modalities:
        image_files = [f for f in os.listdir(patient_folder) if modality in f]

        if not image_files:
            print(f"Warning: No {modality} image found for patient {patient_id}")
            continue

        image_path = os.path.join(patient_folder, image_files[0])
        resized_image = read_and_resize_mri(image_path, new_size)
        images.append(resized_image)

    if len(images) != 5:
        print(f"Warning: Patient {patient_id} does not have all 5 modalities. Skipping this patient.")
        return None

    patient_images = np.stack(images, axis=-1)  # Shape (height, width, 5)
    return patient_images


def extract_all_patients(mri_root_folder, patient_labels, new_size=(128, 128)):
    """Extract features and labels for all patients and cross-check patient IDs."""
    X = []
    y = []

    patient_ids_in_images = set()  # To track patient IDs found in MRI images
    missing_labels = []  # To track patients with missing labels
    inconsistent_mapping = []  # To track any inconsistent mappings

    for fold_name in os.listdir(mri_root_folder):
        fold_path = os.path.join(mri_root_folder, fold_name)

        if not os.path.isdir(fold_path):
            continue  # skip non-folder (e.g., .zip files)

        # Inside the fold (like fold0/picai_public_images_fold0)
        subfolders = os.listdir(fold_path)

        if not subfolders:
            continue  # no subfolder found, skip

        image_subfolder = os.path.join(fold_path, subfolders[0])

        if not os.path.isdir(image_subfolder):
            continue  # extra check, skip if not folder

        for patient_id in os.listdir(image_subfolder):
            patient_folder = os.path.join(image_subfolder, patient_id)

            if not os.path.isdir(patient_folder):
                continue

            # Cross-check if the patient_id in MRI folder exists in clinical data
            if patient_id not in patient_labels:
                print(f"Warning: No label found for patient {patient_id} in clinical data.")
                missing_labels.append(patient_id)
                continue

            # Process MRI images for this patient
            features = process_patient_images(patient_folder, new_size=new_size)

            if features is None:
                continue

            flattened_features = features.flatten()

            # Check if the patient_id's label matches the expected label
            label = patient_labels.get(patient_id)

            if label is None:
                print(f"Warning: No label found for patient {patient_id}. Skipping.")
                continue

            # Add the features and labels to the lists
            X.append(flattened_features)
            y.append(label)

            patient_ids_in_images.add(patient_id)

            # Check if there is a mismatch in mapping
            if patient_id not in patient_labels:
                inconsistent_mapping.append(patient_id)

    # Report missing labels (patients with no label)
    if missing_labels:
        print(f"\n{len(missing_labels)} patients have missing labels.")
        print("Missing label patient IDs:", missing_labels)

    # Report inconsistent mapping (patients that have MRI but not clinical data or vice versa)
    if inconsistent_mapping:
        print(f"\n{len(inconsistent_mapping)} patients have inconsistent mappings between MRI data and clinical labels.")
        print("Inconsistent mapping patient IDs:", inconsistent_mapping)

    # Report any patients from the clinical data that don't have corresponding MRI images
    missing_in_images = set(patient_labels.keys()) - patient_ids_in_images
    if missing_in_images:
        print(f"\n{len(missing_in_images)} patients from clinical data are missing MRI images.")
        print("Missing MRI patient IDs:", missing_in_images)

    X = np.array(X)
    y = np.array(y)

    print(f"Extracted {X.shape[0]} samples.")
    return X, y


def main():
    # Path to your clinical CSV file and MRI root folder
    clinical_csv_path = '/home/ibab/PycharmProjects/mlproject_data/picai_clinical_data.csv'
    mri_root_folder = '/home/ibab/PycharmProjects/mlproject_data/mri_images'

    # Load the clinical data
    print("Loading clinical data...")
    patient_labels = load_clinical_data(clinical_csv_path)

    # Extract features and labels for all patients
    print("Extracting MRI data for all patients...")
    X, y = extract_all_patients(mri_root_folder, patient_labels, new_size=(128, 128))

    # Check the shape of the extracted features and labels
    print(f"Extracted features shape: {X.shape}")
    print(f"Extracted labels shape: {y.shape}")

    # You can now save X and y to a file, or use them in your ML model
    # Example: save the data to .npy files
    np.save('X.npy', X)
    np.save('y.npy', y)
    print("Data saved to 'X.npy' and 'y.npy'")

if __name__ == '__main__':
    main()


