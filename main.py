import nibabel as nib
import numpy as np
import argparse


def extract_nifti_data(img_path: str) -> np.ndarray:
    """Read the data from the nifti file and output it as a numpy array."""
    img = nib.load(img_path)
    data = img.get_fdata()
    return data


def threshold_data(data: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold the data to only keep the voxels with a value above the threshold."""
    thresholded_data = data[data > threshold]
    return thresholded_data


def get_mean(data: np.ndarray) -> float:
    """Get the mean of the array."""
    return np.mean(data)


def main(img_path: str, threshold: float) -> None:
    """Print the average of the thresholded voxel values."""
    data = extract_nifti_data(img_path)
    thresholded_data = threshold_data(data, threshold)
    average = get_mean(thresholded_data)
    print(f"Mean of voxels values above {threshold}: {average}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", dest="img_path", type=str, help="Path to the nifti file.")
    parser.add_argument(
        "--thres", dest="threshold", type=float, help="Threshold to apply on the image."
    )
    args = parser.parse_args()

    main(args.img_path, args.threshold)

import pytest


def create_sample_nifti(mean: float = 4.0) -> nib.Nifti1Image:
        # Example usage with a sample NIfTI file and threshold
    from nibabel import Nifti1Image
    import nibabel as nib
    import numpy as np
    from numpy import random
    data = random.rand(10, 10, 10)
    data_mean = np.mean(data)

    sample_data = Nifti1Image(
        data - data_mean + mean, affine=np.eye(4)
    )  # Create a sample NIfTI image
    return sample_data

def test_main(capsys, tmp_path):
    """Test the main function with a sample nifti file."""
    # Example usage with a sample NIfTI file and threshold
    import nibabel as nib
    sample_data = create_sample_nifti(mean=4.0)
    sample_data_path = tmp_path / "sample.nii.gz"
    nib.save(sample_data, sample_data_path)
    # Example threshold value
    
    main(sample_data_path, 0.0)
    captured = capsys.readouterr()
    assert "Mean of voxels values above 0.0: 4.0\n" == captured.out


def test_get_mean(tmp_path):
    """Test the get_mean function."""
    import numpy as np
    data = np.random.rand(10)
    data-= np.mean(data)
    data+= 4.0
    mean = get_mean(data)
    np.testing.assert_almost_equal(mean, 4.0)

def test_threshold_data():
    """Test the threshold_data function."""
    import numpy as np
    data = np.random.rand(10) + 4.0
    thresholded_data = threshold_data(data, 4.0)
    assert sum(thresholded_data > 4.0) == 10

