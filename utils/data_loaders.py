import numpy as np
import os
import glob
from pathlib import Path
import nibabel as nib
import torch
import json

# MONAI transforms used for preprocessing and augmentation
from monai.transforms import(
    Compose,
    Lambdad,
    LoadImaged,
    EnsureChannelFirstd,
    SqueezeDimd,
    RandSpatialCropSamplesd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    Resized,
    CropForegroundd,
    CenterSpatialCropd,
    RandZoomd,
    SpatialPadd,
    Invertd,
    Transposed
)
from monai.transforms import Compose, Transform, MapTransform
from monai.transforms import ScaleIntensityRangePercentiles
from monai.data import DataLoader, Dataset, CacheDataset, PatchDataset


class RandSpatialCropSamplesdWithMinNonZero_3d(RandSpatialCropSamplesd):
    """
    Extension of MONAI's RandSpatialCropSamplesd for 2D and 3D data.

    This transform filters out sampled patches that contain too little
    non-background information (based on a percentage threshold).
    """

    def __init__(self, percentage: int = 0.05, *args, **kwargs):
        # Initialize the parent RandSpatialCropSamplesd
        super().__init__(*args, **kwargs)
        # Minimum percentage of non-zero (non-background) voxels required
        self.percentage = percentage

    def __call__(self, data, lazy=False):
        # Store only valid cropped samples
        valid_samples = []

        # Keep sampling until enough valid samples are obtained
        while True:
            # Generate candidate samples using the base class
            samples = super().__call__(data)

            # Check each sample for sufficient non-zero content
            for sample in samples:
                flag = True
                for key in list(self.keys):
                    # Count voxels above background threshold
                    if np.sum(sample[key] > (-1 + 10e-5)) < self.percentage * np.prod(sample[key].shape):
                        flag = False
                if flag:
                    valid_samples.append(sample)

            # Stop once we have at least the required number of samples
            if len(valid_samples) >= len(samples):
                break

        # Keep only as many samples as originally requested
        valid_samples = valid_samples[:len(samples)]

        # Safety check: avoid returning an empty list
        if not valid_samples:
            raise ValueError("No valid samples found. Consider adjusting the min_nonzero parameter.")

        return valid_samples


def data_loader_GA(opt, loaders_dict, test_infer=False):
    """
    Main entry point for preparing data loaders.

    Splits the dataset into train / val / test,
    applies preprocessing, and builds PyTorch DataLoaders.
    """

    # List all files in the input directory
    PIDs_ALL_ = os.listdir(opt.input_dir)

    # Keep only CT volumes
    PIDs_ALL = []
    for img in PIDs_ALL_:
        if '_CT.nii' in img:
            PIDs_ALL.append(img)

    # Sort and shuffle deterministically
    PIDs_ALL = sorted(PIDs_ALL)
    np.random.seed(29100)
    np.random.shuffle(PIDs_ALL)

    # Initial assignment
    PIDs_train_A = PIDs_ALL
    PIDs_train_B = PIDs_train_A.copy()

    # Dataset split intervals: train / val / test
    intervals = [[0, 9], [9, 11], [11, 15]]
    indexes = ['', '_val', '_test']

    fnames_A = []
    fnames_B = []
    fnames_dictionary = []

    # Build file lists for each split
    for i in range(3):
        if i == 0:
            PIDs_train_A = PIDs_ALL[intervals[i][0]:intervals[i][1]]
        else:
            PIDs_train_A = PIDs_ALL[intervals[i][0]:intervals[i][1]]

        PIDs_train_B = PIDs_train_A.copy()

        # File paths for target (CT) and source (T2)
        fnames_B.append([opt.input_dir / PID for PID in PIDs_train_A])
        fnames_A.append([opt.input_dir / PID.replace('CT', 'T2') for PID in PIDs_train_B])

        # MONAI-style dictionary for paired loading
        fnames_dictionary.append(
            [{"SRC": img1, "TGT": img2} for (img1, img2) in zip(fnames_A[-1], fnames_B[-1])]
        )

    # Build loaders
    loaders = base_DL(opt, loaders_dict, fnames_dictionary, test_infer)

    return loaders


def base_DL(opt, loaders_dict, fnames_dictionary, test_infer):
    """
    Handles preprocessing, saving cropped patches to disk,
    and constructing PyTorch DataLoaders.
    """

    indexes = ['', '_val', '_test']
    NUM_SAMPLES_MAX = opt.NUM_SAMPLES_MAX

    # -------- TRAIN TRANSFORMS --------
    pre_transforms = []
    for pre_transfrom in loaders_dict['train_pre_transform']:
        pre_transforms.append(eval(pre_transfrom))
    train_transforms = Compose(pre_transforms)

    # -------- VAL TRANSFORMS --------
    pre_transforms = []
    for pre_transfrom in loaders_dict['val_pre_transform']:
        pre_transforms.append(eval(pre_transfrom))
    val_transforms = Compose(pre_transforms)

    # -------- TEST TRANSFORMS --------
    pre_transforms = []
    for pre_transfrom in loaders_dict['test_pre_transform']:
        pre_transforms.append(eval(pre_transfrom))
    test_transforms = Compose(pre_transforms)

    # Skip train split if in test inference mode
    range_start = 2 * test_infer

    headers = {}

    # Loop over train / val / test splits
    for i in range(range_start, 3):

        # Skip if already processed
        if (Path(opt.input_dir_processed) / ('A' + indexes[i])).exists():
            continue

        # Output directories
        outdir_A = os.path.join(opt.input_dir_processed, 'A' + indexes[i])
        outdir_B = os.path.join(opt.input_dir_processed, 'B' + indexes[i])
        os.makedirs(outdir_A, exist_ok=True)
        os.makedirs(outdir_B, exist_ok=True)

        # Optional extra modalities
        extra_folders = ['C', 'D', 'E', 'F']
        outdirs = []

        if loaders_dict['imports'].get('quantity'):
            for f_ind in range(loaders_dict['imports']['quantity'] - 1):
                outdirs.append(os.path.join(opt.input_dir_processed, extra_folders[f_ind] + indexes[i]))
                os.makedirs(outdirs[-1], exist_ok=True)

        # Process each subject
        for j in range(len(fnames_dictionary[i])):
            print(fnames_dictionary[i][j])

            # -------- TRAIN --------
            if i == 0:
                transformed_image = train_transforms(fnames_dictionary[i][j])

                fname_ref = transformed_image[0]['SRC_meta_dict']['filename_or_obj']
                fname_tgt = transformed_image[0]['TGT_meta_dict']['filename_or_obj']

                fnames = []

                # Load headers for extra modalities
                if loaders_dict['imports'].get('extra_names'):
                    for f_ind in loaders_dict['imports']['extra_names']:
                        fnames.append(transformed_image[0][f"{f_ind}_meta_dict"]['filename_or_obj'])
                        if not headers.get(fnames[-1]):
                            headers[fnames[-1]] = nib.load(fnames[-1]).header

                # Load headers for SRC and TGT
                if not headers.get(fname_ref):
                    headers[fname_ref] = nib.load(fname_ref).header
                if not headers.get(fname_tgt):
                    headers[fname_tgt] = nib.load(fname_ref).header

                # Save all sampled patches
                for j in range(len(transformed_image)):
                    fname_out = os.path.join(outdir_A, fname_ref.split('/')[-1].split('.')[0] + ('_sample%.3d.nii.gz' % j))
                    nib.save(nib.Nifti1Image(transformed_image[j]['SRC'][:, :, :].numpy(), None, headers[fname_ref]), fname_out)

                    fname_out = os.path.join(outdir_B, fname_tgt.split('/')[-1].split('.')[0] + ('_sample%.3d.nii.gz' % j))
                    nib.save(nib.Nifti1Image(transformed_image[j]['TGT'][:, :, :].numpy(), None, headers[fname_tgt]), fname_out)

                    if loaders_dict['imports'].get('quantity'):
                        for f_ind in range(loaders_dict['imports']['quantity'] - 1):
                            fname_out = os.path.join(
                                outdirs[f_ind],
                                fnames[f_ind].split('/')[-1].split('.')[0] + ('_sample%.3d.nii.gz' % j)
                            )
                            nib.save(
                                nib.Nifti1Image(
                                    transformed_image[j][loaders_dict['imports']['extra_names'][f_ind]][:, :, :].numpy(),
                                    None,
                                    headers[fnames[f_ind]]
                                ),
                                fname_out
                            )

            # -------- VAL / TEST --------
            else:
                # if validation / else test
                if i == 1:
                    transformed_image = val_transforms(fnames_dictionary[i][j])
                else:
                    transformed_image = test_transforms(fnames_dictionary[i][j])

                # Case: multiple samples
                if isinstance(transformed_image, list):
                    fname_ref = transformed_image[0]['SRC_meta_dict']['filename_or_obj']
                    fname_tgt = transformed_image[0]['TGT_meta_dict']['filename_or_obj']

                    fnames = []

                    if loaders_dict['imports'].get('extra_names'):
                        for f_ind in loaders_dict['imports']['extra_names']:
                            fnames.append(transformed_image[0][f"{f_ind}_meta_dict"]['filename_or_obj'])
                            if not headers.get(fnames[-1]):
                                headers[fnames[-1]] = nib.load(fnames[-1]).header

                    if not headers.get(fname_ref):
                        headers[fname_ref] = nib.load(fname_ref).header
                    if not headers.get(fname_tgt):
                        headers[fname_tgt] = nib.load(fname_ref).header

                    for j in range(len(transformed_image)):
                        fname_out = os.path.join(outdir_A, fname_ref.split('/')[-1].split('.')[0] + ('_sample%.3d.nii.gz' % j))
                        nib.save(nib.Nifti1Image(transformed_image[j]['SRC'][:, :, :].numpy(), None, headers[fname_ref]), fname_out)

                        fname_out = os.path.join(outdir_B, fname_tgt.split('/')[-1].split('.')[0] + ('_sample%.3d.nii.gz' % j))
                        nib.save(nib.Nifti1Image(transformed_image[j]['TGT'][:, :, :].numpy(), None, headers[fname_tgt]), fname_out)

                        if loaders_dict['imports'].get('quantity'):
                            for f_ind in range(loaders_dict['imports']['quantity'] - 1):
                                fname_out = os.path.join(
                                    outdirs[f_ind],
                                    fnames[f_ind].split('/')[-1].split('.')[0] + ('_sample%.3d.nii.gz' % j)
                                )
                                nib.save(
                                    nib.Nifti1Image(
                                        transformed_image[j][loaders_dict['imports']['extra_names'][f_ind]][:, :, :].numpy(),
                                        None,
                                        headers[fnames[f_ind]]
                                    ),
                                    fname_out
                                )

                # Case: single center crop
                else:
                    fname_ref = transformed_image['SRC_meta_dict']['filename_or_obj']
                    fname_tgt = transformed_image['TGT_meta_dict']['filename_or_obj']

                    fnames = []

                    if loaders_dict['imports'].get('extra_names'):
                        for f_ind in loaders_dict['imports']['extra_names']:
                            fnames.append(transformed_image[f"{f_ind}_meta_dict"]['filename_or_obj'])
                            if not headers.get(fnames[-1]):
                                headers[fnames[-1]] = nib.load(fnames[-1]).header

                    if not headers.get(fname_ref):
                        headers[fname_ref] = nib.load(fname_ref).header
                    if not headers.get(fname_tgt):
                        headers[fname_tgt] = nib.load(fname_ref).header

                    fname_out = os.path.join(outdir_A, fname_ref.split('/')[-1].split('.')[0] + '_sCenter.nii.gz')
                    nib.save(nib.Nifti1Image(transformed_image['SRC'][:, :, :].numpy(), None, headers[fname_ref]), fname_out)

                    fname_out = os.path.join(outdir_B, fname_tgt.split('/')[-1].split('.')[0] + '_sCenter.nii.gz')
                    nib.save(nib.Nifti1Image(transformed_image['TGT'][:, :, :].numpy(), None, headers[fname_tgt]), fname_out)

                    if loaders_dict['imports'].get('quantity'):
                        for f_ind in range(loaders_dict['imports']['quantity'] - 1):
                            fname_out = os.path.join(
                                outdirs[f_ind],
                                fnames[f_ind].split('/')[-1].split('.')[0] + '_sCenter.nii.gz'
                            )
                            nib.save(
                                nib.Nifti1Image(
                                    transformed_image[loaders_dict['imports']['extra_names'][f_ind]][:, :, :].numpy(),
                                    None,
                                    headers[fnames[f_ind]]
                                ),
                                fname_out
                            )

    # -------- ON-THE-FLY TRANSFORMS --------
    fly_transforms = []
    for fly_transform in loaders_dict['fly_transform']:
        fly_transforms.append(eval(fly_transform))
    fly_transforms = Compose(fly_transforms)

    sub_datas = ['', '_val', '_test']
    loaders = [None, None, None]
    
    # not useful for these experiments
    extra_folders = ['C', 'D', 'E', 'F']
    outdirs = []

    if loaders_dict['imports'].get('quantity'):
        for f_ind in range(loaders_dict['imports']['quantity'] - 1):
            outdirs.append(extra_folders[f_ind])

    # Build DataLoaders
    for sub_ind in range(range_start, 3):
        f_name_train_A = glob.glob(str(opt.input_dir_processed / ('A' + sub_datas[sub_ind]) / '*.nii.gz'))
        f_name_train_B = glob.glob(str(opt.input_dir_processed / ('B' + sub_datas[sub_ind]) / '*.nii.gz'))

        f_name_trains = []
        for dir_i in outdirs:
            f_name_trains.append(glob.glob(str(opt.input_dir_processed / (dir_i + sub_datas[sub_ind]) / '*.nii.gz')))

        data_loader_A_B = A_B(
            f_name_train_A,
            f_name_train_B,
            others_list=f_name_trains,
            testing=sub_ind != 0,
            paired=opt.paired,
            on_B=loaders_dict['imports']['on_B'] if ('on_B' in loaders_dict['imports'].keys()) else True,
            HR=opt.HR
        )

        loaders[sub_ind] = DataLoader(
            data_loader_A_B,
            batch_size=opt.batch_size if sub_ind == 0 else 1,
            shuffle=sub_ind == 0,
            num_workers=4
        )

    return loaders


class A_B(Dataset):
    """
    Custom PyTorch Dataset for loading paired or unpaired
    SRC (A) and TGT (B) NIfTI volumes.
    """

    def __init__(self, listA, listB, others_list=None, transform=None, testing=False, paired=False, on_B=True, HR=False):
        self.HR = HR
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        print(testing)

        self.imagesA = sorted(listA)
        self.imagesB = sorted(listB)

        self.on_B = on_B
        self.other_dirs_bool = False

        # Handle additional modalities
        if len(others_list):
            self.other_dirs_bool = True
            self.other_dirs = []
            for i in range(len(others_list)):
                print('here', len(self.imagesA), len(others_list[i]))
                self.other_dirs.append(sorted(others_list[i]))

        self.transform = transform
        self.testing = testing
        self.paired = paired

    def __len__(self):
        return len(self.imagesA)

    def __getitem__(self, idx):
        dict_out = {}
        dict_out['TGT_meta_dict'] = {}
        dict_out['SRC_meta_dict'] = {}

        # Load source image
        imageA = nib.load(self.imagesA[idx]).get_fdata()
        dict_out['SRC_meta_dict']['filename_or_obj'] = self.imagesA[idx]

        # Load target image (paired or random)
        if self.testing or self.paired:
            new_id = idx
            imageB = nib.load(self.imagesB[idx]).get_fdata()
            dict_out['TGT_meta_dict']['filename_or_obj'] = self.imagesB[idx]
        else:
            new_id = np.random.randint(len(self.imagesB))
            imageB = nib.load(self.imagesB[new_id]).get_fdata()
            dict_out['TGT_meta_dict']['filename_or_obj'] = self.imagesB[new_id]

        # Load extra modalities if present
        images_other = []
        if self.other_dirs_bool:
            id_others = new_id if self.on_B else idx
            for list_ind in self.other_dirs:
                images_other.append(nib.load(list_ind[id_others]).get_fdata())
            imageA = np.array([imageA] + images_other)

        # Data augmentation (flips)
        if not self.testing:
            if np.random.random() > 0.5:
                imageA = np.flip(imageA, axis=0)
                imageB = np.flip(imageB, axis=0)
            if np.random.random() > 0.5:
                imageA = np.flip(imageA, axis=1)
                imageB = np.flip(imageB, axis=1)
            if np.random.random() > 0.5 and len(imageB.shape) == 3:
                imageA = np.flip(imageA, axis=2)
                imageB = np.flip(imageB, axis=2)

        # Ensure channel dimension
        if not self.other_dirs_bool:
            imageA = imageA[np.newaxis]
        imageB = imageB[np.newaxis]

        # Convert to torch tensors
        imageA = torch.tensor(imageA.copy(), dtype=torch.float32)
        imageB = torch.tensor(imageB.copy(), dtype=torch.float32)

        # Handle 2D case
        if imageA.shape[2] == 1:
            imageA = imageA.permute((0, 3, 1, 2))
            imageB = imageB.permute((0, 3, 1, 2))

        # Optional HR processing
        if self.HR:
            imageA = imageA[:, ::2, ::2]
            imageA = self.up(imageA.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        dict_out['SRC'] = imageA
        dict_out['TGT'] = imageB

        return dict_out
