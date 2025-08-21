import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
from collections import defaultdict
from sklearn.model_selection import train_test_split

class SAMRamanDataset(Dataset):
    def __init__(
        self,
        spectral_data: List[np.ndarray],
        labels: List[np.ndarray],
    ):
        self.spectral_data = spectral_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.spectral_data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return data, label


class SAMRaman:
    def __init__(
        self,
        spectral_dirs: List[str],
        label_dirs: List[str],
        patient_intervals: List[int],
        use_pre_split: bool,
        train_data_path: List[str],
        train_labels_path: List[str],
        test_data_path: List[str],
        test_labels_path: List[str],
        batch_size: int = 16,
        seed: int = 42,
    ):
        self.spectral_dirs = spectral_dirs
        self.label_dirs = label_dirs
        self.patient_intervals = patient_intervals
        self.use_pre_split = use_pre_split
        self.train_data_path = train_data_path
        self.train_labels_path = train_labels_path
        self.test_data_path = test_data_path
        self.test_labels_path = test_labels_path
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(seed)

        self.label_map = self._get_label_mapping()
        self.inverse_map = {idx: l for l, idx in self.label_map.items()}
        self.train_set, self.val_set, self.test_set = self._get_splits()
        self._create_dataloaders()

    def _get_label_mapping(self):
        labels = []
        
        if self.use_pre_split:
            for f in self.train_labels_path:
                labels.extend(np.load(f))
            for f in self.test_labels_path:
                labels.extend(np.load(f))
        else:
            for f in self.label_dirs:
                labels.extend(np.load(f))

        label_map = {l: idx for idx, l in enumerate(np.unique(labels))}

        return label_map

    def _get_splits(self):
        train_set, val_set, test_set = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        
        print(self.use_pre_split)
        
        if self.use_pre_split:
            # use existing test/train split passed in by the user
            for train_data_path, train_labels_path, test_data_path, test_labels_path in zip(
                self.train_data_path, self.train_labels_path, self.test_data_path, self.test_labels_path
            ):
                test_set_data = np.load(test_data_path)
                test_set_labels = np.load(test_labels_path)
                train_and_validation_set_data = np.load(train_data_path)
                train_and_validation_set_labels = np.load(train_labels_path)
                
                # split train set into train and validation sets
                # ensure there is an even distribution of all labels in both sets
                train_set_data, validation_set_data, train_set_labels, validation_set_labels = train_test_split(train_and_validation_set_data, train_and_validation_set_labels, test_size=0.25, stratify=train_and_validation_set_labels, random_state=self.seed)
                
                test_set["data"] = test_set_data
                train_set["data"] = train_set_data
                val_set["data"] = validation_set_data
                
                for l in test_set_labels:
                    test_set["labels"].extend([self.label_map[l]])
                for l in train_set_labels:
                    train_set["labels"].extend([self.label_map[l]])
                for l in validation_set_labels:
                    val_set["labels"].extend([self.label_map[l]])
                
        else:
            # generate test/train/validation split
            for spectral_path, label_path, p_int in zip(
                self.spectral_dirs, self.label_dirs, self.patient_intervals
            ):
                spectra = np.load(spectral_path)
                labels = np.load(label_path)

                for l in np.unique(labels):
                    label_spectra = spectra[labels == l]
                    num_patients = len(label_spectra) // p_int
                    patient_idxs = np.arange(num_patients)
                    patient_idxs = np.random.permutation(patient_idxs)

                    train_split = 0.7
                    validation_split = 0.15
                    #test_split assumed to be anything not included in train or validation

                    train_split_end = int(train_split*num_patients)
                    validation_split_end = int((train_split+validation_split)*num_patients)

                    for i in patient_idxs[:train_split_end]:
                        train_set["data"].extend(
                            label_spectra[i * p_int : i * p_int + p_int]
                        )
                        train_set["labels"].extend([self.label_map[l]] * p_int)

                    for i in patient_idxs[train_split_end:validation_split_end]:
                        val_set["data"].extend(
                            label_spectra[i * p_int : i * p_int + p_int]
                        )
                        val_set["labels"].extend([self.label_map[l]] * p_int)

                    for i in patient_idxs[validation_split_end:]:
                        test_set["data"].extend(
                            label_spectra[i * p_int : i * p_int + p_int]
                        )
                        test_set["labels"].extend([self.label_map[l]] * p_int)
            
        print("train_set length: " + str(len(train_set["labels"])))
        print("val_set length: " + str(len(val_set["labels"])))
        print("test_set length: " + str(len(test_set["labels"])))
        print(train_set["labels"])
        return train_set, val_set, test_set

    def _create_dataloaders(self):
        train_dataset = SAMRamanDataset(
            self.train_set["data"], self.train_set["labels"]
        )
        val_dataset = SAMRamanDataset(self.val_set["data"], self.val_set["labels"])
        test_dataset = SAMRamanDataset(self.test_set["data"], self.test_set["labels"])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def _get_original_label(self, pred_idx):
        return self.inverse_map[pred_idx]
