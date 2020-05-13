import h5py
import numpy as np
import ipdb

# from mpi4py import MPI
from torch.utils.data import Dataset


class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        transform=None,
        episode_length=50,
        training_type="selfsupervised",
        n_time_steps=1,
        action_dim=4
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = filename_list
        self.transform = transform
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim

        self._config_checks()

    def __len__(self):
        # Ensures that there is space for action, ee_yaw_next
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):

        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index][:-8]

        file_number, filename = self._parse_filename(filename)
        unpaired_filename = self._get_unpaired_filename(file_number, filename)
        unpaired_idx = np.random.randint(self.episode_length - 1)

        # TODO: update to newest
        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            list_index,
            unpaired_filename,
            dataset_index,
            unpaired_idx,
        )
        return sample

    def _get_single(
        self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            force = dataset["ee_forces_continuous"][dataset_index]
            # print("FORCE SHAPE", force.shape, force.ndim)

            if force.ndim == 1:
                force = force.reshape((1, 6))

            # force = np.tile(force, (10, 1))
            force = np.repeat(force, 10, axis=0)
            proprio = dataset["proprio"][dataset_index][:8]
            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]
            unpaired_force = np.tile(unpaired_force, (50, 1))

            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_depth = depth

            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "force_cp": force,
                "force_fp": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                ).astype(np.float),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
                "force_fut": dataset["ee_forces_continuous"][dataset_index + 1],
                "contact_now": np.array(
                    [dataset["contact"][dataset_index].sum() > 0]
                ).astype(np.float),
                "dataset_index": np.array([dataset_index]),
                "list_index": np.array([list_index]),
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_unpaired_filename(self, file_number, filename):
        """ Finds the unpaired file"""
        #todo: change to newest with random thresholding
        if file_number < 10:
            comp_number = 19
            unpaired_filename = filename + str(comp_number) + "_1000.h5"
            while unpaired_filename not in self.dataset_path:
                comp_number += -1
                unpaired_filename = filename + str(comp_number) + "_1000.h5"
        else:
            comp_number = 0
            unpaired_filename = filename + str(comp_number) + "_1000.h5"
            while unpaired_filename not in self.dataset_path:
                comp_number += 1
                unpaired_filename = filename + str(comp_number) + "_1000.h5"

        return unpaired_filename

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        #todo: no longer need other training types
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )
