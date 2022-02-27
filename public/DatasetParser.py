import json
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch
from torch.utils.data import Dataset


class DatasetParser(Dataset):
    def __init__(self, test,input_path=None,data=None):
        if input_path is None:
            self.input_path = "../output/challenge_1_train_dataset_full_year_full.json"
        else:
            self.input_path=input_path
        if data is None:
            self.data = self.load()
        else:
            self.data = data
        self.shift = False # to get the difference between 2 time steps
        self.min_max_scale= True
        self.test_size = 500
        self.input_variables = [ "x", "y", "z", "dx", "dy", "dz"]
        self.output_variables = ["x", "y", "z", "dx", "dy", "dz"]
        self.is_test_dataset = test
        self.sample_nb = len(self.data.keys())

        print(f"-- {'test'if test else 'train'} dataset loaded--")
        print(f"size: {self.__len__()}")
        print(f"normalize: {'yes' if self.min_max_scale else 'no'}")

    def __len__(self):
        if self.is_test_dataset:
            return self.test_size
        else:
            return self.sample_nb - self.test_size

    def __getitem__(self, item):
            if self.is_test_dataset:
                return self.create_input_target(item, start_index=self.sample_nb - self.test_size)
            else:
                return self.create_input_target(item)


    def create_input_target(self, item, start_index=0,analytical=False):
        series_length = len(self.parse_data(0, "x",analytical=analytical))
        X = np.zeros((len(self.input_variables),1))
        y = np.zeros((len(self.output_variables), series_length - 1))
        for var_nb, var in enumerate(self.input_variables):
            X[var_nb,0] = self.get_data_preprocessed(item + start_index, var,analytical=analytical)[0]
        for var_nb, var in enumerate(self.output_variables):
            y[var_nb][:] = self.get_data_preprocessed(item + start_index, var,analytical=analytical)[1:]
        return torch.t(torch.from_numpy(X)), torch.t(torch.from_numpy(y))



    def load(self):
        with open(self.input_path, 'r')as f:
            return json.loads(f.read())



    def plot_first_orbits(self, orbit_type):
        plt.figure()
        if orbit_type == "circular":
            variables = {"a": "km", "ex": "", "ey": "", "i": "deg", "raan": "deg", "av": "deg"}
        elif orbit_type == "cartesian":
            variables = {"x": "m", "y": "m", "z": "m", "dx": "m/s", "dy": "m/s", "dz": "m/s"}
        else:
            raise Exception

        for i, variable in enumerate(variables.keys()):
            plt.subplot(3, 2, i + 1)
            for sample in range (5):
                plt.plot(
                    self.get_data_preprocessed(sample, variable)
                )
            plt.title(f"{variable} ({variables[variable]})")
        plt.tight_layout()
        plt.show()


    def get_data_preprocessed(self, sample, variable,analytical=False):
        scale_factor={"x":1200,"y":7000,"z":7000,"dx":1.400,"dy":8.,"dz":8.}
        if self.min_max_scale and variable in scale_factor.keys():
            scale=scale_factor[variable]
        else:
            scale=1

        if self.shift:
            return np.diff(self.parse_data(sample,variable,analytical=analytical))/scale
        else:
            return self.parse_data(sample,variable,analytical=analytical)/scale

    def parse_data(self, sample, variable,analytical=False):
        return np.array(self.data[str(sample)]["eph" if not analytical else "eph_ana"][variable])

    def dickey_fuller(self, sample):
        for variable in self.input_variables:
            p_value = sm.tsa.stattools.adfuller(np.diff(np.array(self.parse_data(sample, variable))))[1]
            print(f"{variable} : {p_value}")

if __name__ == '__main__':
    DatasetParser(True).plot_first_orbits("circular")
    print("done")
