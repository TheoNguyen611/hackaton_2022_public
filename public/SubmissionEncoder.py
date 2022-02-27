import csv
import os
import numpy as np

class SubmissionEncoder:
    def __init__(self, output_path, output_file_name):
        self.output_path = output_path
        self.output_file_name = output_file_name

        self.separator = ";"
        self.value_column_name = "predicted"

    def set_prediction_path(self, prediction_path):
        self.prediction = np.load(prediction_path)

    def set_prediction(self, prediction):
        self.prediction = prediction

    def encode(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        output_file = open(os.path.join(self.output_path + self.output_file_name), "w")
        writer = csv.writer(output_file)
        writer.writerow(["id", self.value_column_name])
        for sample_index, sample in enumerate(self.prediction):
            for time_index, time in enumerate(sample):
                for variable_index, value in enumerate(time):
                    writer.writerow([
                        str(sample_index) + "_" + str(time_index) + "_" + str(variable_index),
                        value])
        output_file.close()


if __name__ == '__main__':
    encoder = SubmissionEncoder("../../output/simpleNet/6_variables/eval/", "submission_simple_net_MSE_500.csv")
    encoder.set_prediction_path("../../output/simpleNet/6_variables/eval/y_pred.npy")
    encoder.encode()
