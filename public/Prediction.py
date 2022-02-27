import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from SimpleNet import DNN
from DatasetParser import DatasetParser
from SubmissionEncoder import SubmissionEncoder

if __name__ == '__main__':
        output_folder="../output/"
        model_name="../output/2000.pt"
        evaluation_dataset_path="../output/challenge_1_test_dataset_full_year_eval.json"
        output_file_name="prediction.csv"

        model = torch.load(os.path.join(output_folder,model_name))
        evaluation_dataset = DatasetParser(True,evaluation_dataset_path)
        evaluation_dataset.test_size=500
        evaluation_loader = DataLoader(
            evaluation_dataset, batch_size=len(evaluation_dataset), drop_last=False
        )
        with torch.no_grad():
            for (X, y) in tqdm(evaluation_loader):
                pred = model(X.float())

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            encoder=SubmissionEncoder(output_folder,output_file_name)
            encoder.set_prediction(pred.numpy())
            encoder.encode()