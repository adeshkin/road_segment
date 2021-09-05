import yaml
import os
import copy
import random
import numpy as np
import pandas as pd
import wandb
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold

from dataset import RoadSegment, RoadSegmentTest, RoadSegmentFolds
from utils import calculate_accuracy, get_transform, calculate_auc_score, set_seed


class Runner:
    def __init__(self, params):
        self.params = params
        self.transforms = get_transform()

        test_df = pd.read_csv(self.params['test_filepath'])
        dataset_test = RoadSegmentTest(test_df, self.params['image_dir'], self.transforms['val'])
        self.data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

        self.device = torch.device(params['device'])
        self.criterion = nn.BCEWithLogitsLoss()

        self.checkpoints_dir = params['checkpoint_dir']
        self.submissions_dir = params['submission_dir']
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.submissions_dir, exist_ok=True)

    def train(self):
        self.model.train()

        epoch_metrics = dict()
        epoch_metrics['loss'] = 0.0
        epoch_metrics['acc'] = 0.0
        epoch_metrics['auc'] = 0.0
        for images, labels in tqdm(self.data_loaders['train']):
            images = images.to(self.device)
            labels = labels.to(self.device).float().view(-1, 1)

            pred_labels = self.model(images)
            loss = self.criterion(pred_labels, labels)
            accuracy = calculate_accuracy(pred_labels, labels)
            auc = calculate_auc_score(pred_labels.cpu().detach().numpy(), labels.cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_metrics['loss'] += loss.cpu().detach()
            epoch_metrics['acc'] += accuracy
            epoch_metrics['auc'] += auc

        for m in epoch_metrics:
            epoch_metrics[m] = epoch_metrics[m] / len(self.data_loaders['train'])

        return epoch_metrics

    def eval(self):
        epoch_metrics = dict()
        epoch_metrics['loss'] = 0.0
        epoch_metrics['acc'] = 0.0
        epoch_metrics['auc'] = 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.data_loaders['val']):
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                pred_labels = self.model(images)
                loss = self.criterion(pred_labels, labels)
                accuracy = calculate_accuracy(pred_labels, labels)
                auc = calculate_auc_score(pred_labels.cpu().detach().numpy(), labels.cpu().numpy())

                epoch_metrics['loss'] += loss.cpu().detach()
                epoch_metrics['acc'] += accuracy
                epoch_metrics['auc'] += auc

        for m in epoch_metrics:
            epoch_metrics[m] = epoch_metrics[m] / len(self.data_loaders['val'])

        return epoch_metrics

    def predict(self):
        self.model.eval()
        results = []
        with torch.no_grad():
            for image, image_id in tqdm(self.data_loader_test):
                image = image.to(self.device)
                pred_label = torch.sigmoid(self.model(image))
                pred_label = pred_label.cpu().item()
                row_dict = {}
                row_dict["Image_ID"] = image_id[0]
                row_dict["Target"] = pred_label
                results.append(row_dict)

        df = pd.DataFrame(results)
        df.to_csv(f"{self.submissions_dir}/{self.params['arch']}_{self.run_name}.csv", index=False)

    def run_folds(self):
        set_seed()

        df = pd.read_csv(self.params['train_filepath'])
        X = df['Image_ID']
        y = df['Target']
        skf = StratifiedKFold(n_splits=self.params['num_splits'], shuffle=True, random_state=42)

        for k_fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            run = wandb.init(project=self.params['project_name'], config=self.params, reinit=True)
            self.run_name = run.name
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            dataset_train = RoadSegmentFolds(X_train, y_train, self.params['image_dir'], self.transforms['train'])
            dataset_val = RoadSegmentFolds(X_test, y_test, self.params['image_dir'], self.transforms['val'])

            self.data_loaders = {'train': DataLoader(dataset_train,
                                                     batch_size=params['batch_size'],
                                                     shuffle=True,
                                                     num_workers=4),
                                 'val': DataLoader(dataset_val,
                                                   batch_size=params['batch_size'],
                                                   shuffle=False,
                                                   num_workers=4)}

            if 'resnet' in self.params['arch']:
                self.model = torchvision.models.__dict__[self.params['arch']](pretrained=True)
                self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            elif 'efficientnet' in self.params['arch']:
                self.model = EfficientNet.from_pretrained(self.params['arch'])
                self.model._fc = nn.Linear(in_features=self.model._fc.in_features, out_features=1, bias=True)

            self.model = self.model.to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=params['lr_scheduler']['step_size'],
                                                                gamma=params['lr_scheduler']['gamma'])
            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_auc = 0.

            for epoch in range(self.params['num_epochs']):
                train_metrics = self.train()
                self.lr_scheduler.step()
                val_metrics = self.eval()

                logs = {f'train': train_metrics,
                        f'val': val_metrics}

                wandb.log(logs, step=epoch)

                current_val_auc = val_metrics['auc']
                if current_val_auc > best_auc:
                    best_auc = current_val_auc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            self.model.load_state_dict(best_model_wts)
            torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/{self.run_name}.pth")
            self.predict()
            run.finish()

    def predict_ensemble(self):
        models = []
        for arch, path in zip(['resnet18', 'resnet18', 'resnet18', 'resnet18', 'resnet18'],
                              ['atomic-moon-1', 'fancy-jazz-2', 'balmy-frost-3', 'vocal-gorge-4', 'gentle-eon-5']):
            if 'resnet' in arch:
                model = torchvision.models.__dict__[arch]()
                model.fc = nn.Linear(model.fc.in_features, 1)
            elif 'efficientnet' in arch:
                model = EfficientNet.from_pretrained(arch)
                model._fc = nn.Linear(in_features=model._fc.in_features, out_features=1, bias=True)

            model.load_state_dict(torch.load(f"{self.checkpoints_dir}/{path}.pth"))
            model.to(self.device)
            model.eval()
            models.append(model)

        results = []
        with torch.no_grad():
            for image, image_id in tqdm(self.data_loader_test):
                image = image.to(self.device)
                if self.params['ensemble_mode'] == 'mean':
                    ensemble_label = 0.0
                elif self.params['ensemble_mode'] == 'geom':
                    ensemble_label = 1.0
                count = 0
                for model in models:
                    pred_label = torch.sigmoid(model(image))
                    pred_label = pred_label.cpu().item()
                    if self.params['ensemble_mode'] == 'mean':
                        ensemble_label += pred_label
                    elif self.params['ensemble_mode'] == 'geom':
                        ensemble_label *= pred_label
                    count += 1

                if self.params['ensemble_mode'] == 'mean':
                    prediction = ensemble_label / count
                elif self.params['ensemble_mode'] == 'geom':
                    prediction = np.power(ensemble_label, 1./count)

                row_dict = {}
                row_dict["Image_ID"] = image_id[0]
                row_dict["Target"] = prediction
                results.append(row_dict)

        df = pd.DataFrame(results)
        df.to_csv(f"{self.submissions_dir}/ensemble_5x_resnet18_last_dance_{self.params['ensemble_mode']}.csv", index=False)


if __name__ == '__main__':
    with open('./configs/default.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    runner = Runner(params)
    # runner.run_folds()
    # runner.run()
    runner.predict_ensemble()
