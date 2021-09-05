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


from dataset import RoadSegment, RoadSegmentTest
from utils import calculate_accuracy, get_transform, read_data, calculate_auc_score


class Runner:
    def __init__(self, params):
        self.params = params
        self.run_name = None

        train_df, val_df, test_df = read_data(params)

        transforms = get_transform()

        dataset_train = RoadSegment(train_df, params['image_dir'], transforms['train'])
        dataset_val = RoadSegment(val_df, params['image_dir'], transforms['val'])
        dataset_test = RoadSegmentTest(test_df, params['image_dir'], transforms['val'])

        self.data_loaders = {'train': DataLoader(dataset_train,
                                                 batch_size=params['batch_size'],
                                                 shuffle=True,
                                                 num_workers=4),

                             'val': DataLoader(dataset_val,
                                               batch_size=params['batch_size'],
                                               shuffle=False,
                                               num_workers=4),

                             'test': DataLoader(dataset_test,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1)}

        self.model = torchvision.models.__dict__[params['arch']](pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        self.device = torch.device(params['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=params['lr_scheduler']['step_size'],
                                                            gamma=params['lr_scheduler']['gamma'])
        self.criterion = nn.BCEWithLogitsLoss()

        self.checkpoints_dir = params['checkpoint_dir']
        self.submissions_dir = params['submission_dir']

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
        #PATH = f"{self.checkpoints_dir}/{self.params['model_filename']}.pth"
        #self.model.load_state_dict(torch.load(PATH))
        #self.model.to(self.device)
        self.model.eval()
        results = []
        with torch.no_grad():
            for image, image_id in tqdm(self.data_loaders['test']):
                image = image.to(self.device)
                pred_label = torch.sigmoid(self.model(image))
                pred_label = pred_label.cpu().item()
                row_dict = {}
                row_dict["Image_ID"] = image_id[0]
                row_dict["Target"] = pred_label
                results.append(row_dict)

        df = pd.DataFrame(results)
        df.to_csv(f"{self.submissions_dir}/{self.params['arch']}_{self.run_name}.csv", index=False)

    def predict_ensemble(self):
        models = []
        for arch, path in zip(['resnet18', 'resnet18', 'resnet18'],
                              ['major-pine-19', 'fiery-sound-29', 'chocolate-waterfall-31']):
            model = torchvision.models.__dict__[arch](pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 1)
            model.load_state_dict(torch.load(f"{self.checkpoints_dir}/{path}.pth"))
            model.to(self.device)
            model.eval()
            models.append(model)

        results = []
        with torch.no_grad():
            for image, image_id in tqdm(self.data_loaders['test']):
                image = image.to(self.device)
                ensemble_label = []
                ensemble_label = 1.0
                count = 0
                for model in models:
                    pred_label = torch.sigmoid(model(image))
                    pred_label = pred_label.cpu().item()
                    ensemble_label.append(pred_label)
                    # ensemble_label *= pred_label
                    count += 1

                prediction = sum(ensemble_label) / len(ensemble_label)
                # prediction = np.power(ensemble_label, 1./count)
                row_dict = {}
                row_dict["Image_ID"] = image_id[0]
                row_dict["Target"] = prediction
                results.append(row_dict)

        df = pd.DataFrame(results)
        df.to_csv(f"{self.submissions_dir}/ensemble_3x_resnet18_runs_19_29_31_arith.csv", index=False)

    def run(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        wandb.init(project=self.params['project_name'], config=self.params)
        self.run_name = wandb.run.name

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.submissions_dir, exist_ok=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_auc = 0.

        self.model = self.model.to(self.device)

        for epoch in range(params['num_epochs']):
            train_metrics = self.train()
            self.lr_scheduler.step()
            val_metrics = self.eval()

            logs = {'train': train_metrics,
                    'val': val_metrics,
                    'lr': self.optimizer.param_groups[0]["lr"]}

            wandb.log(logs, step=epoch)

            current_val_auc = val_metrics['auc']
            if current_val_auc > best_auc:
                best_auc = current_val_auc
                best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_wts)
        self.predict()
        torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/{self.run_name}.pth")


if __name__ == '__main__':
    with open('./configs/default.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    runner = Runner(params)
    # runner.run()
    runner.predict_ensemble()
