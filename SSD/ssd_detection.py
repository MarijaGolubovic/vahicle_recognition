from torch.optim.lr_scheduler import MultiStepLR

from create_data import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from create_data import create_test_dataset, create_test_loader
from training_setup import create_model, train, collate_fn, get_train_transform, get_valid_transform
from training_setup import save_model, save_mAP, save_loss_plot
from evaluate import validate, video, get_metrics

from config import TEST_DIR, TRAIN_DIR, CLASSES, DEVICE, NUM_WORKERS, VALID_DIR, NUM_CLASSES, RESIZE_TO, NUM_EPOCHS, OUT_DIR

import torch
import time
import numpy as np
import os
import argparse



"""This class keeps track of the training and validation loss values
and helps to get the average for each epoch as well."""
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """
    def __init__(
        self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map
        
    def __call__(
        self, 
        model, 
        current_valid_map, 
        epoch, 
        OUT_DIR,
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, f"{OUT_DIR}/best_model.pth")



def main():
    test_dataset = create_test_dataset(TEST_DIR)
    test_loader = create_test_loader(test_dataset, NUM_WORKERS)

    get_metrics(test_loader)
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--detect', 
        default=False,
        type=bool,
        help='run training mode'
    )

    parser.add_argument('--train', type=bool, default=False, help='detection mode')

    args, _ = parser.parse_known_args()
    print(args.detect, type(args.detect), args)
    evaluation = args.detect
    print(evaluation, type(evaluation), args)
    if True:
        video()
    else:
        os.makedirs('outputs', exist_ok=True)
        train_dataset = create_train_dataset(TRAIN_DIR)
        valid_dataset = create_valid_dataset(VALID_DIR)
        train_loader = create_train_loader(train_dataset, NUM_WORKERS)
        valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}\n")

        
        model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)
        model = model.to(DEVICE)
        print(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params, lr=0.0005
        )
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True
        )

        train_loss_hist = Averager()
        # To store training loss and mAP values.
        train_loss_list = []
        map_50_list = []
        map_list = []

        MODEL_NAME = 'model'
        save_best_model = SaveBestModel()

        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

            train_loss_hist.reset()

            start = time.time()
            train_loss = train(train_loader, model, optimizer, train_loss_hist)
            metric_summary = validate(valid_loader, model)
            print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
            print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
            print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")   
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
            print(metric_summary)

            train_loss_list.append(train_loss)
            map_50_list.append(metric_summary['map_50'])
            map_list.append(metric_summary['map'])

            save_best_model(
                model, float(metric_summary['map']), epoch, 'outputs'
            )

            save_model(epoch, model, optimizer)

            save_loss_plot(OUT_DIR, train_loss_list)

            save_mAP(OUT_DIR, map_50_list, map_list)
            scheduler.step()

if __name__ == '__main__':
    main()
