import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os


def load_tensorboard_data(log_dir):
    data = {}
    sub_dirs = ['Accuracy_Training', 'Accuracy_Validation', 'Loss_Training', 'Loss_Validation']
    for sub_dir in sub_dirs:
        path = os.path.join(log_dir, sub_dir)
        event_files = [f for f in os.listdir(path) if 'events.out.tfevents' in f]
        for file in event_files:
            full_path = os.path.join(path, file)
            ea = event_accumulator.EventAccumulator(full_path)
            ea.Reload()
            tag = ea.Tags()["scalars"][0]
            events = ea.Scalars(tag)
            key_name = sub_dir
            data[key_name] = [(e.step, e.value) for e in events]
    return data


def plot_accuracy(data, title):
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in data['Accuracy_Training']], [x[1] for x in data['Accuracy_Training']],
            label='Training Accuracy', color='tab:blue')
    ax.plot([x[0] for x in data['Accuracy_Validation']], [x[1] for x in data['Accuracy_Validation']],
            label='Validation Accuracy', color='tab:red', linestyle='dashed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_loss(data, title):
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in data['Loss_Training']], [x[1] for x in data['Loss_Training']], label='Training Loss',
            color='tab:blue')
    ax.plot([x[0] for x in data['Loss_Validation']], [x[1] for x in data['Loss_Validation']], label='Validation Loss',
            color='tab:red', linestyle='dashed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    plt.show()


# Usage
log_dir = '/Users/cyixiao/Desktop/562-final-project/CNN/train_logs/lr_0.01_wd_0.001'
data = load_tensorboard_data(log_dir)
plot_accuracy(data, 'Accuracy Curve')
plot_loss(data, 'Loss Curve')
