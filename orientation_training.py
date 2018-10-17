import pathlib

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.models import vgg

import freeimage
from zplib.image import colorize
from elegant import load_data, worm_spline

def scale_image(image):
    '''Rescales a 16-bit grayscale image with a mode rescaling and a gamma transform to a float image normalized to the range [0,1]'''
    mode = np.bincount(image.flat)[1:].argmax()+1
    bf = image.astype(np.float32)
    bf -= 200
    bf *= (24000-200) / (mode-200)
    bf = colorize.scale(bf, min=600, max=26000, gamma=0.72, output_max=1)
    return bf

class WormOrientationDataset(data.Dataset):
    def __init__(self, experiment_root, timepoint_filter=None, output_length=760, output_width=240,train=True):
        '''Wrapper class for orientation dataset

        Parameters
            experiment_root: str/pathlib.Path to experiment_root
            timepoint_filter: function used to filter in images for annotated timepoints;
                should be compatible with load_data.scan_experiment_directory
            output_length, output_width: int pixel dimensions for postprocessed output images
            train: bool flag toggling whether to encapsulate train or test dataset;
                if True, keeps the first 75% of annotated data to be used for training, otherwise,
                keeps the last 25% to be use for testing
        '''
        self.experiment_root = pathlib.Path(experiment_root)
        self.filter = filter

        self.output_length = output_length
        self.output_width = output_width # *TOTAL* width of straightened output image

        experiment_image_files = load_data.scan_experiment_dir(experiment_root,timepoint_filter=timepoint_filter)
        self.image_files = [timepoint_images[0] for position,position_images in experiment_image_files.items() for timepoint_images in position_images.values()]
        if train:
            self.image_files = self.image_files[:int(0.75*len(self.image_files))]
        else:
            self.image_files = self.image_files[int(0.75*len(self.image_files)):]
        self.experiment_annotations = load_data.filter_annotations(load_data.read_annotations(experiment_root), load_data.filter_excluded)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        '''A call to this function yields the following sample information:
               orientation - bool flag indicating position;
                   0/False corresponds to default orientation of worms in training set (i.e. facing left with centerline starting at head)
               straightened_image - numpy array representing straightened worm;
                   pixels labeled as background from pose annotations have an intensity of 0
        '''
        image_file  = self.image_files[idx]
        position, timepoint = image_file.parent.name, image_file.name.split()[0]
        timepoint_annotation = self.experiment_annotations[position][1][timepoint]

        image = freeimage.read(str(self.image_files[idx]))
        image = scale_image(image)

        straightened_image = worm_spline.to_worm_frame(
            image,
            timepoint_annotation.get('pose')[0],
            width_tck=timepoint_annotation.get('pose')[1],
            sample_distance=self.output_width/2,standard_length=self.output_length) # Divide by two since to_worm_frame doubles the width
        straightened_mask = worm_spline.worm_frame_mask(timepoint_annotation.get('pose')[1], straightened_image.shape) > 0
        straightened_image[~straightened_mask] = 0

        # Assume all images start in the same orientation; orientation of output is determined by random flipping
        orientation = int(np.random.random() < 0.5) # 0 is default orientation, 1 means it's flipped
        if orientation:
            straightened_image = np.flipud(straightened_image)

        sample_image = vgg_utils.get_VGG_image(straightened_image)

        sample = {'orientation': orientation, 'straightened_image':sample_image}
        return sample


if __name__ == "__main__":
    '''The following script performs the following actions:
           1. Loads training images and creates model (on gpu if available); if model_path is not None, attempts to load from the
    '''

    # Some hard-coded business to do the training on the initial dataset
    experiment_root = pathlib.Path('/mnt/9karray/Sinha_Drew/20180518_spe-9_Run_3/') # Original experiment referenced by the annotations in times.txt
    annotation_list = pathlib.Path('times.txt')
    image_length, image_width = 768, 144
    model_path = pathlib.Path('./') # Directory to load/save model from (creates checkpoints here after each epoch)
    num_epochs = 40

    annotated_images = annotation_list.open('r').read().split('\n')
    def scan_filter(position_name,timepoint_name):
        return position_name + '_' + timepoint_name in annotated_images
    training_dataset = WormOrientationDataset(experiment_root, timepoint_filter=scan_filter,output_length=image_length,output_width=image_width,train=True)
    testing_dataset = WormOrientationDataset(experiment_root, timepoint_filter=scan_filter,output_length=image_length,output_width=image_width,train=False)

    model = vgg_utils.make_compatible_VGG((image_length, image_width))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('model loaded')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Define optimizer after moving model to gpu to maintain consistency of parameter datatypes
    batch_size = 3
    starting_epoch = 0

    # If pre-specified, load the latest model from the directory in model path
    if model_path:
        latest_checkpoint_file = sorted(model_path.glob('model-checkpoint_*.pth'),key=lambda item: int(item.stem.split('_')[1]))[-1]
        print(f'Searched model path; found latest checkpoint file: {str(latest_checkpoint_file)}')
        state = torch.load(latest_checkpoint_file)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        starting_epoch = state['epoch'] + 1

    benchmarking = {}

    for epoch in range(starting_epoch, num_epochs):
        print(f'Starting Epoch {epoch}/{num_epochs}')

        for phase in ['train', 'val']:
            cumulative_loss, cumulative_accuracy = 0, 0.

            if phase == 'train':
                model.train()
                worm_dataset = training_dataset
            else:
                model.eval()
                worm_dataset = testing_dataset

            for batch_num, batch_samples in enumerate(data.DataLoader(worm_dataset, batch_size=batch_size, shuffle=True)):
                optimizer.zero_grad()
                images, orientations = batch_samples['straightened_image'], batch_samples['orientation']
                images, orientations = images.to(device), orientations.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, orientations)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    cumulative_loss += loss.item()*images.size(0)
                    num_correct = (predictions == orientations).cpu().numpy().sum()
                    cumulative_accuracy = (cumulative_accuracy*(batch_num*batch_size) + num_correct)/((batch_num+1)*batch_size)
                print(f'''{phase} - Batch {batch_num+1}/{int(np.ceil(len(worm_dataset)/batch_size))}: cumulative_loss={cumulative_loss}, cumulative_accuracy={cumulative_accuracy}, num_correct_in_batch={num_correct}/{batch_size}''')
            benchmarking[phase] = {'accuracy':cumulative_accuracy, 'loss':cumulative_loss}

        # Save checkpoint at end of each epoch
        if model_path:
            state = {'epoch':epoch,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'benchmarking':benchmarking}
            torch.save(state, model_path / f'model-checkpoint_{epoch}.pth')
