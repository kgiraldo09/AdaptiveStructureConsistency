from pathlib import Path
from monai.networks.layers.factories import Act, Norm

class Options():
    def __init__(self, input_dir = '', input_dir_processed = '', results_dir = '', batch_size = 1, load_model = '', 
                 test=False, NUM_SAMPLES_MAX = 8, HM = False, paired = False, HR = False):
        
        self.HR = HR
        self.paired = paired
        self.HM = HM
        self.NUM_SAMPLES_MAX = NUM_SAMPLES_MAX
        # I/0 parametes
        self.test = test
        self.load_model = load_model
        self.batch_size = batch_size
        self.input_dir = Path(input_dir) if not isinstance(input_dir, list) else [Path(input_dir_) for input_dir_ in input_dir]  # original BraTS23 data folder
        self.input_dir_processed = Path(eval(input_dir_processed)) # where 2d images will be stored for training
        self.results_dir = Path(results_dir)
        #self.EXPERIMENT_PREFIX = Path(eval(EXPERIMENT_PREFIX))
