"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
LINEAR = [4, 250, 250, 250]
USE_CONV = False
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-4
BATCH_SIZE = 512
EVAL_STEP = 10
RECORD_STEP = 10
TRAIN_STEP = 10000
LEARN_RATE = 1e-2
LR_DECAY_RATE = 0.5
# DECAY_STEP = 2000 # For when using step decay, rather than dynamic scheduling
STOP_THRESHOLD = 1e-5
USE_CLIP = False
GRAD_CLIP = 5
USE_WARM_RESTART = True
LR_WARM_RESTART = 200

# Data Specific parameters
X_RANGE = [i for i in range(0, 4)]
Y_RANGE = [i for i in range(1, 1001,2)]
FREQ_LOW = 0.8
FREQ_HIGH = 1.5
NUM_SPEC_POINTS = 300
FORCE_RUN = True
DATA_DIR = ''                # For local usage
# DATA_DIR = 'C:/Users/labuser/'                # For Omar office desktop usage
# DATA_DIR = 'C:/Users/Omar/'                # For Omar office desktop usage
# DATA_DIR = '/home/omar/PycharmProjects/'  # For Omar laptop usage
# Format for geoboundary is [p0_min... pf_min p0_max... pf_max]
GEOBOUNDARY =[1.3, 0.975, 6, 34.539, 2.4, 3, 7, 43.749]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None 
EVAL_MODEL = "Eval_model"
NUM_PLOT_COMPARE = 10
