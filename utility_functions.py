"""
This is the helper functions for various functions
1-4: retrieving the prediction or truth files in data/
5: Put flags.obj and parameters.txt into the folder
6-8: Functions handling flags
"""
import os
import shutil
from copy import deepcopy
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 1
def get_Xpred(path):
    for filename in os.listdir(path):
        if ("Xpred" in filename):
            out_file = filename
            print("Xpred File found", filename)
            break
    return os.path.join(path,out_file)

# 2
def get_Ypred(path):
    for filename in os.listdir(path):
        if ("Ypred" in filename):
            out_file = filename
            print("Ypred File found", filename)
            break
    return os.path.join(path,out_file)

# 3
def get_Xtruth(path):
    for filename in os.listdir(path):
        if ("Xtruth" in filename):
            out_file = filename
            print("Xtruth File found", filename)
            break
    return os.path.join(path,out_file)

# 4
def get_Ytruth(path):
    for filename in os.listdir(path):
        if ("Ytruth" in filename):
            out_file = filename
            print("Ytruth File found", filename)
            break
    return os.path.join(path,out_file)

# 5
def put_param_into_folder(ckpt_dir):
    """
    Put the parameter.txt into the folder and the flags.obj as well
    :return: None
    """
    """
    Old version of finding the latest changing file, deprecated
    # list_of_files = glob.glob('models/*')                           # Use glob to list the dirs in models/
    # latest_file = max(list_of_files, key=os.path.getctime)          # Find the latest file (just trained)
    # print("The parameter.txt is put into folder " + latest_file)    # Print to confirm the filename
    """
    # Move the parameters.txt
    destination = os.path.join(ckpt_dir, "parameters.txt")
    shutil.move("parameters.txt", destination)
    # Move the flags.obj
    destination = os.path.join(ckpt_dir, "flags.obj")
    shutil.move("flags.obj", destination)

# 6
def save_flags(flags, save_dir, save_file="flags.obj"):
    """
    This function serialize the flag object and save it for further retrieval during inference time
    :param flags: The flags object to save
    :param save_file: The place to save the file
    :return: None
    """
    with open(os.path.join(save_dir, save_file),'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object

# 7
def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags

# 8
def write_flags_and_BVE(flags, best_validation_loss, save_dir):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :return: None
    """
    flags.best_validation_loss = best_validation_loss  # Change the y range to be acceptable long string
    # To avoid terrible looking shape of y_range
    yrange = flags.y_range
    # yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    yrange_str = [yrange[0], yrange[-1]]
    copy_flags = deepcopy(flags)
    copy_flags.y_range = yrange_str  # in order to not corrupt the original data strucutre
    flags_dict = vars(copy_flags)
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        print(flags_dict, file=f)
    # Pickle the obj
    save_flags(flags, save_dir=save_dir)

def compare_truth_pred(pred_file, truth_file, cut_off_outlier_thres=None, quiet_mode=False):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    if isinstance(pred_file, str):  # If input is a file name (original set up)
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
    elif isinstance(pred_file, np.ndarray):
        pred = pred_file
        truth = truth_file
    else:
        print('In the compare_truth_pred function, your input pred and truth is neither a file nor a numpy array')
    if not quiet_mode:
        print("in compare truth pred function in eval_help package, your shape of pred file is", np.shape(pred))
    if len(np.shape(pred)) == 1:
        # Due to Ballistics dataset gives some non-real results (labelled -999)
        valid_index = pred != -999
        if (np.sum(valid_index) != len(valid_index)) and not quiet_mode:
            print("Your dataset should be ballistics and there are non-valid points in your prediction!")
            print('number of non-valid points is {}'.format(len(valid_index) - np.sum(valid_index)))
        pred = pred[valid_index]
        truth = truth[valid_index]
        # This is for the edge case of ballistic, where y value is 1 dimensional which cause dimension problem
        pred = np.reshape(pred, [-1, 1])
        truth = np.reshape(truth, [-1, 1])
    mae = np.mean(np.abs(pred - truth), axis=1)
    mse = np.mean(np.square(pred - truth), axis=1)

    if cut_off_outlier_thres is not None:
        mse = mse[mse < cut_off_outlier_thres]
        mae = mae[mae < cut_off_outlier_thres]

    return mae, mse

def plotMSELossDistrib(pred, truth):

    # mae, mse = compare_truth_pred(pred_file, truth_file)
    # mae = np.mean(np.abs(pred - truth), axis=1)
    mse = np.mean(np.square(pred - truth), axis=1)
    # mse = loss
    f = plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Validation Loss')
    plt.ylabel('Count')
    plt.suptitle('Model (Avg MSE={:.4e})'.format(np.mean(mse)))
    # plt.savefig(os.path.join(os.path.abspath(''), 'models',
    #                          'MSEdistrib_{}.png'.format(flags.model_name)))
    return f
    # plt.show()
    # print('Backprop (Avg MSE={:.4e})'.format(np.mean(mse)))


def plotMSELossDistrib_eval(pred_file, truth_file, flags):

    mae, mse = compare_truth_pred(pred_file, truth_file)
    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=100)
    plt.xlabel('Mean Squared Error')
    plt.ylabel('cnt')
    plt.suptitle('(Avg MSE={:.4e})'.format(np.mean(mse)))
    eval_model_str = flags.eval_model.replace('/','_')
    plt.savefig(os.path.join(os.path.abspath(''), 'eval',
                         '{}.png'.format(eval_model_str)))
    print('(Avg MSE={:.4e})'.format(np.mean(mse)))
