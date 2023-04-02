# the loaded data type in MATLAB is struct with 2 fields: channels and userLoc
# channels.shape: (#ant, #sub, #user), userLoc.shape: (#user, 3)
# And also note that each element of channels is complex number.
import numpy as np
import h5py as h5


def dataPrep(inputName=None):
    with h5.File(inputName, 'r') as f:
        fields = [k for k in f.keys()]  # fields = ['ch_grid']
        nested = [k for k in f[fields[0]]]
        data_channels = np.squeeze(np.array(nested))
        decoup = data_channels.view(np.float64).reshape(data_channels.shape + (2,))
        # shape: (#users, #ant, 2), decoup[0,0,0]=real, decoup[0,0,1]=imag
        X_real = decoup[:, :, 0]  # shape: (#users, #ant), all real parts of channels
        X_imag = decoup[:, :, 1]  # shape: (#users, #ant), all imag parts of channels
        X = np.concatenate((X_real, X_imag), axis=-1)

    return X


# --- test script ---
# path = 'C:/Users/yzhan899/Dropbox (ASU)/Research/Paper_Channel Tracking/codes/DeepMIMO_Dataset_Generation v1.1/CT_O1_3p5_BS3_5Paths_norm.mat'
# data = dataPrep(path)
