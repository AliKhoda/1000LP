import numpy as np

# Feature identifiers
select = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8',
        'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16',
        'X_17', 'X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24',
        'X_25', 'X_26', 'X_27', 'X_28', 'X_29', 'X_30', 'X_31', 'X_32',
        'X_33', 'X_34', 'X_35', 'X_36', 'X_37', 'X_38', 'X_39', 'X_40',
        'X_41', 'X_42', 'X_43', 'X_44', 'X_45', 'X_46', 'X_47', 'X_48',
        'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56',
        'X_57', 'X_58', 'X_59', 'X_60', 'X_61', 'X_62', 'X_63', 'X_64',
        'X_65', 'X_66', 'X_67', 'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5',
        'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14',
        'Y_15', 'Y_16', 'Y_17', 'Y_18', 'Y_19', 'Y_20', 'Y_21', 'Y_22',
        'Y_23', 'Y_24', 'Y_25', 'Y_26', 'Y_27', 'Y_28', 'Y_29', 'Y_30',
        'Y_31', 'Y_32', 'Y_33', 'Y_34', 'Y_35', 'Y_36', 'Y_37', 'Y_38',
        'Y_39', 'Y_40', 'Y_41', 'Y_42', 'Y_43', 'Y_44', 'Y_45', 'Y_46',
        'Y_47', 'Y_48', 'Y_49', 'Y_50', 'Y_51', 'Y_52', 'Y_53', 'Y_54',
        'Y_55', 'Y_56', 'Y_57', 'Y_58', 'Y_59', 'Y_60', 'Y_61', 'Y_62',
        'Y_63', 'Y_64', 'Y_65', 'Y_66', 'Y_67', 'Z_0', 'Z_1', 'Z_2', 'Z_3',
        'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12',
        'Z_13', 'Z_14', 'Z_15', 'Z_16', 'Z_17', 'Z_18', 'Z_19', 'Z_20',
        'Z_21', 'Z_22', 'Z_23', 'Z_24', 'Z_25', 'Z_26', 'Z_27', 'Z_28',
        'Z_29', 'Z_30', 'Z_31', 'Z_32', 'Z_33', 'Z_34', 'Z_35', 'Z_36',
        'Z_37', 'Z_38', 'Z_39', 'Z_40', 'Z_41', 'Z_42', 'Z_43', 'Z_44',
        'Z_45', 'Z_46', 'Z_47', 'Z_48', 'Z_49', 'Z_50', 'Z_51', 'Z_52',
        'Z_53', 'Z_54', 'Z_55', 'Z_56', 'Z_57', 'Z_58', 'Z_59', 'Z_60',
        'Z_61', 'Z_62', 'Z_63', 'Z_64', 'Z_65', 'Z_66', 'Z_67', 
        'pose_Rx', 'pose_Ry', 'pose_Rz']
        
lnc = 68 # number of landmarks
rln = 33 # Tip of nose feature location

# Average landmark locations after normalization and rotation
avgpos = np.load('avgpos.npy')

# Set nose tip landmark location to center of the coordinate system
avgpos[rln] = 0
avgpos[rln+lnc] = 0
avgpos[rln+2*lnc] = 0

# standard deviation of landmark locations after normalization and rotation
stdpos = np.load('stdpos.npy')

# rotation matrix
def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
    return np.dot(R_z, np.dot( R_y, R_x ))

# Normalize by moving tip of the nose to [0, 0, 0]
# Rotate to zero roll, yaw, and pitch
def refine_3d_corr(feat):
    
    refined = np.zeros((feat.shape[0], 3*lnc), dtype='float32')
    for frame in range(feat.shape[0]):
    
        # Translate based on the location of the tip of the nose
        x = feat[frame,:lnc]-feat[frame,rln]
        y = feat[frame,lnc:2*lnc]-feat[frame,lnc+rln]
        z = feat[frame,2*lnc:3*lnc]-feat[frame,2*lnc+rln]

        # Get head pose and calculate rotation matrix
        Rx = feat[frame,3*lnc]
        Ry = feat[frame,3*lnc+1]
        Rz = feat[frame,3*lnc+2]

        R = eulerAnglesToRotationMatrix([-Rx, -Ry, -Rz])

        # Rotate
        rotated = np.dot(R,[x, y, z])

        # Replace nose tip coordinates with head pose
        rotated[0,rln] = Rx*180/np.pi
        rotated[1,rln] = Ry*180/np.pi
        rotated[2,rln] = Rz*180/np.pi
        
        refined[frame,:] = rotated.flatten()
            
    # Normalize
    refined[:,rln] -= np.mean(refined[:,rln])
    refined[:,rln+lnc] -= np.mean(refined[:,rln+lnc])
    refined[:,rln+2*lnc] -= np.mean(refined[:,rln+2*lnc])
    
    return refined

# load openface output in npy 
# Please assert that after loading the landmarks are scaled to a fixed scale. e.g. inner eye landmark distance of 0.5 pixels
# The scaling part of the code is missing as in the original experiment setup, all videos were of faces at the same scale
def readnpz(filename):
    npz = np.load(filename)
    selected = np.array(npz[select].tolist(), dtype='float32')
    
    # Normalize by pose and translate
    refined = refine_3d_corr(selected)

    # Normalize by training average
    refined -= avgpos
    refined /= stdpos
    return refined
