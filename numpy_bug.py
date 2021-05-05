import numpy as np
import warnings

np.seterr(all='warn')
warnings.filterwarnings('error')


# Triggers Warning
try:
    x = np.exp(np.float64(-1000))

except RuntimeWarning:
    print("Warning triggered for float64 type")


# Doesn't Trigger Warning
try:
    x = np.exp(np.float32(-1000))

except RuntimeWarning:
    print("Message won't print - Warning not triggered for float32 type")