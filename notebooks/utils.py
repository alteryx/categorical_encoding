import numpy as np
import pandas as pd
import category_encoders as ce

pd.options.display.float_format = '{:.2f}'.format # to make legible

def load_data():
    df = pd.DataFrame({
    'state':["CA", "MA", "CA", "NY", "CA", "NY"]})
    return df
