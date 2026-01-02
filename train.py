# فایل: C:\Users\nader\Desktop\api\train.py
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
import pickle
import numpy as np
import os

# ساخت داده مصنوعی (چون شاید فایل csv را نداشته باشید)
# اگر فایل csv دارید خط بعد را پاک کنید و pd.read_csv بزنید
df = pd.DataFrame(np.random.rand(100, 6), columns=['f1','f2','f3','f4','f5','out1'])

x = df.drop(columns=['out1'])
y = df['out1']

model = Sequential()
model.add(Dense(20, input_dim=5, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=5, verbose=0)

# ذخیره مدل در مسیر جاری
current_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_path, 'co2_model.pkl')

with open(save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"مدل با موفقیت در این مسیر ساخته شد: {save_path}")