from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle  # <--- کتابخانه مورد نیاز

# 1. آماده‌سازی داده‌ها
# فرض می‌کنیم فایل csv موجود است
try:
    df = pd.read_csv('co2 (1).csv')
except:
    # داده مصنوعی برای تست در صورت نبود فایل
    import numpy as np
    df = pd.DataFrame(np.random.rand(100, 6), columns=['f1','f2','f3','f4','f5','out1'])

x = df.drop(columns=['out1'])
y = df['out1']

# 2. ساخت مدل
model = Sequential()
model.add(Dense(20, input_dim=5, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. آموزش
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model.fit(x_train, y_train, epochs=10)

# 4. === ذخیره به صورت PKL ===
# فایل را در حالت write binary (wb) باز می‌کنیم
with open('co2_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("مدل با فرمت pkl ذخیره شد.")