
# Stock Price Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv("HDFC.csv")

df.head()

df.tail(5)

df.info()

df.shape

df.describe()

df.info()

df.select_dtypes(exclude="object").corr()

df.duplicated().sum()

df.isnull().sum()

"""### Data Preprocessing"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df['Trades'] = imputer.fit_transform(df[['Trades']])
df['Deliverable Volume'] = imputer.fit_transform(df[['Deliverable Volume']])
df['%Deliverble'] = imputer.fit_transform(df[['%Deliverble']])

df=df.drop("Series",axis=1)

df=df.drop("Symbol",axis=1)



df

df.info()

df['Price_Range'] = df['High'] - df['Low']

df

X=df[['Open','Close']]
Y=df['Turnover']



from sklearn.metrics import r2_score

"""### Using RandomForest"""

from sklearn.ensemble import RandomForestRegressor

reger=RandomForestRegressor(n_estimators=100, random_state=42)

reger.fit(X,Y)

y_pred1=reger.predict(X)

comparison_df = pd.DataFrame({
    'Actual': Y,
    'Predicted': y_pred1
})
print(comparison_df.head())

Ac=[]

print("Accuracy score of the predictions: {0}".format(r2_score(Y, y_pred1)))
Ac.append(r2_score(Y, y_pred1))



root= tk.Tk()
root.title("Stock Price Prediction")

















































canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()

label1 = tk.Label(root, text='Type Open Rate: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

label2 = tk.Label(root, text=' Type Close Rate: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

def values():
    global Open #our 1st input variable
    Open = float(entry1.get())

    global Close #our 2nd input variable
    Close = float(entry2.get())

    Prediction_result  = ('Predicted Stock Turnover: ', reger.predict([[Open ,Close]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)

button1 = tk.Button (root, text='Predict Stock Turnover Price',command=values, bg='orange')
canvas1.create_window(270, 150, window=button1)

figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['Open'].astype(float),df['Turnover'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend(['Turnover'])
ax3.set_xlabel('Open')
ax3.set_title('Open Vs. Turnover')

figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['Close'].astype(float),df['Turnover'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root)
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['Turnover'])
ax4.set_xlabel('Close')
ax4.set_title('Close Vs. Turnover')

root.mainloop()







