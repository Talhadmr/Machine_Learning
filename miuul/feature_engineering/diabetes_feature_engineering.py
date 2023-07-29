import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("diabetes.csv")

df.shape
df.dtypes
df.describe().T
df.head()

df.groupby(['Pregnancies'])["Outcome"].agg(["mean","count", "sum"])
sns.catplot(x='Pregnancies', y='Outcome', data=df, kind='bar')
plt.show()
#buradan anlaşılacağı üzere hamilelik durumu diyabet ile pozitif korelasyon gösteriyor
#yukarıdaki uygulamayı diğer kolonlar üzerine de yapılabilir.

sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()

#kolon isimlerinin sadece küçük harflerden oluşmasını istiyorum

df.columns = [col.lower() for col in df.columns]

