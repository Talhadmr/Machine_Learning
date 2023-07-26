#kütüphanleri include ederek başlayalım
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from   sklearn import linear_model
from numpy import log 

#csv dosyası içinde bulunan verilerimizi read csv\
#fonk yardımı ile dataframe çevirelim
df = pd.read_csv("mammals.csv")

#elimizdeki verileri daha iyi görebilmek için bir grafik oluşturalım
#plt.scatter(df['body'], df['brain'], c = "red")
#plt.show()

#degişkenlerin giriş sayısına bakalım
b_data = df['body']
v_data = df['brain']
#print(b_data.shape)
#veri setinde 62 kayıt bulunmaktadır

#verilerin arasındaki korelasyona bakalım
#print(np.corrcoef(b_data,v_data))
#aralarında 0.93416 pozitif bir korelasyon var
b_data = sm.add_constant(b_data)

#nodel oluşturuluyor 
regression1 = sm.OLS(v_data,b_data).fit()

regression2 = smf.ols(formula= 'brain ~ body', data= df).fit()

#model oluşturulduktan sonra modelimizin tahmin gücünü test etmek için test veri seti oluşturuyoruz

new_b = np.linspace(0, 7000, 10)

b_pred = regression2.predict(exog = dict(body = new_b))

#tahmin verilerimizi daha iyi görebilmek için grafik oluşturalım
#plt.scatter(new_b,b_pred)
#plt.show()

######## neden ols kullandık bilimiyorum neden constant oluşturduk

#şimdi bir de sklearn ile model oluşturalım

s_reg = linear_model.LinearRegression()

s_reg.fit(b_data,v_data)
pn_b = np.linspace(0, 7000, 10)
pn_b = pn_b[:, np.newaxis]
#s_pred = s_reg.predict(pn_b)

df['log_body'] = log(df['body'])
df['log_brain'] = log(df['brain'])

plt.scatter(df['log_body'],df['log_brain'])
plt.show()