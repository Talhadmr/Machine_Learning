import numpy as np 


# In[70]:


x = [[1,2,3],[4,5,6]]


# In[72]:


arr = np.array(x)


# In[83]:


arr[:2, 1:2]


# In[116]:


arr.transpose()


# In[127]:


features = {"limbs":[0,4,4,4,8],            "herbivore":["No","No","Yes","Yes","No"]}

animals = ["Python", "Iberian Lynx",           "Giant Panda", "Field Mouse", "Octopus"]


# In[128]:


import pandas as pd


# In[129]:


df = pd.DataFrame(features, index = animals)


# In[130]:


df


# In[131]:


df.head(3)


# In[132]:


df['limbs'][2:5]


# In[133]:


df.loc['Python']


# In[134]:


df['limbs'].describe()


# In[135]:


df['class'] = ['reptile','mammal', 'mammal','mammal', 'molusc']


# In[149]:


grouped = df.groupby(df['herbivore'])


# In[150]:


grouped.groups


# In[151]:


df


# In[153]:


grouped.size()


# In[170]:


from numpy import mean
grouped = df['limbs'].groupby(df['herbivore']).aggregate(mean)


# In[172]:


print(grouped)


# In[173]:


x = np.linspace(-5, 5, 200)


# In[175]:


y1 = x**2
y2 = x**3


# In[176]:


import matplotlib.pyplot as plt


# In[214]:


fig, ax =plt.subplots()
ax.plot(x,y1, 'r', label = r"$y_1 = x^2$", linewidth = 2)
ax.plot(x,y2, 'k--', label = r"$y_2 = x^3$", linewidth = 2)
ax.legend(loc = 2)
ax.set_xlabel(r"$x$", fontsize = 18)
ax.set_ylabel(r"$y$", fontsize = 18)
ax.set_title("My figure")
plt.show()


# In[193]:





# In[ ]:




