#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Import the numpy package under the np
import numpy as np

#Create a null vector of size 10
a = np.zeros(10)


# In[8]:


#Create a vector with values ranging from 10 to 49

b =np.arange(10,50)
b


# In[14]:


#Find the shape of previous array in question 3
c = np.shape(b)


# In[17]:


#Print the type of the previous array in question 3
b.dtype


# In[20]:


#Print the numpy version and the configuration
print(np.__version__)
print(np.show_config())


# In[21]:


#Print the dimension of the array in question 3
b.ndim


# In[25]:


#Create a boolean array with all the True values
d = np.ones((2,2),dtype = bool)
d


# In[29]:


#Create a two dimensional array
d2 = np.array([[2, 4, 6], [6, 8, 10]], np.int32)
d2


# In[30]:


#Create a three dimensional array
d3 = np.array([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7 ], [7, 8, 9]]])
d3


# In[34]:


#Reverse a vector (first element becomes last)
e =np.array((1,2,3,4,5,6))
e[: : -1]


# In[36]:


#Create a null vector of size 10 but the fifth value which is 1
x = np.zeros(10)
x
print(" fifth value is 1")
x[5] = 1
x


# In[37]:


#Create a 3x3 identity matrix
idn =np.identity(3)
idn


# In[39]:


#Convert the data type of the given array from int to float
arr = np.array([1, 2, 3, 4, 5])
arr.astype(float)


# In[40]:


#Multiply arr1 with arr2
arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  

arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2


# In[44]:


#Make an array by comparing both the arrays provided above
arr3 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 

arr4 = np.array([[0., 4., 1.],

            [7., 2., 12.]])

comparison = arr3 == arr4 
aq = comparison.all() 
aq


# In[63]:


#Extract all odd numbers from arr with values(0-9)
od = np.arange(0,10)

ab = od[od % 2 == 1]
ab


# In[65]:


#Replace all odd numbers to -1 from previous array
od[od % 2 == 1] = -1
od


# In[67]:


#Replace the values of indexes 5,6,7 and 8 to 12
arrry = np.arange(0,20)
arrry
arrry[5]
arrry[6]
arrry[7]
arrry[8]
arrry[9]
arrry[10]
arrry[11]
arrry[12]




# In[74]:


#Create a 2d array with 1 on the border and 0 inside
arrrry = np.ones((4,4))
arrrry[1:-1,1:-1]=0
arrrry


# In[ ]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
#Replace the value 5 to 12


# In[81]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]])
#Convert all the values of 1st array to 64


# In[97]:


#Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
ary =  np.arange(0,9).reshape(3,3)
ary
ary[0]


# In[92]:


#Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
ary[0][1]


# In[101]:


#make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows
ary
ary[0:2,2]


# In[103]:


#Create a 10x10 array with random values and find the minimum and maximum values
yx = np.random.random((10,10))
print("Original Array:")
print(yx) 
yxmin, yxmax = yx.min(), yx.max()
print("Minimum and Maximum Values:")
print(yxmin, yxmax) 


# In[105]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
#Find the common items between a and b
ag = np.intersect1d(a, b)
ag


# In[107]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
#Find the positions where elements of a and b match
np.where(a == b)


# In[ ]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
#Find all the values from array data where the values from array names are not equal to Will


# In[110]:


#Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.
rand_arr = np.random.uniform(1,15, size=(5,3))
print(rand_arr)


# In[111]:


#Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16
rand_arr = np.random.uniform(1,16, size=(2,2,4))
print(rand_arr)




# In[116]:


#Swap axes of the array you created in Question 32
np.swapaxes(rand_arr, 1, 2)


# In[125]:


#Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
al = np.arange(10)
sqr= np.sqrt(al)
sqr


# In[145]:


#Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays
aj = np.random.rand(10)
ak = np.random.rand(10)
print(aj)
print(ak)


# In[143]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
#Find the unique names and sort them out!
np.unique(names)


# In[142]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
#From array a remove all items present in array b
result = np.setdiff1d(a, b)
result


# In[141]:


#Following is the input NumPy array delete column two and insert following new column in its place.
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]]
sa = np.delete(sa , 1, axis = 1) 
arr = np.array([[10,10,10]])
sb = np.insert(sb , 1, arr, axis = 1) 
sb
                        
                        


# In[131]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
#Find the dot product of the above two matrix
p = x.dot(y)
p


# In[135]:


#Generate a matrix of 20 random values and find its cumulative sum
wd = np.arange(20)
np.cumsum(wd)


# In[ ]:





# In[ ]:




