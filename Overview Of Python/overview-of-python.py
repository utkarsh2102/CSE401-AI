#!/usr/bin/env python
# coding: utf-8

# ### Overview of Python
#
# This contains the solution of `List` and `Tuple` questions as the lab assignment.

# #### List

# ###### Q1. Write a Python program to sum all the items in a list.

# In[1]:


numbers = list(map(int, input("Input the numbers: ").split()))

# way 1:
totalSum = 0
for i in numbers:
    totalSum += i

print(f"Total sum is: {totalSum}")

# way 2:
print()
print("Alternatively, we use the 'sum' function in Python:")
print(f"sum(numbers): {sum(numbers)}")


# ###### Q2. Write a Python program to get the largest number from a list.

# In[2]:


numbers = list(map(int, input("Input the numbers: ").split()))

# way 1:
numbers.sort()
print(f"The largest number is: {numbers[-1]}")

# way 2:
print()
print("Alternatively, we can use the 'max' function in Python:")
print(f"max(numbers): {max(numbers)}")


# ###### Q3. Write a Python program to get the smallest number from a list.

# In[3]:


numbers = list(map(int, input("Input the numbers: ").split()))

# way 1:
numbers.sort()
print(f"The smallest number is: {numbers[0]}")

# way 2:
print()
print("Alternatively, we can use the 'min' function in Python:")
print(f"min(numbers): {min(numbers)}")


# ###### Q4. Write a Python program to multiply all the items in a list.

# In[4]:


numbers = list(map(int, input("Input the numbers: ").split()))

# way 1:
product = 1
for i in numbers:
    product *= i
print(f"The product of all items: {product}")

# way 2:
import math
print()
print("Alternatively, we can use the 'prod' function from the 'math' library:")
print(f"math.prod(numbers): {math.prod(numbers)}")


# #### Tuple

# ###### Q1. Write a Python program to create a tuple.

# In[5]:


someTuple = (21, 12)
print(type(someTuple))
print(someTuple)


# ###### Q2. Write a Python program to create a tuple with different data types.

# In[6]:


someTuple = (21, 'debian', 12)
print(someTuple)


# ###### Q3. Write a Python program to create a tuple with numbers and print one item.

# In[7]:


someTuple = tuple(map(int, input("Enter the items: ").split()))
itemNumber = int(input("Enter the item number you want to print: "))

print(f"Item number {itemNumber} is '{someTuple[itemNumber-1]}'.")


# ###### Q4. Write a Python program to add an item in a tuple.

# In[8]:


someTuple = tuple(map(int, input("Enter the items: ").split()))
addItem = input("Enter the item to add: ")

# way 1:
newTuple = someTuple + (addItem,)
print(f"The new tuple is: {newTuple}")

# way 2:
print()
print("Alternatively, we can convert the tuple into list and add items.")
someList = list(someTuple)
someList.append(addItem)
newTuple = tuple(someList)
print(f"The new tuple is: {newTuple}")


# ###### Q5. Write a Python program to convert a tuple to a string.

# In[9]:


someTuple = tuple(map(str, input("Enter the characters: ").split()))
convertedString = ''.join(someTuple)
print(f"Converted string: '{convertedString}'.")

