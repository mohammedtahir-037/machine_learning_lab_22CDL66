

'''************    program ********************************'''
import pandas as pd
from collections import Counter
import math

tennis = pd.read_csv('PlayTennis.csv')
print("\nGiven Play Tennis Data Set:\n\n",tennis)

def entropy(alist):
    c=Counter(x for x in alist)
    instance=len(alist)
    prob=[x/instance for x in c.values()]
    return sum([-p*math.log(p,2)for p in prob])

def information_gain(d,split,target):
    splitting=d.groupby(split)
    n=len(d.index)
    agent=splitting.agg({target:[entropy,lambda x:len(x)/n]})[target]
    agent.columns=['Entropy','observations']
    newentropy=sum(agent['Entropy']*agent['observations'])
    oldentropy=entropy(d[target])
    return oldentropy-newentropy

def id3(sub, target, a):
    count = Counter(x for x in sub[target])      
    if len(count) == 1:
        return next(iter(count))   
    else:
        gain = [information_gain(sub, attr, target) for attr in a]
        print("Gain=",gain)
        maximum = gain.index(max(gain)) 
        best = a[maximum]
        print("Best Attribute:",best) 
        tree = {best:{}}
        remaining = [i for i in a if i != best]
        for val, subset in sub.groupby(best): 
            subtree = id3(subset,target,remaining) 
            tree[best][val] = subtree
        return tree
names=list(tennis.columns)
print("List of Attributes:",names)
names.remove('PlayTennis')
print("Predicting Attributes:",names)

tree=id3(tennis,'PlayTennis',names)
print("\n\nThe Resultant Decision Tree is :\n") 
print(tree)



''' ****************  OUTPUT ******************** ''''
Given Play Tennis Data Set:

    PlayTennis   Outlook Temperature Humidity    Wind
0          No     Sunny         Hot     High    Weak
1          No     Sunny         Hot     High  Strong
2         Yes  Overcast         Hot     High    Weak
3         Yes      Rain        Mild     High    Weak
4         Yes      Rain        Cool   Normal    Weak
5          No      Rain        Cool   Normal  Strong
6         Yes  Overcast        Cool   Normal  Strong
7          No     Sunny        Mild     High    Weak
8         Yes     Sunny        Cool   Normal    Weak
9         Yes      Rain        Mild   Normal    Weak
10        Yes     Sunny        Mild   Normal  Strong
11        Yes  Overcast        Mild     High  Strong
12        Yes  Overcast         Hot   Normal    Weak
13         No      Rain        Mild     High  Strong
List of Attributes: ['PlayTennis', 'Outlook', 'Temperature', 'Humidity', 'Wind']
Predicting Attributes: ['Outlook', 'Temperature', 'Humidity', 'Wind']
Gain= [0.2467498197744391, 0.029222565658954647, 0.15183550136234136, 0.04812703040826927]
Best Attribute: Outlook
Gain= [0.01997309402197489, 0.01997309402197489, 0.9709505944546686]
Best Attribute: Wind
Gain= [0.5709505944546686, 0.9709505944546686, 0.01997309402197489]
Best Attribute: Humidity


The Resultant Decision Tree is :

{'Outlook': {'Overcast': 'Yes', 'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}}, 'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
