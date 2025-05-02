'''****************************DATASET FOR Cndidatelimination***************************'''
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes



'''****************************      PROGRAM     ***************************'''

import csv
a=[]
csvfile=open('1.csv','r') 
reader=csv.reader(csvfile) 
for row in reader:
a.append(row) 
print(row)
num_attributes=len(a[0])-1 
print("Initial hypothesis is ") 
S=['0']*num_attributes 
G=['?']*num_attributes 
print("The most specific : ",S) 
print("The most general : ",G)

for j in range(0,num_attributes): 
S[j]=a[0][j]
print("The candidate algorithm \n") 
temp=[]

for i in range(0,len(a)): 
if(a[i][num_attributes]=='Yes'):
for j in range(0,num_attributes): 
if(a[i][j]!=S[j]):
S[j]='?'
for j in range(0,num_attributes):
for k in range(1,len(temp)):
if temp[k][j]!='?' and temp[k][j]!=S[j]: 
del temp[k]
print("For instance {0} the hypothesis is S{0}".format(i+1),S) 
if(len(temp)==0):
print("For instance {0} the hypothesis is G{0}".format(i+1),G) 
else:
print("For instance {0} the hypothesis is G{0}".format(i+1),temp)

if(a[i][num_attributes]=='No'):
for j in range(0,num_attributes): 
if(S[j]!=a[i][j] and S[j]!='?'):
G[j]=S[j]
temp.append(G)
G=['?']*num_attributes
print("For instance {0} the hypothesis is S{0}".format(i+1),S) 
print("For instance {0} the hypothesis is G{0}".format(i+1),temp)




'''*********************** OUTPUT *************************************'''



['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
Initial hypothesis is 
The most specific :  ['0', '0', '0', '0', '0', '0']
The most general :  ['?', '?', '?', '?', '?', '?']
The candidate algorithm 

For instance 1 the hypothesis is S1 ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
For instance 1 the hypothesis is G1 ['?', '?', '?', '?', '?', '?']
For instance 2 the hypothesis is S2 ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
For instance 2 the hypothesis is G2 ['?', '?', '?', '?', '?', '?']
For instance 3 the hypothesis is S3 ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
For instance 3 the hypothesis is G3 [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'Same']]
For instance 4 the hypothesis is S4 ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For instance 4 the hypothesis is G4 [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]
