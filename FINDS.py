

'''****************************  PROGRM ***************************'''
import csv
num_attributes=6 
a=[]
print("\n The given training data set \n") 
csvfile=open("finddss.csv",'r')
reader=csv.reader(csvfile)
for row in reader: 
    a.append(row) 
    print(row)
print("The initial values of hypothesis ") 
hypothesis=['0']*num_attributes 
print(hypothesis)

for j in range(0,num_attributes): 
    hypothesis[j]=a[0][j]

for i in range(0,len(a)): 
    if(a[i][num_attributes]=='Yes'):
        for j in range(0,num_attributes): 
            if(a[i][j]!=hypothesis[j]):
                hypothesis[j]='?' 
            else:
                hypothesis[j]=a[i][j]
    print("For training instance no:",i," the hypothesis is ",hypothesis) 
print("The maximally specific hypothesis is ",hypothesis)




''' ********************* OUTPUT ***********************'''
 The given training data set 

['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
The initial values of hypothesis 
['0', '0', '0', '0', '0', '0']
For training instance no: 0  the hypothesis is  ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
For training instance no: 1  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
For training instance no: 2  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
For training instance no: 3  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 4  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 5  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 6  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 7  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 8  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 9  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 10  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 11  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 12  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 13  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 14  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 15  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 16  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 17  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 18  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
For training instance no: 19  the hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
The maximally specific hypothesis is  ['Sunny', 'Warm', '?', 'Strong', '?', '?']
