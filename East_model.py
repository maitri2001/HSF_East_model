# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:12:16 2024

@author: maitr
"""

from __future__ import print_function, division
import sys,os
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation

import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library

from scipy.sparse import csr_matrix # for sparse matrix tranformation

try: # import python 3 zip function in python 2 and pass if already using python 3
    import itertools.izip as zip
except ImportError:
    pass 
from collections import Counter
# Define system parameters
L=5
a = 1
J = 1 # hopping parameter
v0=0;

basis = spinless_fermion_basis_1d(L=L,a=1)#,Nf=L//2 +2)
print(basis)
hop_pm = [[-J,(i-1),i,(i+1)%L] for i in range(1,L-1,1)] # OBC
hop_mp = [[J,(i-1),i,(i+1)%L] for i in range(1,L-1,1)] # OBC
stagg_pot = [[v0*(-1)**i,i,i] for i in range (L)]

### Create the static and dynamic lists
static = [["n+-",hop_pm],["n-+", hop_mp]]
dynamic = []

# Build real-space Hamiltonian
H = hamiltonian(static,dynamic,basis=basis,dtype = np.float64 )
H_R = H.toarray() # Real space Hamiltonian in matrix form 
SH_R = csr_matrix(H_R) ## transform the Hamiltonian in Sparse matrix form

print("The Hamiltonian in real space is :")
print(H_R)
print("The Hamiltonian in sparse matrix form :")
print(SH_R)

#### Diagonalize the real space Hamiltonian
E,V = H.eigh()
print(E);

#print(basis.Ns) # No of state available in this basis for this filling  

          
            
s = [] #This will take care of which basis states are already taken/categorized in fragments
 
dim = [] # here will shall store dimensionality of each fragment
s_frag=[] 
 
for i in range(basis.Ns):
    if i not in s: # start with an i which is not already categorized into fragments
        psi = np.zeros(basis.Ns)
        psi[i] = 1 # state which is not already categorized into fragments
        v1 = np.zeros(basis.Ns)
        v1 = psi  #starting point
        # now with that starting state, we shall see which states it goes into
        # p is just a dummy variable to keep evolution with Hamiltonian running unless it saturates
        for p in range(basis.Ns): 
            v2 = np.zeros(basis.Ns)
            u = H.dot(v1)
            for j in range(basis.Ns): # usual updating of v2
                if v1[j] != 0:
                    v2[j] = 1
                if u[j] != 0:
                    v2[j] = 1
            if (v1.all) == (v2.all): # saturation condition
                break
            else: # if does not saturate, again evolve with H, by going next value of n
                v1 = v2
        #the for loop with p will exit once the cycle of a single fragment saturates
        count = 0
        s_tem =[]
        for l in range(basis.Ns):
            if v2[l]== 1:
                count = count + 1 #counting number of non zero elements in saturated v2
                s.append(l) #keepung track which states are already in one fragment
                s_tem.append(l)
                
        s_frag.append(s_tem)        
        dim.append(count) ## number of elements in a fragment of that cycle
        # now will update i such that it is not in s anymore

print(dim) 
print(s_frag)    
freq=[{i:dim.count(i) for i in dim}] ;
x=Counter(dim)      
print(x)
frag_size=list(x.keys()); #  No of connected States in that fragment
no_of_frag=list(x.values()); # No of fragment avaible with that size
total=np.sum(no_of_frag)
plt.plot(frag_size,no_of_frag,'k.')
plt.xlabel("frag_size")
plt.ylabel("no_of_frag")
plt.show()

##################################
#Using Transfer matrix method fragment Calculation  

# Define the transfer matrix

def Matrix(M):
    for row in M:
        for element in row:
            print(element, end='\t')
        print()  



def Matrix_power(Matrix,power):
    result = np.eye(len(Matrix))
    for i in range(power):
        result = np.dot(result,Matrix)
    return result
  
    
def Num_frag_T_method(matrix):
    sum = 0
    for row in matrix:
        for element in row :
            sum += element
    return sum

## Calculating the total number of fragment

T2tot = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
print("The transfer matrix T2tot is =")
Matrix(T2tot)

T2toteigenvalues = np.linalg.eigvals(T2tot)
print("The eigenvalues of the Transfer matrix for total frag is =",T2toteigenvalues)

result_matrix_tot_frag =  Matrix_power(T2tot,L-2) 
print("\n For total frag the transfer matrix for L site is T^(L-2)\t =")
Matrix(result_matrix_tot_frag)


Total_frag=Num_frag_T_method(result_matrix_tot_frag)
print("The total number of fragments using Transfer matrix is =", Total_frag)

'''L = [5,6,8]
for i in L:
    print("The Transfer matrix for L =",L)
    print((Matrix_power(T2,L-2)))
    print ("The total number of fragments using Transfer matrix is for L =",L )
    print("is ",Num_frag_T_method(Matrix_power(T2,L-2)))'''


## Calculating the number of frozen state using transfer matrix method
 
T2f = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
print("The transfer matrix for frozen state T2f is =")
Matrix(T2f) 

T2feigenvalues = np.linalg.eigvals(T2f)
print("The eigenvalues of the Transfer matrix for frozen state are =",T2feigenvalues)

result_matrix_froz =  Matrix_power(T2f,L-2) 
print("\n The transfer matrix for L site is T^(L-2)\t =")
Matrix(result_matrix_froz) 

froz_frag=Num_frag_T_method(result_matrix_froz)
print("The total number of frozen fragments using Transfer matrix is =", froz_frag)


####### Analysis on a particular fragment

dim_particular_frag = [] # here will shall store dimensionality of that particular fragment

s_particular_frag=[] # storing the indices of the states in this fragment

psi = np.zeros(basis.Ns)
state_str = '01010'
state_index = basis.index(state_str)

#print("The state index is",state_index)
psi[state_index] = 1 
v1 = np.zeros(basis.Ns)
v1 = psi  #starting point

# p is just a dummy variable to keep evolution with Hamiltonian running unless it saturates
for p in range(basis.Ns): 
    v2 = np.zeros(basis.Ns)
    u = H.dot(v1)
    for j in range(basis.Ns): # usual updating of v2
        if v1[j] != 0:
            v2[j] = 1
        if u[j] != 0:
            v2[j] = 1
    if (v1.all) == (v2.all): # saturation condition
        break
    else: # if does not saturate, again evolve with H, by going next value of n
        v1 = v2
#the for loop with p will exit once the cycle of a fragment saturates
count = 0
s_tem =[]
for l in range(basis.Ns):
    if v2[l]== 1:
        count = count + 1 #counting number of non zero elements in saturated v2
        s.append(l) #keepung track which states are already in one fragment
        s_tem.append(l)    
s_particular_frag.append(s_tem)        
dim_particular_frag.append(count) 

print("Size of this particular fragment is :",dim_particular_frag) 
print("Index of the states in this fragment:",s_particular_frag) 