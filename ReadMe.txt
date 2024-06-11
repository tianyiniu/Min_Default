
Equal Freq "regular" 
	     training_cat1/2   training_cat3    testing_cat1/2  testing_cat3
cvc          132                  99                32              24
cvcvc        132                  99                32              24
vcvc         132                  99                32              24
cvcv           0                  99                 0              24
-----------------
total        396 + 396            396                96             96       



Min Default "regular" 
	     training_cat1/2   training_cat3    testing_cat1/2  testing_cat3
cvc          177                  31                32              24
cvcvc        177                  31                32              24
vcvc         177                  31                32              24
cvcv           0                  31                 0              24
-----------------
total        531 + 531            124               96             96       


Min Default "islands of reliability version for training of cat3 only
cvcv[high_vowel]:  45              
nasal_consonants: 45
Other: 34

high_vowels = ['IY1', 'IH1', 'UW1','UH1','IY0','IH0','UW0','UH0']
nasal_cons = ['M','L','N','NG']






Maj Default 
	     training_cat1/2   training_cat3    testing_cat1/2  testing_cat3
cvc          39                  238                32              24
cvcvc        39                  238               32              24
vcvc         39                  238                32              24
cvcv           0                 238                 0              24
-----------------
total        117 + 117           952               96             96       


-------------------------------



Equal Freq test, L, and Mutants:          
cvc            32              
cvcvc          32              
vcvc           32              
cvcv            0


Mutants: words from "regular" training with the last consonant C changed as follows:
If C belongs to class 1: randomly change it either to class 2 or class 3
If C belongs to class 2: randomly change it either to class 3 or class 1
If C belongs to class 3: randomly change it either to class 2 or class 1


New_templates
	     testing_cat1/2   testing_cat3    
vc            20                20               
cvcc          20                20
ccv                             20
------------------------------------ 
              100

New segments: 
-h final words are "mutants" of templates cvc and cvcvc



