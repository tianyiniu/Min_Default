
Equal Freq "regular" 
	     training_cat1/2   training_cat3    testing_cat1/2  testing_cat3
cvc          132                  99                32              24
cvcvc        132                  99                32              24
vcvc         132                  99                32              24
cvcv           0                  99                 0              24
-----------------
total        396                  396                96             96       



Equal Freq test L and Mutants:          
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
------------------ 
              80

The low number '20' is explained by the fact that there are few VC words in general, and VC class 2 words in particular. 



