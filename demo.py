import os

str_RC = "python test.py --strategy RC --testdata 'SET14' --srcpath 'SET14_noisy' --tarpath 'SET14' "
str_AL = "python test.py --strategy AL --testdata 'SET14' --srcpath 'SET14_noisy' --tarpath 'SET14' "
p_RC = os.system(str_RC)
p_AL = os.system(str_AL)
print(p_RC)
print(p_AL)
