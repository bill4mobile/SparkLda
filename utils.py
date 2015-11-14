# tools for spark lda

import sys
import math

def digamma(x):
    '''
     first derivative function of gamma
    '''
    x = x + 6.0
    p = 1.0/(x*x)
    p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
    p=p+math.log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0);
    return p

def log_sum(arr):
    sum = 0.0
    for ele in arr:
        sum += math.exp(ele)
    return math.log(sum)

def lgamma(x):
    z=1.0/(x*x)
    x=x+6.0
    z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x
    z=(x-0.5)*math.log(x)-x+0.918938533204673+z-math.log(x-1)-math.log(x-2)-math.log(x-3)-math.log(x-4)-math.log(x-5)-math.log(x-6)
    return z
    


