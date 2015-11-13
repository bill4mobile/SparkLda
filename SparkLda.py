# lda training ... 

import math
import setting
from utils import *
from operator import add

from pyspark import SparkContext

def Expectation(doc, beta, alpha, K, NumTotalDoc):
    '''
     * doc : current doc

     * beta is a matrix that: beta(z, w) represents the probility of Word w assigned with Topic z.

     * alpha is a vextor that: alpha(z) represents the average probility of topic z in all docunents.
    
     * K : number of topics
    
     * NumTotalDoc : total number of documents.
     
     return : current doc + phi of current doc, for input of Maximization.
    '''
    
    MAX_ITR = setting.E_MAX_ITR;
    terms = doc.strip().split(' ')
    N = len(terms)
    word_ids = [ int(x.split(':')[0]) for x in terms ]
    word_counts = [ float(x.split(':')[1]) for x in terms ]
    var_gamma = [ alpha + NumTotalDoc * 1.0 / K ] * K
    di_gamma = [digamma(alpha + NumTotalDoc * 1.0 / K)] * K
    phi = [ [ 1.0 / K ] * K ] * N
    

    converged = 1
    likelihood_old, i = 0, 0
    while i < MAX_ITR and not converged < setting.E_CON_THRES:
        for n in range(N):
            old_phi = phi[n]
            phi[n] = [ di_gamma[t] + beta[t][word_ids[t]] for t in range(K) ]
            phi_sum = log_sum(phi[n])
            phi[n] = [ math.exp( phi[n][t] - phi_sum ) for t in range(K) ]
            var_gamma = [var_gamma[t] + word_counts[n] * (phi[n][t] - old_phi[t]) for t in range(K)]
            di_gamma = [digamma(x) for x in var_gamma]
            
        likelihood = compute_likelihood(word_ids, word_counts, alpha, beta, phi, var_gamma, di_gamma, K)
        converged = (likelihood_old - likelihood) / (1+likelihood_old)
        likelihood_old = likelihood
        i = i+1
        
    res = [("likelihood", likelihood)]
    for i in range(N):
        for j in range(K):
            res.append(('{0},{1}'.format(word_ids[i], j), phi[i][j]))
    return res

def compute_likelihood(word_ids, word_counts, alpha, beta, phi, var_gamma, di_gamma, K):
    gamma_sum = sum(var_gamma)
    digamma_sum = digamma(gamma_sum)
    res = lgamma(alpha * K) - K * lgamma(alpha) -lgamma(gamma_sum)
    for k in range(K):
        res += (alpha - var_gamma[k]) * (di_gamma[k] - digamma_sum) - lgamma(var_gamma[k])
        for n in range(len(word_ids)):
            res += word_counts[n] * ( phi[n][k] * ( \
                    (di_gamma[k] - digamma_sum) - log(phi[n][k]) + beta[k][word_ids[n]]) )
    return res
    
def print_beta(line, beta):
    return [("beta", len(beta))]

def update_beta(M_res, beta):
    likelihood = 0
    for ele in Mres:
        if ele == 'likelihood':
            likelihood += float(ele[1])
            

if __name__=="__main__":
    beta = [[0.0001]*2000000]*10

    #for line in open(sys.argv[1]):
    #    print Expectation(doc = line, beta = beta , alpha = 0.1, K = 10, NumTotalDoc = 10)
    sc = SparkContext(appName="PythonWordCount")

    text = sc.textFile(sys.argv[1])

    text.cache()

    NumDoc = text.count()

    new_beta = text.flatMap(lambda line, beta=beta : Expectation(line, beta , 0.1, 10, NumDoc)).reduceByKey(add)
    #new_beta = text.flatMap(lambda line, beta=beta: print_beta(line, beta)).reduceByKey(add)

    output = new_beta.collect()

    for ele in output:
        print ele

    
