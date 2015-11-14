# lda training ... 

import math
import setting
from utils import *
import random
#from operator import add
#from pyspark import SparkContext

def Expectation(doc, beta, alpha, K):
    '''
     * doc : current doc

     * beta is a matrix that: beta(z, w) represents the probility of Word w assigned with Topic z.

     * alpha is a vextor that: alpha(z) represents the average probility of topic z in all docunents.
    
     * K : number of topics
    
     * NumDoc : total number of documents.
     
     return : current doc + phi of current doc, for input of Maximization.
    '''
    
    MAX_ITR = setting.E_MAX_ITR;
    terms = doc.strip().split(' ')
    N = len(terms)
    word_ids = [ int(x.split(':')[0]) for x in terms ]
    word_counts = [ float(x.split(':')[1]) for x in terms ]
    total_words = sum(word_counts)
    var_gamma = [ alpha +  total_words * 1.0 / K ] * K
    di_gamma = [digamma(alpha + total_words * 1.0 / K)] * K
    phi = [ [ math.log(1.0 / K) ] * K ] * N

    converged = 1
    likelihood_old, i = 0, 0
    while i < MAX_ITR and not converged < setting.E_CON_THRES:
        for n in range(N):
            old_phi = phi[n]
            phi[n] = [ di_gamma[t] + beta[t][word_ids[n]] for t in range(K) ]
            phi_sum = log_sum(phi[n])
            phi[n] = [ math.exp( phi[n][t] - phi_sum ) for t in range(K) ]
            var_gamma = [var_gamma[t] + word_counts[n] * (phi[n][t] - old_phi[t]) for t in range(K)]
            di_gamma = [digamma(x) for x in var_gamma]
        likelihood = compute_likelihood(word_ids, word_counts, alpha, beta, phi, var_gamma, di_gamma, K)
        converged = (likelihood_old - likelihood) / (likelihood_old - 1e-6)
        print "Like_old:{0}, like_new:{1}, converged:{2}".format(likelihood_old, likelihood, converged)   
        likelihood_old = likelihood
        print i
        i = i+1
        
    res = [("likelihood", likelihood)]
    for i in range(N):
        for j in range(K):
            res.append(('{0},{1}'.format(j, word_ids[i]), phi[i][j]))
    return res

def compute_likelihood(word_ids, word_counts, alpha, beta, phi, var_gamma, di_gamma, K):
    gamma_sum = sum(var_gamma)
    digamma_sum = digamma(gamma_sum)
    res = lgamma(alpha * K) - K * lgamma(alpha) -lgamma(gamma_sum)
    for k in range(K):
        res += (alpha - var_gamma[k]) * (di_gamma[k] - digamma_sum) - lgamma(var_gamma[k])
        for n in range(len(word_ids)):
            if phi[n][k] > 0:
                res += word_counts[n] * ( phi[n][k] * ( \
                    (di_gamma[k] - digamma_sum) - math.log(phi[n][k]) + beta[k][word_ids[n]]) )
    return res
    
def update_beta(M_res, K, NumTerm):
    likelihood = 0
    beta = [ [ 1e-6 / NumTerm ] * NumTerm] * K
    sum_beta = [0.0] * K
    for ele in M_res:
        if ele[0] == 'likelihood':
            likelihood += float(ele[1])
        else:
            i = int(ele[0].split(',')[0])
            j = int(ele[0].split(',')[1])
            beta[i][j] += float(ele[1])
            sum_beta[i] = float(ele[1])
    beta = [ [ math.log(beta[i][j]/sum_beta[i]) for j in range(NumTerm) ] for i in range(K) ]
    return (beta, likelihood)
            
def reduce(arr):
    res = []
    pre_key = ''
    count = 0
    for line in arr:
        if line[0] == 'likelihood':
            res.append(line)
            continue
        if pre_key == "" or pre_key != line[0]:
            if pre_key:
                res.append((line[0], count))
            pre_key = line[0]
            count = float(line[1])
        else:
            count+=float(line[1])
    return res

def rand_init_beta(NumTerm, K):
    beta = [ [ random.random() for i in range(NumTerm) ] for j in range(K)]
    sum_beta = [sum(beta[i]) for i in range(K)]
    beta = [ [ math.log(beta[i][j]/sum_beta[i]) for j in range(NumTerm) ] for i in range(K) ]
    return beta

if __name__=="__main__":

    NumTerm, K, NumDoc = 7, 2, 4
    Alpha = 0.1
    beta = rand_init_beta(NumTerm,  K)

    # local test code start
    for it in range(4):
        e_res = []
        i = 0
        for line in open(sys.argv[1]):
            print "Doc: "+ str(i)
            i = i + 1
            print beta
            e_res += Expectation(line, beta , Alpha, K)
        e_res = sorted(e_res, key=lambda x:x[0])
        e_res = reduce(e_res)
        (beta, likelihood) = update_beta(e_res, K, NumTerm)
        for ele in e_res:
            print ele[0], ele[1]

        
    # local test code end
    
    '''
    sc = SparkContext(appName="SparkLda")
    text = sc.textFile(sys.argv[1])
    text.cache()
    NumDoc = text.count()
    likelihood, likelihood_old = 0, 0
    
    # E_M iterate for beta
    for i in range(3):
        new_beta = text.flatMap(lambda line, beta=beta : Expectation(line, beta , Alpha, K)).reduceByKey(add)
        output = new_beta.collect()
        print >> open('beta.'+str(i), 'w'), beta
        (beta, likelihood) = update_beta(output, K, NumTerm)
        print "likelihood {0} is {1}".format(i, likelihood)
    '''
    
