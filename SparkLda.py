# lda training ... 

import sys
import math
import setting
from utils import *
import random
if len(sys.argv) > 2 and sys.argv[2] != "local" or len(sys.argv) <= 2:
    from operator import add
    from pyspark import SparkContext

def Expectation(doc, alpha, K):
    '''
     * doc : current doc

     * beta is a matrix that: beta(z, w) represents the probility of Word w assigned with Topic z.

     * alpha is a vextor that: alpha(z) represents the average probility of topic z in all docunents.
    
     * K : number of topics
    
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
    phi = [ [ 0.0 ] * K ] * N

    converged = 1
    likelihood_old, i = 0, 0
    beta = beta_global.value
    while i < MAX_ITR and not converged < setting.E_CON_THRES:
        for n in range(N):
            #phi[n] = [ math.exp(di_gamma[k]) * beta[k][word_ids[n]] for k in range(K) ]
            phi[n] = [ math.exp(di_gamma[k]) * beta[k][word_ids[n]] for k in range(K) ]
            phi_sum = sum(phi[n])
            phi[n] = [ phi[n][k] / phi_sum for k in range(K) ]
        for k in range(K):
            var_gamma[k] = alpha + sum([ word_counts[n] * phi[n][k] for n in range(N)])
            di_gamma[k] = digamma( var_gamma[k] )
        likelihood = compute_likelihood(word_ids, word_counts, alpha, beta, phi, var_gamma, di_gamma, K)
        converged = abs( (likelihood_old - likelihood) / (likelihood_old + 1e-8) )
        #print "like_old: {0}, like_new: {1}, converged: {2}".format(likelihood_old, likelihood, converged)
        likelihood_old = likelihood
        i = i+1
        
    res = [("likelihood", likelihood)]
    for i in range(N):
        for j in range(K):
            #print "CCC i:{0}, j:{1}".format(i,j)
            res.append(('{0},{1}'.format(j, word_ids[i]), phi[i][j]))
    return res

def compute_likelihood(word_ids, word_counts, alpha, beta, phi, var_gamma, di_gamma, K):
    gamma_sum = sum(var_gamma)
    digamma_sum = digamma(gamma_sum)
    res = lgamma(alpha * K) - K * lgamma(alpha) -lgamma(gamma_sum)
    for k in range(K):
        res += (alpha - var_gamma[k]) * (di_gamma[k] - digamma_sum) + lgamma(var_gamma[k])
        for n in range(len(word_ids)):
            if phi[n][k] > 0.0:
                res += word_counts[n] * 1.0 * ( phi[n][k] * ( \
                    (di_gamma[k] - digamma_sum) - math.log(phi[n][k]) + math.log(beta[k][word_ids[n]])) )
    return res
    
def update_beta(M_res, K, NumTerm):
    likelihood = 0.0
    beta = [ [ 1e-6 / NumTerm for i in range(NumTerm)] for j in range(K)]
    for ele in M_res:
        if ele[0] == 'likelihood':
            likelihood += float(ele[1])
        else:
            i = int(ele[0].split(',')[0])
            j = int(ele[0].split(',')[1])
            #print "XXX i:{0}, j:{1}, ele[1]: {2}, beta_i_j: {3}".format(i,j, ele[1], beta[i][j])
            beta[i][j] += float(ele[1])
    sum_beta = [sum(beta[i]) for i in range(K)]
    beta = [ [ beta[i][j]/sum_beta[i] for j in range(NumTerm) ] for i in range(K) ]
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
                res.append((pre_key, count))
            pre_key = line[0]
            count = float(line[1])
        else:
            count+=float(line[1])
    if pre_key:
        res.append((pre_key, count))
    return res

def rand_init_beta(NumTerm, K):
    #beta = [ [ random.random() + 1.0 / NumTerm for i in range(NumTerm) ] for j in range(K)]
    beta = [ [ 0.0 ] * NumTerm for j in range(K)]
    #sum_beta = [sum(beta[i]) for i in range(K)]
    #beta = [ [ beta[i][j]/sum_beta[i] for j in range(NumTerm) ] for i in range(K) ]
    #print beta
    return beta

if __name__=="__main__":

    NumTerm, K = setting.NUM_TERM, setting.K
    Alpha = setting.ALPHA

    # local test code start
    
    if len(sys.argv) > 2 and sys.argv[2] == "local":
        print "initialing beta ... "
        beta = rand_init_beta(NumTerm,  K)
        print "initialing beta success!  "
        for it in range(20):
            e_res = []
            i = 0
            for line in open(sys.argv[1]):
                print "Doc: "+ str(i)
                i = i + 1
                e_res += Expectation(line, beta , Alpha, K)
            e_res = sorted(e_res, key=lambda x:x[0])
            e_res = reduce(e_res)
            #print e_res
            (beta, likelihood) = update_beta(e_res, K, NumTerm)
        print beta
        sys.exit(0)

     
    # local test code end
    
    
    sc = SparkContext(appName="SparkLda")
    text = sc.textFile(sys.argv[1]).repartition(200)
    print "caching file ..."
    text.cache()
    print "counting file ..."
    NumDoc = text.count()
    likelihood, likelihood_old = 0, 0
    print "initialing beta ... "
    beta = rand_init_beta(NumTerm,  K)
    
    print "initialing beta success!  "
    
    # E_M iterate for beta
    for i in range(20):
        print "starting iteration {0} ...".format(i)
        print sys.getsizeof(beta)
        beta_global = sc.broadcast(beta)
        print "broadcast success"
        #new_beta = text.flatMap(lambda line, beta=beta : Expectation(line, beta , Alpha, K)).reduceByKey(add)
        new_beta = text.flatMap(lambda line :  Expectation(line, Alpha, K) ).reduceByKey(add)
        output = new_beta.collect()
        #print >> open('beta.'+str(i), 'w'), beta
        (beta, likelihood) = update_beta(output, K, NumTerm)
        #print beta
        print "likelihood {0} is {1}".format(i, likelihood)
    print >> open('beta.final', 'w'), beta
    
