import random
import time

import numpy as np
import pandas as pd
from scipy import stats


def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an NxD pandas DataFrame
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    n, d = data.shape
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
    newCost=1
    oldCost=0
    while abs(oldCost-newCost)>eps:
        oldCost=newCost
        #E step:
        Carray=np.zeros((n,k))
        for i in range(n):
            Carray[i,np.argmin(np.sum(np.square(np.tile(data[i,],(k,1))-mu),axis=1))]=1
        n_hat=np.sum(Carray,axis=0)
        P=n_hat/n
        newCost = 0
        for i in range(k):
            mu[i,:]= np.dot(Carray[:,i],data)/n_hat[i]
            # summed squared distance of points in the cluster from the mean
            dist = np.dot(Carray[:,i],np.sum((data-np.tile(mu[i,:],(n,1)))**2,axis=1))
            newCost += dist
        clusters=np.nonzero(Carray)[1]
    # print (newCost)

    return (mu,clusters)

class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    @property
    def bic(self):
        """
        Computes the Bayesian Information Criterion for the trained model.
        Note: `n_train` and `max_ll` set during @see{fit} may be useful
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)


    def Gaussian(self,x, mu, var):
        d = x.shape[0]
        res = -np.sum((x-mu)**2)/(2*var)
        res = np.exp(res)
        res = res/np.power(2*np.pi*var, d/2.0)
        return res

    def e_step(self, data):
        n,d = np.shape(data) # n data points of dimension d
        k=self.k
        P=self.pi
        Mu=self.mu
        probs = np.zeros((n,k)) # posterior probabilities to compute
        LL = 0    # the LogLikelihood
        # print ("var= ", self.sigsq)
        for i in range(n):
            sumPoint = 0
            x = data.iloc[:,i]
            
            for j in range(k):
                # print ("mu = ", Mu[j])
                # print ("var = ", self.sigsq)
                # print ("point: ", [x])
                sumPoint += P[j]*self.Gaussian(x, np.array(Mu[j]), self.sigsq[j]) #stats.multivariate_normal.pdf(x, Mu[j], self.sigsq[j].)
                # print(sumPoint)
            LL += np.log(sumPoint)
            for j in range(k):
                
                probs[i, j] = P[j]*self.Gaussian(x, Mu[j].flatten(), self.sigsq[j])/sumPoint #stats.multivariate_normal.pdf(x, Mu[j].flatten(), self.sigsq[j].flatten()[0])/sumPoint        
        return (LL, probs)

    def m_step(self, data, pz_x):
        p_z=pz_x
        n,d = np.shape(data) # n data points of dimension d
        K=self.k
        N = np.zeros((K,1))
        sigsq=np.zeros((K,1))

        N[:,0] = np.sum(p_z, axis=0)
        # print("N = ", N)
        # print("d = ", d)
        # print("P_Z = ", p_z)
        Pi = N/n
        Mu = np.dot(p_z.transpose(), data)
        Mu = np.divide(Mu, N)
        for j in range(K):
            mean = Mu[j, :]
            # print("Mean = ", mean)
            x_mu = np.sum((data-mean)**2, axis=1)
            # print("data-mean = ", x_mu)
            sigsq[j,0] = (np.dot(p_z[:,j], x_mu))/(2*N[j])
        # print("sigsq = ", sigsq)
        # print("MU: ", Mu)
        # print("Pi: ", Pi)
        return {
            'pi': Pi,
            'mu': Mu,
            'sigsq': sigsq,
        }

    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        # print(self.params['sigsq'])
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]

    def e_step(self, data):
        n, d=np.shape(data)
        for i in range(d):
            dummies=np.array(pd.get_dummies(data.iloc[:,i], dummy_na=True)) #take dummy matrix with NaN col
            self.alpha[i] = np.c_[self.alpha[i], np.ones(self.k)]
            alpha_trans = np.transpose(self.alpha[i])
            product = np.dot(dummies, alpha_trans)
            
            if i<1:
                P_init = product #p(x|x, k, a)
            else:
                P_init = np.multiply(P_init, product)
        # print(data, "\n\n=========\n\n", P_init)
        P = np.multiply(P_init, self.pi) #unnormalized posterior prob

        #normalize:
        P_row_sum = np.array([np.sum(P, axis=1)]).transpose()
        P_norm = np.divide(P, P_row_sum)
        # print(P, "\n\n--------- \n", P_norm ,"\n\n---------\n", np.sum(P_norm, axis=1))
        
        ####LL part ####
        logpi = self.pi
        logpi[self.pi==0] = 1
        logpi = np.log(logpi)

        ll1 = np.sum(np.dot(P_norm, logpi))
        
        P_init[P_init==0]=1
        logP_init = np.log(P_init)
        ll2=np.sum(np.multiply(P_norm, logP_init))

        ll=ll1+ll2

        # print("Log likelihood = ",ll)

        return(ll, P_norm)


    def m_step(self, data, p_z):
        pi = np.sum(p_z, axis=0)
        n,d  = data.shape
        new_pi=pi/n

        new_alpha=self.alpha
        for i in range(d):
            dummies = np.array(pd.get_dummies(data.iloc[:,i]))
            new_alpha[i] = np.dot(p_z.transpose(), dummies)
            rowsum = np.array([np.sum(new_alpha[i], axis=1)]).transpose()
            new_alpha[i] = np.divide(new_alpha[i], rowsum)

        return {
            'pi': new_pi,
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        p=0
        l=self.max_ll
        n=self.n_train
        for i in range(len(self.alpha)):
            p+=self.k*(np.shape(self.alpha[i])[0]-1)
        p+=self.k
        p+=len(self.pi)-1
        return (l-0.5*p*np.log(n))


        
