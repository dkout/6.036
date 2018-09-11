from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    loss_sum=0
    for point in range(len(feature_matrix)):
        if labels[point]*(np.dot(theta, feature_matrix[point])+theta_0)<1:
            loss_sum+=1-labels[point]*(np.dot(theta, feature_matrix[point])+theta_0)
    return loss_sum/len(feature_matrix)


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label*(np.dot(feature_vector, current_theta)+current_theta_0)<=0:
        current_theta+=label*feature_vector
        current_theta_0+=label
    return (current_theta, current_theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Section 1.4a
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """

    theta=np.zeros(len(feature_matrix[0]))
    theta_0=0
    for t in range(T):
        for point in range(len(feature_matrix)):
            label=labels[point]
            feature_vector=feature_matrix[point]
            if label*(np.dot(feature_vector, theta)+theta_0)<=0:
                theta+=label*feature_vector
                theta_0+=label
    return (theta, theta_0)


def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.4b
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    theta=np.zeros(len(feature_matrix[0]))
    theta_0=0
    theta_sum=0
    theta_0_sum=0
    for t in range(T):
        for point in range(len(feature_matrix)):
            label=labels[point]
            feature_vector=feature_matrix[point]
            if label*(np.dot(feature_vector, theta)+theta_0)<=0:
                theta+=label*feature_vector
                theta_0+=label
            theta_sum+=theta
            theta_0_sum+=theta_0
    return (theta_sum/(T*len(feature_matrix[0])), theta_0/(T*len(feature_matrix[0])))

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label*(np.dot(current_theta, feature_vector)+ current_theta_0)<=1:
        current_theta=(1-L*eta)*current_theta+eta*label*feature_vector
        current_theta_0+=eta*label
    else:
        current_theta=(1-L*eta)*current_theta

    return(current_theta, current_theta_0)


def pegasos(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta=np.zeros(len(feature_matrix[0]))
    theta_0=0
    i=0
    for t in range(T):
        for point in range(len(feature_matrix)):
            i+=1
            label=labels[point]
            feature_vector=feature_matrix[point]
            eta=1/np.sqrt(i)#(point+1)*(t+1))
            (theta, theta_0)=pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)
    return (theta, theta_0)




### Part II

def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    l=[]
    for point in range(len(feature_matrix)):
        if np.dot(theta, feature_matrix[point])+theta_0<=0:
            l.append(-1)
        else:
            l.append(1)
    return (np.array(l))

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (theta, theta_0)=perceptron(train_feature_matrix, train_labels, T)
    train_classification=classify(train_feature_matrix, theta, theta_0)
    val_classification = classify(val_feature_matrix, theta, theta_0)
    return(accuracy(train_classification, train_labels), accuracy(val_classification, val_labels))

def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (theta, theta_0)=average_perceptron(train_feature_matrix, train_labels, T)
    train_classification=classify(train_feature_matrix, theta, theta_0)
    val_classification = classify(val_feature_matrix, theta, theta_0)
    return(accuracy(train_classification, train_labels), accuracy(val_classification, val_labels))

def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Section 2.9
    Trains a linear classifier using the pegasos algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (theta, theta_0)=pegasos(train_feature_matrix, train_labels, T, L)
    train_classification=classify(train_feature_matrix, theta, theta_0)
    val_classification = classify(val_feature_matrix, theta, theta_0)
    return(accuracy(train_classification, train_labels), accuracy(val_classification, val_labels))

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    wordlist=input_string.lower().split()
    bigrams=[]
    trigrams=[]
    for i in range(len(wordlist)):
        if i>2 and i%2==0:
            bigrams.append(' '.join((wordlist[i-1], wordlist[i])))
        if i>3 and i%3==0:
            trigrams.append(' '.join((wordlist[i-2], wordlist[i-1], wordlist[i])))
    return wordlist+bigrams+trigrams

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    stopwords={'.',',',' ', 'these', 'off', 'doing', 'theirs', 'than', 'too', 'up', 'most', 'nor', 'while', 'itself', 'few', 'can', 'before', 'how', 's', 'not', \
    'some', 'ourselves', 'on', 'such', 'herself', 'their', 'against', 'himself', 'she', 'that', 'between', 'each', 'other', 't', 'will', 'its', 'been', \
    'which', 'should', 'from', 'her', 'has', 'they', 'my', 'ours', 'any', 'into', 'under', 'myself', 'our', 'now', 'we', 'where', 'as', 'at', 'did', 'am', \
    'both', 'because', 'him', 'until', 'same', 'after', 'were', 'it', 'then', 'who', 'be', 'is', 'by', 'i', 'having', 'more', 'and', 'your', 'have', 'are', \
    'during', 'all', 'you', 'no', 'just', 'those', 'through', 'of', 'whom', 'about', 'so', 'the', 'an', 'his', 'or', 'when', 'do', 'was', 'only', 'once', 'further',\
     'over', 'very', 'yourselves', 'above', 'again', 'me', 'does', 'them', 'what', 'here', 'there', 'yourself', 'he', 'if', 'don', 'but', 'in', 'to', 'a', 'with', 'for', 'hers',\
     'why', 'yours', 'below', 'down', 'had', 'this', 'own', 'being', 'out', 'themselves', '(',')'}

    dictionary = {} # maps word to unique index
    word_count={}
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in stopwords:
                word_count[word]=word_count.get(word, 0)+1
    for word in word_count:
        if word_count[word]<5*len(texts):
            dictionary[word] = len(dictionary)
    return dictionary

    # dictionary = {} # maps word to unique index
    # for text in texts:
    #     word_list = extract_words(text)
    #     for word in word_list:
    #         if word not in dictionary:
    #             dictionary[word] = len(dictionary)
    # return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
        feature_matrix[i]=feature_matrix[i]#/np.linalg.norm(feature_matrix[i])
    return feature_matrix

def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """
    add_feature_matrix=np.ndarray((len(reviews), 0))
    total=0
    for i in range(len(reviews)):
        total+=len(reviews[i])
    avg=total/len(reviews)
    for i in range(len(reviews)):
        if len(reviews[i])>=avg:
            add_feature_matrix[i]=1
        else:
            add_feature_matrix[i]=0
    return add_feature_matrix

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    final=np.hstack((bow_feature_matrix, additional_feature_matrix))
    for i in range(len(final)):
        final[i]=final[i]/np.linalg.norm(final[i])
    return final

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
