import pandas as pd
import math
import random

# import for multinomial naive bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from collections import Counter
import seaborn as sns
# KNN
def knn():
    # Data creating manually
    data = 	[["man",180,80],
    ["woman",160,60],
    ["man",170,70],
    ["man",175,74],
    ["man",175,70],
    ["man",160,69],
    ["woman",170,68],
    ["man",170,55],
    ["woman",155,55],
    ["woman",150,54],
    ["woman",152,60],
    ["woman",165,60]]
    
    
    # Inputs from user
    boy = input("Enter height: \n")
    boy = float(boy)
    
    kilo = input("Enter weight: \n")
    kilo = float(kilo)
    
    yeni = [boy,kilo]
    
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(tablo,yeni):
    	distance = ((tablo[1] - yeni[0]) ** 2) + ((tablo[2] - yeni[1]) ** 2)
    	return math.sqrt(distance)
    
    for i in range(len(data)):
        data[i].append(euclidean_distance(data[i],yeni))
        
    data.sort(key = lambda data: data[3])
    
    # K value from user
    k = input("Please enter K value:\n")
    k = int(k)
    
    # Plurality Approach
    woman = 0
    man = 0
    for i in range(0,k):
        if data[i][0] == "man":
            man += 1
        elif data[i][0] == "woman":
            womann += 1
    
    if woman > man:
        print("Plurality vote: Gender = woman")
    else:
        print("Plurality vote: Gender = man")
    
    
    # Weighted Approach
    weights = []
    
    for i in range(0,k):
        if (data[i][3] != 0):
            weights.append([i,(1/(data[i][3])**2)])
        
    weights.sort(key = lambda data: data[1],reverse=True)
    
    print("Weighted vote: Gender = ", data[weights[0][0]][0])


def kmeans():
    
    
    dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)
    # make as panda dataframe for easy understanding
    df = pd.DataFrame(dataset, columns=['var1', 'var2'])
    df.head(2)
    
    sns.scatterplot(data=df, x="var1", y="var2")
    plt.show()
    
    model = KMeans()

    # visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
    
    
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
    
    
    Counter(kmeans.labels_)
    Counter({2: 50, 0: 50, 3: 50, 1: 50})
    
    
    
    
    sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                marker="X", c="r", s=80, label="centroids")
    plt.legend()
    plt.show()
                        

# PCA
def pca():
    dataX = 	[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]
    dataY = 	[2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]
    
    meanX = 0.0
    meanY = 0.0
    
    for i in range(len(dataX)):
        meanX += dataX[i]
        
    meanX =  meanX /len(dataX)
    
    for i in range(len(dataY)):
        meanY += dataY[i]
        
    meanY =  meanY /len(dataY)
            
    cov_matrix = []
    
    
    for i in range(len(dataX)):
        dataX[i] = dataX[i] - meanX
        
    for i in range(len(dataY)):
        dataY[i] = dataY[i] - meanY
    
    # X, X
    sum = 0.0
    for i in range(len(dataX)):
        sum = sum +(dataX[i] * dataX[i])
    
    sum = sum / (len(dataX)-1)
    
    cov_matrix.append(sum)
    
    # X, Y
    sum = 0.0
    for i in range(len(dataX)):
        sum = sum +(dataX[i] * dataY[i])
    
    sum = sum / (len(dataX)-1)
    
    cov_matrix.append(sum)
    
    # Y, X
    sum = 0.0
    for i in range(len(dataX)):
        sum = sum +(dataX[i] * dataY[i])
    
    sum = sum / (len(dataX)-1)
    
    cov_matrix.append(sum)
    
    # Y, Y
    sum = 0.0
    for i in range(len(dataY)):
        sum = sum +(dataY[i] * dataY[i])
    
    sum = sum / (len(dataX)-1)
    
    cov_matrix.append(sum)
    print("Covariance Matrix: ",cov_matrix)




# Gaussian Naive Bayes
def gaussiannaivebayes():
    data = 	[["R","H"],
        ["R","H"],
        ["B","E"],
        ["G","E"],
        ["G","E"],
        ["G","H"],
        ["B","E"],
        ["Y","H"],
        ["Y","E"],
        ["G","E"],
        ["Y","E"],
        ["B","E"],
        ["B","E"],
        ["G","H"]]
    
    
    # Tüm evetler
    evet = 0
    for i in range(len(data)):
        if data[i][1] == "E" :
            evet += 1
    
    # Tüm hayırlar
    hayir = 0
    for i in range(len(data)):
        if data[i][1] == "H" :
            hayir += 1
            
    # Tüm güneşliler
    gunesli= 0
    for i in range(len(data)):
        if data[i][0] == "G" :
            gunesli += 1
    
    # Tüm Yağmurlular
    yagmurlu = 0
    for i in range(len(data)):
        if data[i][0] == "Y" :
            yagmurlu += 1
            
    # Tüm Bulutlular
    bulutlu = 0
    for i in range(len(data)):
        if data[i][0] == "B" :
            bulutlu += 1
    
    
    
    # Güneşli evet oranı
    gunesliE = 0
    for i in range(len(data)):
        if data[i][0] == "G" and data[i][1] == "E" :
            gunesliE += 1
            
    # Güneşli hayır oranı
    gunesliH = 0
    for i in range(len(data)):
        if data[i][0] == "G" and data[i][1] == "H" :
            gunesliH += 1 
            
    # Bulutlu evet oranı
    bulutluE = 0
    for i in range(len(data)):
        if data[i][0] == "B" and data[i][1] == "E" :
            bulutluE += 1
            
    # Bulutlu hayır oranı
    bulutluH = 0
    for i in range(len(data)):
        if data[i][0] == "B" and data[i][1] == "H" :
            bulutluH += 1
            
    # Yağmurlu evet oranı
    yagmurluE = 0
    for i in range(len(data)):
        if data[i][0] == "Y" and data[i][1] == "E" :
            yagmurluE += 1
            
    
    # Yağmurlu hayır oranı
    yagmurluH = 0
    for i in range(len(data)):
        if data[i][0] == "Y" and data[i][1] == "H" :
            yagmurluH += 1
    
    
    weather = input("Enter weather condition:\n S for Sunny\n R for Rainy \n C for Cloudy\n")
    
    playrate = input("Futbol oynanacak mı?:\n Evet için E\n Hayır için H \n")
    
    print("\nOynama oranı\n")
    if weather == "G" and oynama == "E":
        print(((gunesliE/evet)*(evet/len(data))) / (gunesli/len(data)))
        
    if weather == "G" and oynama == "H":
        print( (gunesliH/hayir)*(hayir/len(data)) / (gunesli/len(data)) )
        
    if havadurumu == "Y" and oynama == "E":
        print(((yagmurluE/evet)*(evet/len(data))) / (yagmurlu/len(data)))
        
    if havadurumu == "Y" and oynama == "H":
        print(((yagmurluH/hayir)*(hayir/len(data))) / (yagmurlu/len(data)))
        
    if havadurumu == "B" and oynama == "E":
        print(((bulutluE/evet)*(evet/len(data))) / (bulutlu/len(data)))
        
    if havadurumu == "B" and oynama == "H":
        print(((bulutluH/hayir)*(hayir/len(data))) / (bulutlu/len(data)))
        


#Multinomial Naive Bayes
def multinomialnaivebayes():
    dataset = [["Chinese Beijing Chinese", "Ç"],
               ["Chinese Chinese Shangai", "Ç"],
               ["Chinese Macao Shangai", "Ç"],
                ["Tokyo Japan Chinese", "J"]]         
             
    dataset = pd.DataFrame(dataset)
    dataset.columns = ["Text", "Categories"]
    nltk.download('stopwords')
     
    corpus = []
     
    for i in range(0, 3):
        text = re.sub('[^a-zA-Z]', '', dataset['Text'][i])
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        text = ''.join(text)
        corpus.append(text)
        
    
    cv = CountVectorizer(max_features = 1500)
    
    X = dataset.iloc[:, 0].values
    y = dataset.iloc[:, 1].values
    train = X
    
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    model.fit(X, y)
    
    
    def predict_category(s, train=train, model=model):
        return model.predict([s])
    
    input_string = input("Test için bir string giriniz\n")
        
    print("Kategori: ",predict_category(input_string))

# Basit doğrusal regresyon

def regression():
    data = [[2,8],
            [6,5],
            [7,7],
            [9,4],
            [8,6]]
    
    xort = 0
    yort =0
    xy = 0
    xsqr = 0
    for i in range(len(data)):
        xy += (data[i][0] * data[i][1])
        xort += data[i][0]
        yort += data[i][1]
        xsqr += (data[i][0]**2)
    
    xort /= len(data)
    yort /= len(data)
    
    
    b = (xy - len(data) * xort * yort ) / (xsqr - len(data)*(xort**2))
    
    a = yort - b*xort
    
    input_string = input("Y tahmini yapabilmek için x değerini giriniz \n")
            
    y = a + b * int(input_string)
    print("Y değeri: ",y)
    
    y_head = []
    for i in range(len(data)):
        appending = a + b * data[i][0]
        y_head.append(appending)
        
    top = 0
    for i in range(len(data)):
        if len(data) < 30:
            top += ((data[i][1] - y_head[i]) **2)        
            s = (top/(len(data) -2)) ** (1/2)
    
    print("Gerçek değer denklemi = ",a,"+",b,"*","(x)","+",s)
    gercek_deger = a+b*int(input_string)+s
    print("\nGerçek değer: ",gercek_deger)




#MAIN

def indirect(i):
    switcher={
                1:knn,
                2:kmeans,
                3:pca,
                4:gaussiannaivebayes,
                5:multinomialnaivebayes,
                6:regression
                }
    func=switcher.get(i,lambda :'Invalid')
    return func()

i = -1
while(i != 0):
    i = input("KNN algoritması için 1'e\nK-Means Clustering algoritması için 2'ye\nPCA algoritması için 3'e\nGaussian Naive Bayes algoritması için 4'e\nMultinomial Naive Bayes algoritması için 5'e\nBasit doğrusal regresyon için 6'ya\nÇıkış için 0'a basınız\n")
    if i == "1":
        indirect(1)
    elif i == "2":
        indirect(2)
    elif i == "3":
        indirect(3)
    elif i == "4":
        indirect(4)
    elif i == "5":
        indirect(5)
    elif i == "6":
        indirect(6)
    else:
        i = 0
    
