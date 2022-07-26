# Machine Learning Methods

## K-Nearest Algorithm

<img src="/images/knn.png" alt="k-nearest neighbor" style="height: 300px; width:400px;"/>


🔸 A dataset which has categorical data (men-woman) has defined.
```
data = [["man",180,80], ["woman",160,60], ["man",170,70], ["man",175,74], ["man",175,70], ["man",160,69], ["woman",170,68], ["man",170,55], ["woman",155,55], ["woman",150,54], ["woman",152,60], ["woman",165,60]]
```

🔸 Requested input from user for weight and height and they have been assigned to new variable.
```
height = input("Enter height: \n")
height = float(height) 
weight = input("Enter weight: \n")
weight = float(weight) 
new = [height,weight]
```
🔸 A function has been created appropriate to dataset for calculate the eucledian distance. In here, input from user and dataset has been used as a parameter of function. In function,  differences beetween requested input from user and dataset rows has been calculated and taken its square root. As a result of this calculations, euclidian distance has been found.

```
def euclidean_distance(data,new):
 distance = ((data[1] - new[0]) ** 2) + ((data[2] - new[1]) ** 2)
    return math.sqrt(distance)
```
🔸 A new list has been created. Created list has been defined as data = [“kategori”, “boy”, “kilo”, “girilen değer ile öklit uzaklığı”]. And then, this list has been sorted by distances.

```
for i in range(len(new)):
    data[i].append(euclidean_distance(data[i],new)) 
data.sort(key = lambda data: data[3])
```

🔸 An input refers k value requested from user.

```
k = input("Please enter k value:\n")
k = int(k)
```

🔸 Here, calculations has been done with two diferent approaches:

* **Plurality Vote Approach:** Smallest values has been choose according to k value. Which categories of chosen one was more than others, this category determined as predicted category.

```
woman = 0
man = 0

for i in range(0,k):
    if data[i][0] == "man":
        man += 1
    elif data[i][0] == "woman":
        woman += 1 
if woman > man:
    print("Plurality vote: Gender = woman")
else:
    print("Plurality vote: Gender = man")
```    
* **Weighted Vote Approach:** Here, the smallest distances to k value has been calculated. As a result of calculation; the category that gives biggest value is assinged as category of prediction.
```
for i in range(0,k):
    if (data[i][3] != 0):
        weights.append([i,(1/(data[i][3])**2)]) 
weights.sort(key = lambda data: data[1],reverse=True)
print("Weighted vote: Gender = ", data[weights[0][0]][0])
```
## Principal Component Analysis (PCA)

<img src="/images/pca.png" alt="pca" style="height: 300px; width:400px;"/>

🔸 dataX and dataY determinde manually.
```
dataX = 	[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]

dataY = 	[2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]
```
 🔸 Average values of each column has been calculated. With this, “covariance matrix” will created.
```
for i in range(len(dataX)):
    meanX += dataX[i] 
meanX =  meanX /len(dataX)

for i in range(len(dataY)):
    meanY += dataY[i] 
meanY =  meanY /len(dataY)
```
🔸 The mean of each X and Y value was subtracted.
```
for i in range(len(dataX)):
    dataX[i] = dataX[i] - meanX
for i in range(len(dataY)):
    dataY[i] = dataY[i] - meanY
```
🔸 X,X; X,Y; Y,X; Y,Y values has been calculated for “covariance matrix” components and each value has been assigned its related component.


* X,X
```
sum = 0.0
for i in range(len(dataX)):
    sum = sum +(dataX[i] * dataX[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)
```
* X,Y
```
sum = 0.0
for i in range(len(dataX)):
    sum = sum +(dataX[i] * dataY[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)
```
* Y,X
```
sum = 0.0
for i in range(len(dataX)):
    sum = sum +(dataX[i] * dataY[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)
```
* Y,Y
```
sum = 0.0
for i in range(len(dataY)):
    sum = sum +(dataY[i] * dataY[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)
```
🔸 Printing “covariance matrix” to the screen.
```
print("Covariance Matrix: ",cov_matrix)
```
## K-means Clustering  

<img src="/images/k-means.png" alt="k-means clustering" style="height: 300px; width:400px;"/>

🔸 The dataset has 200 exampke and in start position, 4 "center" has been determined.
```
dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)
```
🔸 Columns of dataset, has 200 random examples, has been determined as "var1" and "var2".
```
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
sns.scatterplot(data=df, x="var1", y="var2")
plt.show()
```

🔸 K-means model has been determined. 
```
model = KMeans()
```
🔸 Dataframe that has been created with 4 clusters assign to the model with different parameters.
```
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
Counter(kmeans.labels_)
Counter({2: 50, 0: 50, 3: 50, 1: 50})
```
🔸 Results of training step, each cluster has been visualized with different colors. (With seaborn library)
```
sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()
```
## Gaussian Naive Bayes Method

<img src="/images/gaussian.png" alt="gaussian" style="height: 300px; width:400px;"/>

🔸 The first index of data has been determined as weather and the second one has been playing state of football. 
```
data = 	[["R","N"], ["R","N"], ["C","Y"], ["S","Y"], ["S","Y"], ["S","N"], ["C","Y"], ["R","N"], ["R","Y"], ["S","Y"], ["R","Y"], ["C","Y"], ["C","Y"], ["S","N"]]
```
* In here: 
    
    R = Rainy

    C = Cloudy

    S = Sunny

    Y = Yes

    N = No

Counter for whole yes number;
```
yes = 0
for i in range(len(data)):
    if data[i][1] == "Y" :
        yes += 1
```

Counter for whole no number;
```
no = 0
for i in range(len(data)):
    if data[i][1] == "N" :
        no += 1
```
Counter for whole "sunny" number;
```
sunny= 0
for i in range(len(data)):
    if data[i][0] == "S" :
        sunny += 1
```


Calculated counter of rainy days and assign to a counter called rainy.
```
rainy = 0
for i in range(len(data)):
    if data[i][0] == "R" :
        rainy += 1
```
Calculated counter of cloudy days and assign to a counter called cloudy.
```
cloudy = 0
for i in range(len(data)):
    if data[i][0] == "C" :
        cloudy += 1
```
Calculated counter of whole days has condition sunny and yes.
```
sunnyY = 0
for i in range(len(data)):
    if data[i][0] == "S" and data[i][1] == "Y" :
        sunnyY += 1
```

Calculated counter of whole days has condition sunny and no.
```
sunnyN = 0
for i in range(len(data)):
    if data[i][0] == "S" and data[i][1] == "N" :
        sunnyN += 1 
```
Calculated counter of whole days has condition cloudy and yes.
```
cloudyY = 0
for i in range(len(data)):
    if data[i][0] == "C" and data[i][1] == "Y" :
        cloudyY += 1
```
Calculated counter of whole days has condition cloudy and no.
```
cloudyN = 0
for i in range(len(data)):
    if data[i][0] == "C" and data[i][1] == "N" :
        cloudyN += 1
```
Calculated counter of whole days has condition rainy and yes.
```
rainyY = 0
for i in range(len(data)):
    if data[i][0] == "R" and data[i][1] == "Y" :
        rainyY += 1
```
Calculated counter of whole days has condition rainy and no.
```
rainyY = 0
for i in range(len(data)):
    if data[i][0] == "R" and data[i][1] == "Y" :
        rainyY += 1
```

🔸 An input taken from user for testing.
```
weather = input("Please input weather:\n S for sunny\n R for rainy\n C for cloudy\n")
playrate = input("Futbol oynanacak mı?:\n Evet için E\n Hayır için H \n")
```
🔸 Each state calculated with input taken from user.
```
if weather == "S" and playrate == "Y":
    print(((sunnyY/yes)*(yes/len(data))) / (sunny/len(data))) 
if weather == "S" and playrate == "N":
    print( (sunnyN/no)*(no/len(data)) / (sunny/len(data)) ) 
if weather == "R" and playrate == "Y":
    print(((rainyY/yes)*(yes/len(data))) / (rainy/len(data))) 
if weather == "R" and playrate == "N":
    print(((rainyN/no)*(no/len(data))) / (rainy/len(data))) 
if weather == "C" and playrate == "Y":
    print(((cloudyY/yes)*(yes/len(data))) / (cloudy/len(data))) 
if weather == "C" and playrate == "H":
    print(((cloudyN/no)*(no/len(data))) / (cloudy/len(data)))
```
## MULTINOMIAL NAIVE BAYES METHOD
🔸 Words and categories defined.
```
dataset = [["Chinese Beijing Chinese", "Ç"],
           ["Chinese Chinese Shangai", "Ç"],
           ["Chinese Macao Shangai", "Ç"],
            ["Tokyo Japan Chinese", "J"]]         
```
🔸 Dataset has been defined as dataframe and categories column as “Categories”; words column as "Text".
```
dataset = pd.DataFrame(dataset)
dataset.columns = ["Text", "Categories"]
```
🔸 Stopwords method has been used for extracting insignificant words.
```
nltk.download('stopwords')
```
Her bir kelimeler boşlukları silinerek ve büyük harfleri küçük harflere dönüştürülerek birleştirildi. Birleştirilen kelimeler corpus adındaki yapıya atandı ve bu sayede tanımlana yapılabilecek.
```
corpus = []
for i in range(0, 3):
    text = re.sub('[^a-zA-Z]', '', dataset['Text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = ''.join(text)
    corpus.append(text)
```
Scikit-learn tarafından sunulan “CountVectorizer” ile bir doküman kolestiyonu bir terim vektörüne dönüştürülür.
cv = CountVectorizer(max_features = 1500)

Verisetinde tanımlanan metinler  X olarak ve kategoriler y yani hedef olarak tanımlanmaktadır.
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
train = X

Eğitim yapılabilmesi için model yapılandırması yapılır ve tanımlanan X ve y değerleri modele fit edilir.
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

Tahmin yapılabilmesi için bir fonksiyon tanımlanır. Bu sayede kullanıcıdan alınan bilgiyi fonksyiona parametre olarak gönderdiğimizde bize bilginin hangi kategoriye ait olduğu değeri döndürülür.
def predict_category(s, train=train, model=model):
    return model.predict([s]
input_string = input("Test için bir string giriniz\n") 
print("Kategori: ",predict_category(input_string))

## BASİT DOĞRUSAL REGRESYON
Veriler “x” (bağımsız değişken) ve “y” (bağımlı değişken) ikilileri şeklinde tanımlandı.
data = [[2,8], [6,5], [7,7], [9,4], [8,6]]

“xy” değişkenine “x” ve “y” değerlerinin çarpımı atandı. “xort” ve “yort” değerlerine her bir x değerlerinin ortalaması ve y değerlernin ortalaması atandı. 
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

“b” (regresyon katsayısı) daha önce hesaplanan değerler kullanılarak hesaplanmaktadır.

b = (xy - len(data) * xort * yort ) / (xsqr - len(data)*(xort**2))

“a” (sabit) değeri de hesaplanan “b” katsayısı sayesinde hesaplanmaktadır.
a = yort - b*xort

Kullanıcıdan hedef tahmini yapılabilmesi için bir değer alınmaktadır. 
input_string = input("Y tahmini yapabilmek için x değerini giriniz \n")

Girilen değer, oluşturulan hesaplama denklemi kullanılarak bulunuyor.
y = a + b * int(input_string)
print("Y değeri: ",y)

Son olarak da hatalar bulunarak gerçek değer denkleminin hesaplanması yapıldı ve “y” gerçek değeri yazdırıldı.
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


