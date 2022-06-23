# Machine Learning Methods

## K-Nearest Algoritm
A dataset which has categorical data (men-woman) has defined.

data = [["men",180,80], ["woman",160,60], ["men",170,70], ["men",175,74], ["men",175,70], ["men",160,69], ["woman",170,68], ["men",170,55], ["woman",155,55], ["woman",150,54], ["woman",152,60], ["woman",165,60]]


Requested input from user for weight and height and they have been assigned to new variable.

boy = input("Enter height:\n")
boy = float(boy) 
kilo = input("Enter weight:\n")
kilo = float(kilo) 
yeni = [boy,kilo]

Öklit mesafesinin hesaplanması için kullanılan veri setine uygun bir hesaplama foksiyonu yazıldı. Burada parametre olarak kullanıcı tarafından alınan veriler gönderildi ve veri setindeki her bir kendi ilgili sütunu ile fark hesaplaması yapıldı. Bu sayede öklit uzaklığı bulundu.
def euclidean_distance(tablo,yeni):
 distance = ((tablo[1] - yeni[0]) ** 2) + ((tablo[2] - yeni[1]) ** 2)
    return math.sqrt(distance)

Yeni bir liste oluşturuldu. Oluşturulan liste data = [“kategori”, “boy”, “kilo”, “girilen değer ile öklit uzaklığı”] olarak tanımlandı. Daha sonrasında bu liste uzaklıklara göre sıralandı.
for i in range(len(data)):
    data[i].append(euclidean_distance(data[i],yeni)) 
data.sort(key = lambda data: data[3])

Kullanıcıdan bir k değeri girilmesi istendi.

k = input("K değerini giriniz:\n")
k = int(k)

Burada iki farklı yaklaşım ile hesaplama yapıldı:

[1]	Çoğulcul Yaklaşım: k değerine göre en küçükler seçildi ve seçilenlerin kategorisinde hangisi çoğunluktaysa onun kategorisi girilen değerin kategorisi olarak belirlendi.

kadin = 0

erkek = 0

for i in range(0,k):
    if data[i][0] == "erkek":
        erkek += 1
    elif data[i][0] == "kadin":
        kadin += 1 
if kadin > erkek:
    print("Plurality vote: Gender = kadin")
else:
    print("Plurality vote: Gender = erkek")
    
[2]	Ağırlıklı Oylama: Burada k değerine kadar en küçük uzaklığa sahip olanların hesaplaması yapıldı. Yapılan hesaplamada en yüksek değeri veren kategori, girilen değerin kategorisi olarak atandı.

for i in range(0,k):
    if (data[i][3] != 0):
        weights.append([i,(1/(data[i][3])**2)]) 
weights.sort(key = lambda data: data[1],reverse=True)
print("Weighted vote: Gender = ", data[weights[0][0]][0])

## TEMEL BİLEŞENLER ANALİZİ (PCA)

Kullanılacak verilerin tanımlanması yapıldı.

dataX = 	[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]

dataY = 	[2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]

Her bir sütunun ortalaması alındı bu şekilde “covariance matrix” hesaplaması yapılacak.
for i in range(len(dataX)):
    meanX += dataX[i] 
meanX =  meanX /len(dataX)
for i in range(len(dataY)):
    meanY += dataY[i] 
meanY =  meanY /len(dataY)
Her bir X ve Y değerinden ortalamaları çıkartıldı.
for i in range(len(dataX)):
    dataX[i] = dataX[i] - meanX
for i in range(len(dataY)):
    dataY[i] = dataY[i] - meanY
“covariance matrix” bileşenleri için X,X; X,Y; Y,X; Y,Y hesaplamaları gerçekleştirildi ve her biri kendi ilgili alanında “cov_matrix”’e atandı.

X,X

sum = 0.0
for i in range(len(dataX)):
    sum = sum +(dataX[i] * dataX[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)

X,Y

sum = 0.0
for i in range(len(dataX)):
    sum = sum +(dataX[i] * dataY[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)

Y,X

sum = 0.0
for i in range(len(dataX)):
    sum = sum +(dataX[i] * dataY[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)

Y,Y

sum = 0.0
for i in range(len(dataY)):
    sum = sum +(dataY[i] * dataY[i])
sum = sum / (len(dataX)-1)
cov_matrix.append(sum)

“covariance matrix” ekrana yazdırıldı.
print("Covariance Matrix: ",cov_matrix)

## K-MEANS KÜMELEME YÖNTEMİ
Veri seti 200 tane örnek içermektedir ve başlangıçta 4 tane “center” tanımlanmaktadır.
dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)
200 tane rastgele örnekler içeren verisetinin sütunları “var1” ve “var2” olarak tanımlanmaktadır. Bu veriler görselleştirilmektedir.
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
sns.scatterplot(data=df, x="var1", y="var2")
plt.show()

“K-means” modeli tanımlanmaktadır. 
model = KMeans()

4 cluster oluşrurularak oluşturulan “dataframe” modele belirli parametrelerde atanmaktadır.
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)
Counter(kmeans.labels_)
Counter({2: 50, 0: 50, 3: 50, 1: 50})

Eğitim sonucunda oluşan her bir küme farklı olarak renklendirilip ekrana bastırılmaktadır.
sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show() 
GAUSSIAN NAIVE BAYES YÖNTEMİ
Verinin ilk içeri “index”i hava durumu ve ikinci “index”i futbol oynanma durumunu belirtmektedir.
data = 	[["Y","H"], ["Y","H"], ["B","E"], ["G","E"], ["G","E"], ["G","H"], ["B","E"], ["Y","H"], ["Y","E"], ["G","E"], ["Y","E"], ["B","E"], ["B","E"], ["G","H"]]
Bütüm “evet”lerin sayımı yapılmaktadır ve “evet” adındaki bir sayıcıya atılmaktadır.
evet = 0
for i in range(len(data)):
    if data[i][1] == "E" :
        evet += 1
Bütüm “hayirlar”ların sayımı yapılmaktadır ve “hayir” adındaki bir sayıcıya atılmaktadır.
hayir = 0
for i in range(len(data)):
    if data[i][1] == "H" :
        hayir += 1

    Bütüm “gunesli”lerin sayımı yapılmaktadır ve “gunesli” adındaki bir sayıcıya atılmaktadır.
gunesli= 0
for i in range(len(data)):
    if data[i][0] == "G" :
        gunesli += 1

Bütüm “yagmurlu”ların sayımı yapılmaktadır ve “yagmurlu” adındaki bir sayıcıya atılmaktadır.
yagmurlu = 0
for i in range(len(data)):
    if data[i][0] == "Y" :
        yagmurlu += 1

Bütüm “bulutlu”ların sayımı yapılmaktadır ve “bulutlu” adındaki bir sayıcıya atılmaktadır.
bulutlu = 0
for i in range(len(data)):
    if data[i][0] == "B" :
        bulutlu += 1

Bütüm “gunesli evet”lerin sayımı yapılmaktadır ve “gunesliE” adındaki bir sayıcıya atılmaktadır.
gunesliE = 0
for i in range(len(data)):
    if data[i][0] == "G" and data[i][1] == "E" :
        gunesliE += 1

Bütüm “gunesli hayır”lerin sayımı yapılmaktadır ve “gunesliE” adındaki bir sayıcıya atılmaktadır.
gunesliH = 0
for i in range(len(data)):
    if data[i][0] == "G" and data[i][1] == "H" :
        gunesliH += 1 

Bütüm “bulutlu evet”lerin sayımı yapılmaktadır ve “bulutluE” adındaki bir sayıcıya atılmaktadır.
bulutluE = 0
for i in range(len(data)):
    if data[i][0] == "B" and data[i][1] == "E" :
        bulutluE += 1

Bütüm “bulutlu hayır”ların sayımı yapılmaktadır ve “bulutluH” adındaki bir sayıcıya atılmaktadır.
bulutluH = 0
for i in range(len(data)):
    if data[i][0] == "B" and data[i][1] == "H" :
        bulutluH += 1

Bütüm “yagmurlu evet”lerin sayımı yapılmaktadır ve “yagmurluE” adındaki bir sayıcıya atılmaktadır.
yagmurluE = 0
for i in range(len(data)):
    if data[i][0] == "Y" and data[i][1] == "E" :
        yagmurluE += 1
        
Bütüm “yağmurlu hayır”ların sayımı yapılmaktadır ve “yagmurluH” adındaki bir sayıcıya atılmaktadır.
yagmurluH = 0
for i in range(len(data)):
    if data[i][0] == "Y" and data[i][1] == "H" :
        yagmurluH += 1

Kullanıcıdan hava durumu ve merak ettiği oynanma durumu için bir girdi alındı.
havadurumu = input("Hava durumunu giriniz:\n Güneşli için G\n Yağmurlu için Y \n Bulutlu için B\n")
oynama = input("Futbol oynanacak mı?:\n Evet için E\n Hayır için H \n")

Kullanıcıdan alınan değerlere göre her bir durum bağıntısının ortalamaları hesaplandı ve hesaplanan değerler kullanıcıya döndürüldü.
if havadurumu == "G" and oynama == "E":
    print(((gunesliE/evet)*(evet/len(data))) / (gunesli/len(data))) 
if havadurumu == "G" and oynama == "H":
    print( (gunesliH/hayir)*(hayir/len(data)) / (gunesli/len(data)) ) 
if havadurumu == "Y" and oynama == "E":
    print(((yagmurluE/evet)*(evet/len(data))) / (yagmurlu/len(data))) 
if havadurumu == "Y" and oynama == "H":
    print(((yagmurluH/hayir)*(hayir/len(data))) / (yagmurlu/len(data))) 
if havadurumu == "B" and oynama == "E":
    print(((bulutluE/evet)*(evet/len(data))) / (bulutlu/len(data))) 
if havadurumu == "B" and oynama == "H":
    print(((bulutluH/hayir)*(hayir/len(data))) / (bulutlu/len(data)))

## MULTINOMIAL NAIVE BAYES YÖNTEMİ
Kategorilerine uygun kelimelerin bulunduğu veriseti tanımlandı.
dataset = [["Chinese Beijing Chinese", "Ç"],
           ["Chinese Chinese Shangai", "Ç"],
           ["Chinese Macao Shangai", "Ç"],
            ["Tokyo Japan Chinese", "J"]]         

Tanımlanan veriseti “dataframe” olarak tanımlandı ve kelimelin bulunduğu sütun “Text” olarak, kategorilerin olduğu sütun “Categories” olarak tanımlandı.
dataset = pd.DataFrame(dataset)
dataset.columns = ["Text", "Categories"]

Verisetinde bulunan anlamsız kelimelerin çıkarılabilmesi için “stopwords” kullanılmaktadır. 
nltk.download('stopwords')

Her bir kelimeler boşlukları silinerek ve büyük harfleri küçük harflere dönüştürülerek birleştirildi. Birleştirilen kelimeler corpus adındaki yapıya atandı ve bu sayede tanımlana yapılabilecek.
corpus = []
for i in range(0, 3):
    text = re.sub('[^a-zA-Z]', '', dataset['Text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = ''.join(text)
    corpus.append(text)

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


