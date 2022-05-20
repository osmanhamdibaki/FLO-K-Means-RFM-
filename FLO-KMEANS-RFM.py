


import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv("datasets/flo_data_20K.csv")
df = df_.copy()

# tarih değişkenine çevirme
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()

km_df = rfm.copy()

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()


# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))


# recency_score ve frequency_score ve monetary_score'u tek bir değişken olarak ifade edilmesi ve RFM_SCORE olarak kaydedilmesi
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

rfm.head()

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

rfm = rfm.drop(["customer_id"], axis=1)
km_df = km_df.drop(["customer_id"], axis=1)
## DEĞİŞKENLERİN BİRBİRİNİ EZMEMESİ İÇİN STANDARTLAŞTIRMA YAPIYORUZ.

km_df.head()

sc = MinMaxScaler((0, 1))
km_df = sc.fit_transform(km_df)

km_df = pd.DataFrame(km_df)
km_df.columns = ['Recency', 'Frequency', 'Monetary']
km_df.head()

### OPTİMUM KÜME SAYISININ BELİRLENMESİ

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(km_df)
elbow.show()

elbow.elbow_value_
## optımum kume sayısı -->  5

## FİNAL CLUSTERLARIN OLUŞTURULMASI

kmeans = KMeans(n_clusters=5).fit(km_df)
clusters = kmeans.labels_

km_df["cluster_no"] = clusters

km_df["cluster_no"] = km_df["cluster_no"] + 1

km_df.head()

rfm.head()

km_df.reset_index(inplace=True)
rfm.reset_index(inplace=True)
km_rfm = rfm.merge(km_df, on="index", how="left")
km_rfm = km_rfm.drop(["Recency", "Frequency", "Monetary"],axis=1)
km_rfm.head()

dummy_df = km_rfm["cluster_no"]
dummy_df = pd.get_dummies(dummy_df)
dummy_df = dummy_df.reset_index()
km_rfm = pd.concat([km_rfm, dummy_df], axis=1)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

km_df["cluster_no"].value_counts()
km_rfm.groupby("segment").agg( { 1 : "sum",
                                 2 : "sum",
                                 3 : "sum",
                                 4 : "sum",
                                 5 : "sum"})

# Buradan elde edeceğimiz sonuç bize hangi kümelerde hangi segmentlere göre aksiyon alınacağını gösterecektir.
# 1.kümede sadece champions için aksiyon alınabilir. 5.kümede aksiyon ağırlığı champions ve loyal_customer
# segmentindekilere verilebilir şeklinde yorumlar yaparak hangi kümelerde hangi segmentlere ağırlık verileceğine
# karar verilebilir.
