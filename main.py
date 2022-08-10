import csv
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

rows = []

with open('Final_data.csv', 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        rows.append(i)

headers = rows[0]
star_data = rows[1:]

star_masses = []
star_radiuses = []

for star in star_data:
    star_masses.append(star[3])
    star_radiuses.append(star[4])

X = []

for index, star_mass in enumerate(star_masses):
  temp_list = [star_radiuses[index], star_mass]
  X.append(temp_list)

WCSS = []

for i in range(1,11):
  k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
  k_means.fit(X)
  WCSS.append(k_means.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), WCSS, marker='o', color='red')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()