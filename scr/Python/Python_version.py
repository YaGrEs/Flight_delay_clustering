import tkinter as tk
from tkinter import filedialog, ttk

import os
import time
import psutil
import gc
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from minisom import MiniSom

import umap.umap_ as umap


FEATURES = [
    'DEP_DELAY',
    'CRS_ELAPSED_TIME',
    'DISTANCE',
    'DAY_OF_WEEK',
    'CARRIER_DELAY',
    'WEATHER_DELAY',
    'LATE_AIRCRAFT_DELAY',
    'NAS_DELAY',
    'OP_UNIQUE_CARRIER',
    'ORIGIN'
]

def load_and_preprocess(path, max_rows=200000):

    df = pd.read_csv(path)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        print(f"Sampled {max_rows} rows")

    df.columns = df.columns.str.strip()

    df = df[FEATURES]

    df = df.dropna()

    categorical = [
        'OP_UNIQUE_CARRIER',
        'ORIGIN'
    ]

    for col in categorical:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()

    X = scaler.fit_transform(df)

    return X, df

def describe_cluster(row):

    dep = row['DEP_DELAY']

    weather = row['WEATHER_DELAY']

    carrier = row['CARRIER_DELAY']

    nas = row['NAS_DELAY']

    late = row['LATE_AIRCRAFT_DELAY']

    distance = row['DISTANCE']


    if weather > 30:

        return 'Задержки из-за погодных условий'


    elif carrier > 30:

        return 'Задержки по вине авиакомпании'


    elif nas > 30:

        return 'Задержки из-за загруженности воздушного пространства'


    elif late > 30:

        return 'Задержки из-за позднего прибытия самолёта'


    elif distance > 2000 and dep > 20:

        return 'Задержки на дальнемагистральных рейсах'


    elif dep > 60:

        return 'Серьёзные системные задержки'


    elif dep < 10:

        return 'Рейсы без задержек'


    else:

        return 'Смешанные операционные задержки'

def som_clustering(X):

    som = MiniSom(
        3,
        3,
        X.shape[1],
        sigma=1.0,
        learning_rate=0.5
    )

    som.random_weights_init(X)

    som.train_random(
        X,
        1000
    )

    winners = np.array([
        som.winner(x)
        for x in X
    ])

    labels = np.array([
        w[0] * 3 + w[1]
        for w in winners
    ])

    return labels

def rf_clustering(X):

    y_dummy = np.random.randint(
        0,
        2,
        len(X)
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X, y_dummy)

    leaves = rf.apply(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = kmeans.fit_predict(leaves)

    return labels

def ocsvm_clustering(X):

    model = OneClassSVM(
        gamma='auto',
        nu=0.15
    )

    model.fit(X)

    labels = model.predict(X)

    return labels

def calculate_metrics(X, labels):

    if len(set(labels)) < 2:

        return {
            'Silhouette': 'N/A',
            'Davies-Bouldin': 'N/A',
            'Calinski-Harabasz': 'N/A'
        }

    return {

        'Silhouette':
            round(
                silhouette_score(X, labels),
                4
            ),

        'Davies-Bouldin':
            round(
                davies_bouldin_score(X, labels),
                4
            ),

        'Calinski-Harabasz':
            round(
                calinski_harabasz_score(X, labels),
                4
            )
    }

def save_plot(X, labels, algo, summary):
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        random_state=42
    )

    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(14, 8))

    unique_clusters = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        mask = labels == cluster
        cluster_name = summary.loc[cluster, 'Description']

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            s=15,
            alpha=0.8,
            label=f'Кластер {cluster}: {cluster_name}'
        )

    for cluster in unique_clusters:
        center = embedding[labels == cluster].mean(axis=0)
        cluster_name = summary.loc[cluster, 'Description']

        plt.text(
            center[0],
            center[1],
            f'Центр {cluster}',
            fontsize=9,
            fontweight='bold',
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='black'
            )
        )

    plt.title(
        f'{algo} - UMAP Clustering',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)

    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0.9,
        title='Кластеры',
        title_fontsize=10
    )

    plt.tight_layout()

    filename = f'{algo}_UMAP.png'

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5  # Добавляем отступы для легенды
    )

    plt.close()

    return filename

class App:

    def __init__(self, root):

        self.root = root

        self.root.title(
            'Flight Delay Clustering'
        )

        self.algorithm = tk.StringVar(
            value='SOM'
        )

        ttk.Label(
            root,
            text='Choose Algorithm'
        ).pack(pady=5)

        algo_box = ttk.Combobox(
            root,
            textvariable=self.algorithm,
            values=[
                'SOM',
                'RandomForest',
                'OneClassSVM'
            ]
        )

        algo_box.pack(pady=5)

        ttk.Button(
            root,
            text='Load CSV & Run',
            command=self.run
        ).pack(pady=10)

        self.output = tk.Text(
            root,
            width=90,
            height=35
        )

        self.output.pack(pady=10)

    def run(self):

        path = filedialog.askopenfilename()

        if not path:
            return

        X, df = load_and_preprocess(path)

        algo = self.algorithm.get()

        process = psutil.Process(os.getpid())

        gc.collect()

        memory_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()

        if algo == 'SOM':

            labels = som_clustering(X)

        elif algo == 'RandomForest':

            labels = rf_clustering(X)

        else:

            labels = ocsvm_clustering(X)

        end_time = time.time()

        execution_time = round(end_time - start_time, 4)

        gc.collect()

        memory_after = process.memory_info().rss / (1024 * 1024)

        memory_used = round(memory_after - memory_before, 4)

        if memory_used <= 0:
            memory_used = 0.01

        metrics = calculate_metrics(
            X,
            labels
        )

        df['Cluster'] = labels

        summary = df.groupby(
            'Cluster'
        ).mean(numeric_only=True)

        summary['Description'] = summary.apply(
            describe_cluster,
            axis=1
        )

        image_file = save_plot(
            X,
            labels,
            algo,
            summary
        )

        self.output.delete(
            '1.0',
            tk.END
        )

        self.output.insert(
            tk.END,
            f'Algorithm: {algo}\n\n'
        )

        self.output.insert(
            tk.END,
            f'Execution Time: '
            f'{execution_time} sec\n'
        )

        self.output.insert(
            tk.END,
            f'Memory Used: '
            f'{memory_used} MB\n\n'
        )

        self.output.insert(
            tk.END,
            'Metrics:\n\n'
        )

        for k, v in metrics.items():
            self.output.insert(
                tk.END,
                f'{k}: {v}\n'
            )

            self.output.insert(
                tk.END,
                '\nCluster Descriptions:\n\n'
            )

            for cluster_id, row in summary.iterrows():
                self.output.insert(
                    tk.END,
                    f'Cluster {cluster_id}: '
                    f"{row['Description']}\n"
                )

            self.output.insert(
                tk.END,
                f'\nPlot saved as:\n'
                f'{image_file}\n'
            )

root = tk.Tk()

app = App(root)

root.mainloop()