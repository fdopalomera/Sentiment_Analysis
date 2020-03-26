import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from time import time


def stem_txt(text):
    
    porter = PorterStemmer()
    tokens = word_tokenize(text)
    stems = [porter.stem(word) for word in tokens]
    stem_sentence = " ".join(stems)
    
    return stem_sentence


def lemma_txt(text):

    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemma = [lemma.lemmatize(word) for word in tokens]
    lemma_sentence = " ".join(lemma)
    
    return lemma_sentence


def text_preprocessor(text, text_normalize=lemma_txt):
    text = text.lower()
    text = re.sub('\[.*?¿\]\%', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[''"",,,<<>>]', '', text)
    text = re.sub('\n', '', text)
    text = lemma_txt(text)
    return text


# Función get_words
def get_words(df, target, text, vectorizer, n_words=20):

    # Iterador
    for label in df[target].unique():
        
        # Se instancia el vecotrizador
        genre_series = df[df[target] == label][text]
        vectorizer_class = vectorizer
        disperse_matrix = vectorizer_class.fit_transform(genre_series)

        # Generar variables con las palabras y sus cantidades
        words = vectorizer_class.get_feature_names()
        count_words = disperse_matrix.toarray().sum(axis=0)

        # Creación de DataFrame y posterior orden
        df_words = pd.DataFrame({'word': words, 'frequency': count_words})
        label_words = df_words.sort_values('frequency', ascending=False)[:n_words]['word'].to_list()
        print('\n{}\n'.format(label),label_words)
    
    return 

def clf_metrics(clf, X_train, y_train, X_test, y_test):
    """
    Imprime un reporte con las métricas de problemas de clasificación clásicas:

    """    
    tic = time()
   # Corrroboración preproceso target
    if (y_train.dtype =='object') & (y_test.dtype == 'object'):

        lbl_encoder = LabelEncoder()
        y_train = lbl_encoder.fit_transform(y_train)
        y_test = lbl_encoder.transform(y_test)

    # Entrenar el modelo
    clf.fit(X_train, y_train)
    # Imprimir mejores parámetros sí el objeto 
    if isinstance(clf, GridSearchCV):
        print(clf.best_params_)
    # Predecir la muestra de validación
    y_hat = clf.predict(X_test)
    # Métricas
    metrics = {'ROC_Score': roc_auc_score(y_test, y_hat).round(3),
               'Confusion_Matrix': confusion_matrix(y_test, y_hat).round(3),
               'Classification_Report': classification_report(y_test, y_hat)}
    for key, value in metrics.items():
        print('{}:\n{}'.format(key, value))
    return print("Realizado en {:.3f}s".format(time() - tic))


def compare_classifiers(estimators, X_test, y_test, n_cols=2):

    """
    Compara en forma gráfica las métricas de clasificación a partir de una lista de 
    tuplas con los modelos (nombre_modelo, modelo_entrendo) 
    """

    rows = np.ceil(len(estimators)/n_cols)
    height = 2 * rows
    width = n_cols * 5
    fig = plt.figure(figsize=(width, height))

    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for n, model in enumerate(estimators):

        y_hat = model[1].predict(X_test) 
        dc = classification_report(y_test, y_hat, output_dict=True)

        plt.subplot(rows, n_cols, n + 1)

        for i, j in enumerate(['0', '1', 'macro avg']):

            tmp = {'0': {'marker': 'x', 'label': f'Class: {j}'},
                   '1': {'marker': 'x', 'label': f'Class: {j}'},
                   'macro avg': {'marker': 'o', 'label': 'Avg'}}

            plt.plot(dc[j]['precision'], [1], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['recall'], [2], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['f1-score'], [3], marker=tmp[j]['marker'],color=colors[i], label=tmp[j]['label'])
            plt.axvline(x=.5, ls='--')

        plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])
        plt.title(model[0])
        plt.xlim((0.1, 1.0))

        if (n + 1) % 2 == 0:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    fig.tight_layout()
    
    return