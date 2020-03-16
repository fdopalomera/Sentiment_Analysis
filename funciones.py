import re
import string
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
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
    # Entrenar el modelo
    clf.fit(X_train, y_train)
    # Imprimir mejores parámetros
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