# import the missing libraries
import unicodedata
import re
from typing import List


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def keyword_score(query: str, doc_text: str, keywords: List[str]) -> int:
    """
    Calcula un puntaje para un documento basado en coincidencias con palabras clave y la consulta.

    Args:
        query (str): La consulta del usuario.
        doc_text (str): El texto del documento a evaluar.
        keywords (List[str]): Lista de palabras clave a verificar en el documento.

    Returns:
        int: El puntaje calculado (valores más altos son mejores).
    """
    score = 0
    norm_doc = normalize_text(doc_text)
    norm_query = normalize_text(query)
    if norm_query in norm_doc:
        score += 2
    for kw in keywords:
        if normalize_text(kw) in norm_doc:
            score += 1
    return score


def rerank_docs(query: str, docs: list, keywords: List[str] = []) -> list:
    """
    Reordena una lista de documentos en función de embeddings y puntajes de palabras clave.

    Args:
        query (str): La consulta del usuario.
        docs (list): Lista de objetos documento (deben tener 'page_content' y opcionalmente 'similarity').
        keywords (List[str], opcional): Lista de palabras clave para usar en la evaluación. Por defecto [].

    Returns:
        list: La lista reordenada de documentos (mejor puntaje primero).
    """
    reranked = []
    for doc in docs:
        embedding_score = getattr(doc, "similarity", 0)
        k_score = keyword_score(query, doc.page_content, keywords)
        final_score = embedding_score + k_score
        reranked.append((final_score, doc))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in reranked]
