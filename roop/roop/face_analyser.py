import threading
from typing import Any, Optional, List
import insightface
import numpy as np
import cv2

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    # Especificar las nuevas dimensiones
   
    # Obtener todas las caras
    many_faces = get_many_faces(frame)
    if many_faces:
        print(f"Se detectaron {len(many_faces)} caras.")
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    else:
        print("No se encontraron caras en la imagen.")
    return None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        face_analyser = get_face_analyser()
        faces = face_analyser.get(frame)
        return faces
    except ValueError as e:
        print(f"Error en la detección de caras: {e}")
        return None

def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = np.sum(np.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
