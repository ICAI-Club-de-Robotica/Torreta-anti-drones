import torch
import sys
import numpy as np

class DroneDetector():
    """ Detector de drones."""

    def __init__(self) -> None:
        # Configuración del modelo de deteccion
        yolo_path = f"{sys.path[0]}\\yolov5"
        weights_path = f"{sys.path[0]}\\exp\\weights\\best.pt"
        # Se establece el modelo de deteccion de drones utilizando los pesos entrenados
        self._model = torch.hub.load(yolo_path,'custom',path=weights_path,source='local')

    def detect(self, image: np.ndarray, conf_tres: float = 0.3):
        """ 
            Busca posibles los drones en la imagen proporcionada y devuelve los resultados de la detection con los drones encontrados .
        """
        # Se realiza la inferencia de la imgen y se obtienen los resultados
        results = self._model(image)
        # Extraccion de las coordenadas x e y del objecto detectado
        results = results.pandas().xywh[0]
        detection = []
        # Para cada clase detectada
        i = 0
        for r in range(0,len(results)):
            # Comprueba que la clase detectada sea 'drone'
            if results.name[r] == "drone":
                if results.confidence[r] >= conf_tres:
                    # Se añade a la lista
                    detection.append({"id": i,"xc":results.xcenter[r],"yc":results.ycenter[r],"w":results.width[r], \
                        "h":results.height[r],"con":results.confidence[r]})
                    i += 1
        # Se devuelven las posiciones de los objectos detectados
        return detection