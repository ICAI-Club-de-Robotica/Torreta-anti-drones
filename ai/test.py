from drone_detector import DroneDetector
import cv2
import sys

def show_detection_results(input_img, detection):
    output_img = input_img
    # Para cada objeto detectado
    for j,drone in enumerate(detection):
        print(f"El drone{j} esta en ({drone['xc']},{drone['yc']})")
        p1 = (int(drone['xc']-drone['w']/2),int(drone['yc']-drone['h']/2))
        p2 = (int(drone['xc']+drone['w']/2),int(drone['yc']+drone['h']/2))
        # Enmarca el dron detectado con un rectangulo
        output_img =cv2.rectangle(output_img,p1,p2,(0,0,255),2)
    # Muestra los resultados 
    cv2.imshow("Detection",output_img)
    cv2.waitKey(1000)

if __name__ == "__main__":

    # Se crea el detector de drones
    dd = DroneDetector()
    # Se obtienen images
    for i in range(0,9):
        # Lectura de datos
        image = cv2.imread(f"{sys.path[0]}\\images\\drone{i}.jpg")  
        # Se realiza la detecccion
        detection = dd.detect(image)
        # Se muestran los resultados
        show_detection_results(image,detection)
    
    vc = cv2.VideoCapture(0)
    while True:
        # Se obtiene imagen de la camara
        retval, frame  = vc.read()
        # Se realiza la detecccion
        detection = dd.detect(frame,conf_tres=0.1)
        # Se muestran los resultados
        show_detection_results(frame,detection)
        