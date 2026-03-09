# CNN desde cero con NumPy

Implementación de una **Convolutional Neural Network (CNN)** construida **desde cero** usando **NumPy**, orientada a entender el flujo completo de una red convolucional sin depender de frameworks de alto nivel para la arquitectura principal.

El proyecto incluye:

- **Convolución 2D manual**
- **Max Pooling manual**
- **Capa Softmax densa**
- **Forward pass y backward pass**
- **Entrenamiento sobre MNIST**
- **Visualizaciones de mapas de activación y predicciones**

> Objetivo del repositorio: aprender cómo funciona una CNN por dentro, no solo usarla como caja negra.

---

## Contenido del repositorio

- `cnn.ipynb`: notebook principal con la implementación completa de la CNN, entrenamiento y visualizaciones.
- `README.md`: descripción general del proyecto.

---

## Arquitectura implementada

La red sigue una estructura sencilla pero muy didáctica:

```text
Input (1 x 28 x 28)
    ↓
Conv2D (8 filtros de 3x3)
    ↓
Max Pooling (2x2)
    ↓
Flatten
    ↓
Softmax (10 clases)
```

Pensada para clasificación de dígitos manuscritos de **MNIST**.

---

## Componentes principales

### 1. `Conv2d`
Capa convolucional implementada manualmente.

Características:
- número configurable de filtros (`out_channels`)
- tamaño de kernel configurable (`kernelh`, `kernelw`)
- soporte opcional de `padding`
- inicialización tipo **He**, apropiada para activaciones estilo ReLU
- cálculo explícito de regiones y convolución por suma ponderada
- versión con `backprop` para actualizar filtros manualmente

### 2. `Pooling`
Implementación de **Max Pooling** manual.

Características:
- tamaño de pooling configurable
- selección explícita del máximo por región
- `backprop` devolviendo gradiente solo a los máximos activados

### 3. `Softmax`
Capa final totalmente conectada con salida probabilística.

Características:
- aplanado manual de la entrada
- proyección lineal a 10 clases
- softmax para obtener probabilidades
- retropropagación para actualizar pesos y sesgos

---

## Dataset

Se utiliza el dataset **MNIST** cargado desde `tensorflow.keras.datasets`.

- imágenes en escala de grises de `28x28`
- 10 clases: dígitos del `0` al `9`

En el notebook, las imágenes se normalizan con:

```python
x = (image / 255.0) - 0.5
```

Y se reformatean a:

```python
(1, 28, 28)
```

para representar explícitamente el canal.

---

## Entrenamiento

El notebook incluye un bucle de entrenamiento manual con:

- `cross-entropy loss`
- cálculo explícito del gradiente inicial
- propagación inversa por:
  - `Softmax`
  - `Pooling`
  - `Conv2d`
- actualización por descenso de gradiente simple

Configuración usada en la versión actual del notebook:

```python
num_epochs = 1
max_steps_per_epoch = 3000
lr = 0.005
```

---

## Resultados observados

Según la ejecución guardada en el notebook:

- **Test Loss:** `0.4596`
- **Test Accuracy:** `0.851`

Es decir, aproximadamente **85.1% de accuracy** en un subconjunto de test de 2000 imágenes tras un entrenamiento corto.

Esto no compite con una CNN moderna de producción, pero para una implementación manual desde cero está **muy bien como prueba de comprensión estructural**.

---

## Visualización incluida

El proyecto no solo clasifica: también muestra visualmente lo que pasa dentro de la red.

Incluye funciones para:

- visualizar la imagen de entrada
- mostrar los **feature maps** generados por los filtros
- ver las **probabilidades por clase**
- comparar **predicción vs etiqueta real**
- crear salidas útiles para demostraciones o vídeos

Esto le da al repo un valor didáctico extra: no solo entrenas la red, sino que **ves qué está detectando**.

---

## Cómo ejecutarlo

### Requisitos

```bash
pip install numpy matplotlib tensorflow
```

### Ejecución

1. Clona el repositorio:

```bash
git clone https://github.com/davidba10/CNN.git
cd CNN
```

2. Abre el notebook:

```bash
jupyter notebook cnn.ipynb
```

3. Ejecuta las celdas en orden.

---

## Qué demuestra este proyecto

Este repositorio demuestra comprensión práctica de:

- operación de convolución
- extracción local de características
- reducción espacial con pooling
- clasificación final con softmax
- backpropagation en redes convolucionales sencillas
- entrenamiento manual de una CNN sin abstractions mágicas

En otras palabras: aquí no se llama a una CNN ya hecha. Aquí se construye la máquina a mano, tornillo por tornillo.

---

## Limitaciones actuales

Como proyecto de aprendizaje, tiene varias limitaciones razonables:

- todo está concentrado en un notebook
- la implementación está pensada para claridad, no para velocidad
- no hay separación formal en módulos (`layers.py`, `train.py`, etc.)
- no hay tests automáticos ni gradient checking
- la arquitectura es deliberadamente simple
- se usa `tensorflow.keras.datasets` solo para cargar MNIST

---

## Posibles mejoras futuras

Algunas mejoras naturales para una siguiente versión:

- separar capas en archivos Python reutilizables
- añadir activaciones explícitas (por ejemplo, ReLU tras convolución)
- incorporar mini-batches reales
- añadir gradient checking numérico
- comparar resultados contra una implementación equivalente en PyTorch o Keras
- guardar pesos y permitir inferencia posterior
- añadir métricas más completas y visualización de errores
- refactorizar el notebook para dejar una versión más limpia de portfolio

---

## Motivación

Construir una CNN desde cero obliga a entender cosas que los frameworks suelen ocultar:

- cómo se mueven los tensores
- qué forma tiene cada salida
- dónde nace cada gradiente
- qué parte del input activa cada filtro
- cómo fluye la información desde píxeles hasta probabilidades

Y esa es justamente la gracia del proyecto.

---

## Autor

**David**  
Repositorio: `davidba10/CNN`
