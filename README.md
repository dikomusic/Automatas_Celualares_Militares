# Simulador Táctico Militar: Operaciones Combinadas

Un entorno de simulación estocástica desarrollado en Python para predecir el resultado de operaciones militares. El sistema transforma mapas topográficos en matrices de navegación y utiliza Autómatas Celulares junto con el Algoritmo Evolutivo de Rechenberg para simular el comportamiento, desgaste y moral de las tropas bajo fuego.

**Institución:** Escuela Militar de Ingeniería (EMI) "Mcal. Antonio José de Sucre" - U.A.L.P.
**Asignatura:** Modelación y Simulación de Sistemas
**Gestión:** 2024

---

## ⚙️ Características Técnicas Principales

* **Procesamiento Topográfico Estocástico:** Lectura de mapas topográficos mediante espacios de color (HSV) para asignar matemáticamente costos de movimiento y bonos de cobertura.
* **Autómatas Celulares (Niebla de Guerra):** Entidades autónomas que evalúan su vecindad de Moore extendida para transitar entre estados discretos (Supresión, Retirada, Reorganización).
* **Optimización Evolutiva (Rechenberg 1/5):** El motor ejecuta simulaciones "invisibles" previas al despliegue para calibrar dinámicamente la agresividad y búsqueda de cobertura de las tropas.
* **Doctrina de las 6 Armas:** Implementación de mecánicas de armadura pesada (Caballería), bonos de radar (Comunicaciones) y regeneración de recursos (Logística).

---

## 🛠️ Requisitos del Sistema

Este proyecto fue desarrollado utilizando librerías estándar optimizadas para procesamiento matricial y renderizado gráfico 2D. No requiere software comercial de simulación.

* **Python:** Versión 3.10 o superior.
* **Numpy:** Para el cálculo de matrices y resolución de probabilidades (Bernoulli/Gauss).
* **Pygame (2.6+):** Para el motor de visualización táctica en tiempo real.

---

## 🚀 Instalación y Ejecución

Se recomienda estrictamente la creación de un entorno virtual para aislar las dependencias del proyecto. Ejecute los siguientes comandos en su terminal:

1. **Crear el entorno virtual:**
   ```bash
   python -m venv env
2. Activar el entorno virtual:
Windows: .\env\Scripts\activate

En macOS/Linux: source env/bin/activate

3. Instalar dependencias requeridas:


pip install numpy pygame

4. Ejecutar el simulador:


python main.py
