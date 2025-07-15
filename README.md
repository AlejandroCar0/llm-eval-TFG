---

# LLM Profiler

🚀 Herramienta para caracterizar modelos de lenguaje (LLMs) con monitorización en tiempo real y análisis de recursos mediante **Ollama**, **Prometheus** y **Streamlit**.

---

## 📋 Tabla de Contenidos

* [🎯 Resumen](#-resumen)
* [✨ Funcionalidades](#-funcionalidades)
* [🔧 Requisitos Previos](#-requisitos-previos)
* [🚀 Instalación](#-instalación)
* [🎮 Inicio Rápido](#-inicio-rápido)
* [📖 Uso](#-uso)
* [📊 Panel Interactivo](#-panel-interactivo)
* [📁 Estructura del Proyecto](#-estructura-del-proyecto)
* [🤝 Contribuir](#-contribuir)
* [📄 Licencia](#-licencia)
* [🙏 Agradecimientos](#-agradecimientos)

---

## 🎯 Resumen

Este proyecto proporciona un marco robusto para:

* Evaluar el rendimiento de LLMs en máquinas remotas
* Monitorizar recursos del sistema (CPU, GPU, memoria) durante inferencias
* Analizar respuestas del modelo y obtener puntuaciones
* Visualizar resultados mediante un panel interactivo

---

## ✨ Funcionalidades

* 🔗 **Evaluación Remota**: Ejecución de pruebas en máquinas remotas vía SSH
* 📡 **Monitorización en Tiempo Real**: Seguimiento de recursos con Prometheus
* 🎮 **Métricas de GPU**: Uso y memoria de GPU monitorizados
* 📊 **Panel Interactivo**: Visualización en Streamlit
* 🤖 **Soporte Multimodelo**: Evaluación de varios modelos mediante Ollama
* 📝 **Registro Detallado**: Logs extensos para depuración y análisis

---

## 🔧 Requisitos Previos

* Python 3.8+
* Acceso SSH a la(s) máquina(s) remota(s)
* [Prometheus](https://prometheus.io) (para recolección de métricas)
* [Ollama](https://ollama.com) (para inferencia LLM)

---

## 🚀 Instalación

### 1. Instalar Prometheus

#### macOS (ARM64):

```bash
curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.darwin-arm64.tar.gz
tar -xvf prometheus-3.3.1.darwin-arm64.tar.gz
sudo mv ./prometheus-3.3.1.darwin-arm64/prometheus /usr/local/bin/
sudo mv ./prometheus-3.3.1.darwin-arm64/promtool /usr/local/bin/
```

#### Linux (AMD64):

```bash
curl -LO https://github.com/prometheus/prometheus/releases/download/v3.3.1/prometheus-3.3.1.linux-amd64.tar.gz
tar -xvf prometheus-3.3.1.linux-amd64.tar.gz
sudo mv ./prometheus-3.3.1.linux-amd64/prometheus /usr/local/bin/
sudo mv ./prometheus-3.3.1.linux-amd64/promtool /usr/local/bin/
```

#### Verificar instalación:

```bash
prometheus --version
```

#### Instalación alternativa (local):

```bash
export PATH=$HOME/prometheus/:$PATH
```

---

### 2. Configurar el entorno Python

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🎮 Inicio Rápido

1. Configura tu experimento:

```bash
python3 llm_llama_eval.py --help
```

2. Ejecuta la evaluación:

```bash
python3 llm_llama_eval.py --host <ip-remota> --user <usuario> --key <ruta-clave.pem>
```

3. Lanza el panel:

```bash
streamlit run data_representation/dashboard.py
```

---

## 📖 Uso

### Script principal: `llm_llama_eval.py`

```bash
python3 llm_llama_eval.py [OPCIONES]
```

| Opción                         | Obligatorio | Descripción                       |
| ------------------------------ | ----------- | --------------------------------- |
| `-i`, `--ip-address TEXT`      | ✅           | IP de la máquina remota           |
| `-u`, `--user TEXT`            | Opcional    | Usuario SSH                       |
| `-p`, `--password TEXT`        | Opcional    | Contraseña SSH                    |
| `-pk`, `--private-key TEXT`    | Opcional    | Ruta a la clave privada (.pem)    |
| `-ov`, `--ollama-version TEXT` | Opcional    | Versión de Ollama                 |
| `-nv`, `--node-version TEXT`   | Opcional    | Versión del Node Exporter         |
| `-ro`, `--reinstall-ollama`    | Opcional    | Fuerza la reinstalación de Ollama |

#### Ejemplos:

**Con clave privada**:

```bash
python3 llm_llama_eval.py -i 192.168.1.100 -u ubuntu -pk ./clave.pem
```

**Con contraseña**:

```bash
python3 llm_llama_eval.py -i 192.168.1.100 -u ubuntu -p mipassword
```

**Con versiones específicas**:

```bash
python3 llm_llama_eval.py -i 192.168.1.100 -u ubuntu -pk ./clave.pem -ov v0.1.32 -nv v1.5.0
```

---

### Script del Panel: `dashboard.py`

```bash
streamlit run data_representation/dashboard.py [OPCIONES]
```

| Opción              | Descripción                                      |
| ------------------- | ------------------------------------------------ |
| `-d`, `--directory` | Directorio que contiene los archivos de métricas |
| `-h`, `--help`      | Muestra la ayuda                                 |

---

## 📊 Panel Interactivo

El panel proporciona:

* ⏱️ Métricas de rendimiento: tiempos de respuesta, puntuaciones
* 📈 Uso de recursos: CPU, memoria, GPU
* 🧠 Comparativas: análisis lado a lado de modelos
* 📤 Exportación: descarga de resultados en varios formatos

### Modos de ejecución:

#### Opción 1: Subida manual

```bash
streamlit run data_representation/dashboard.py
```

Abre en [http://localhost:8501](http://localhost:8501) y sube manualmente tus archivos en este orden:

1. Métricas de Ollama
2. Puntuaciones de modelos
3. Métricas de Prometheus
4. Información general

---

#### Opción 2: Carga automática desde directorio

```bash
streamlit run data_representation/dashboard.py -- --directory ./experiment_results/2025-07-11-10-30-15
```

Archivos requeridos en el directorio:

* `ollama_metrics.csv` o `ollama_metrics*.csv`
* `models_score.csv` o `*score*.csv`
* `prometheus_metrics.csv` o `prometheus_metrics*.csv`
* `general_info.txt` o `general_info*.txt`

---

## 📁 Estructura del Proyecto

```
llm-eval-TFG/
├── data_representation/     # Panel e interfaz
│   └── dashboard.py
├── experiment_results/      # Resultados almacenados
├── gpu_exporter/            # Recolección de métricas GPU
├── logger/                  # Utilidades de logging
├── metrics/                 # Métricas recolectadas
├── ollama/                  # Integración con Ollama
├── prometheus/              # Configuración de Prometheus
├── versions/                # Gestión de versiones
├── llm_llama_eval.py        # Script principal
├── requirements.txt         # Dependencias Python
└── README.md                # Este archivo
```

---

## 🤝 Contribuir

1. Haz un fork del repositorio
2. Crea una rama de feature:

   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Realiza tus cambios y haz commit:

   ```bash
   git commit -m "Agrega nueva funcionalidad"
   ```
4. Haz push:

   ```bash
   git push origin feature/nueva-funcionalidad
   ```
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

---

## 🙏 Agradecimientos

* [Prometheus](https://prometheus.io) por la monitorización
* [Ollama](https://ollama.com) por la inferencia de LLMs
* [Streamlit](https://streamlit.io) por la interfaz visual

---
