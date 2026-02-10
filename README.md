# Temperature Prediction - TinyML on RP2040

<p align="right">
  <a href="README.md"><img src="https://img.shields.io/badge/lang-PT--BR-009c3b?style=flat-square&logo=googletranslate&logoColor=white" alt="Portugues"/></a>
  <a href="README.en.md"><img src="https://img.shields.io/badge/lang-EN-002868?style=flat-square&logo=googletranslate&logoColor=white" alt="English"/></a>
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Raspberry%20Pi%20Pico-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white" alt="Raspberry Pi Pico"/>
  <img src="https://img.shields.io/badge/TensorFlow%20Lite%20Micro-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Lite Micro"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/C%2FC%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C/C++"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/CMake-064F8C?style=for-the-badge&logo=cmake&logoColor=white" alt="CMake"/>
</p>

CNN 1D embarcada em um Raspberry Pi Pico (RP2040) para previsao de temperatura a +5, +10 e +15 minutos
usando dados de dois sensores I2C - AHT20 e BMP280.

---

## Como funciona

O firmware coleta amostras dos sensores a cada 31 segundos, mantem uma janela deslizante de
10 timesteps x 4 features (Temp_AHT20, Umid_AHT20, Temp_BMP280, Press_BMP280), normaliza os dados
com parametros Z-score gerados no treino e executa inferencia via TensorFlow Lite Micro.
As previsoes sao exibidas no serial e no display OLED SSD1306.

```
Sensores (I2C0) -> Buffer [10x4] -> Z-score -> CNN 1D (TFLM) -> Previsoes [+5, +10, +15 min]
```

---

## Estrutura do repositorio

```
temperature_mlp/
|-- data/
|   |-- temp.csv
|-- firmware/
|   |-- main.c
|   |-- tflm_wrapper.cpp/.h
|   |-- scaler_params.h
|   |-- lib/
|       |-- aht20.c/.h
|       |-- bmp280.c/.h
|       |-- ssd1306.c/.h
|       |-- font.h
|-- images/
|   |-- Conv1D/
|   |-- MLP/
|   |-- LSTM/
|   |-- GRU/
|-- models/
|   |-- Conv1D/
|   |-- MLP/
|   |-- LSTM/
|   |-- GRU/
|-- notebooks/
    |-- temperature_prediction_CNN_1D_tinyml.ipynb
    |-- temperature_prediction_MLP_tinyml.ipynb
    |-- temperature_prediction_LSTM_tinyml.ipynb
    |-- temperature_prediction_GRU_tinyml.ipynb
```

---

## Dataset e features

Fonte: [Vitoria da Conquista Weather Data - September 2025](https://www.kaggle.com/datasets/jonassouza872/vitoria-da-conquista-weather-data-september) (Kaggle)

O dataset original contem 82.432 medicoes coletadas ao longo de 30 dias (31/08 - 30/09/2025) no bairro Candeias, Vitoria da Conquista - BA, com 7 sensores de temperatura, 2 de umidade e 2 de pressao, registrando amostras a cada 30 segundos via Raspberry Pi Pico. Para este projeto, o dataset foi limpo e reduzido para manter apenas as colunas dos sensores AHT20 e BMP280 - os mesmos utilizados no hardware embarcado - resultando em 4 features temporais usadas no treinamento.

| Feature | Sensor | Descricao |
|---|---|---|
| Temp_AHT20 | AHT20 (I2C0 0x38) | Temperatura do ar (C) |
| Umid_AHT20 | AHT20 (I2C0 0x38) | Umidade relativa (%) |
| Temp_BMP280 | BMP280 (I2C0 0x76) | Temperatura barometrica (C) |
| Press_BMP280 | BMP280 (I2C0 0x76) | Pressao atmosferica (hPa) |

Janela temporal: **10 amostras** (~5 min de historico) -> previsao em **+5, +10 e +15 min**.

![Series temporais](images/Conv1D/01_series_temporais.png)

---

## Treinamento

| Parametro | Valor |
|---|---|
| Split | 70% treino / 15% validacao / 15% teste |
| Amostras de treino | 57.673 |
| Amostras de validacao | 12.359 |
| Amostras de teste | 12.359 |
| Epochs | 300 (max) |
| Batch size | 512 |
| Otimizador | Adam (lr = 0.0005) |
| Loss | MSE |
| Regularizacao | L2 (0.0001) + Dropout 20% |
| EarlyStopping | patience=50, monitor=val_loss, restore_best_weights=True |
| ReduceLROnPlateau | patience=20, factor=0.5, min_lr=1e-7 |

O scaler (StandardScaler) foi ajustado **somente no conjunto de treino** e aplicado nos demais splits para evitar data leakage. Outliers foram verificados pelo metodo IQR (1.5x) - nenhum removido. As sequencias foram geradas por janela deslizante de 10 timesteps sobre os dados normalizados, produzindo tensores de entrada no formato `[amostras, 10, 4]`.

---

## Comparacao de arquiteturas

Todas as arquiteturas foram treinadas com os mesmos dados, janela temporal e horizontes de previsao.

| Modelo | MAE | R2 | Parametros | TFLite | Deploy RP2040 |
|---|---|---|---|---|---|
| MLP | 0.3739 | 0.9834 | 1.891 | 6.86 KB | OK |
| **Conv1D** | **0.3370** | **0.9849** | **1.963** | **8.59 KB** | **OK** |
| LSTM | 0.3264 | 0.9881 | 3.235 | - | NAO |
| GRU | 0.2230 (*) | 0.9931 (*) | 2.611 | - | NAO |

> (*) Melhor metrica geral, mas **nao suportado pelo TFLite Micro** no RP2040.

### Por que Conv1D?

LSTM e GRU geram operacoes `TensorListReserve` nao implementadas no TFLite Micro para Cortex-M0+,
impossibilitando o deploy. Entre os modelos deployaveis:

- Conv1D reduz o MAE em **9,9%** vs MLP (0.337 vs 0.374)
- R2 superior em todos os horizontes de previsao
- Captura padroes temporais locais via filtros convolucionais - vantagem sobre o MLP
- Apenas **72 parametros a mais** e **+1.73 KB** no TFLite
- Cabe confortavelmente na flash e na arena de 60 KB do TFLM

---

## Resultados Conv1D

<table>
<tr>
  <td><img src="images/Conv1D/05_scatter_predictions.png" alt="Scatter predictions"/></td>
  <td><img src="images/Conv1D/07_error_distribution.png" alt="Distribuicao de erros"/></td>
</tr>
<tr>
  <td><img src="images/Conv1D/06_timeseries_predictions.png" alt="Serie temporal predita"/></td>
  <td><img src="images/Conv1D/04_curvas_aprendizado.png" alt="Curvas de aprendizado"/></td>
</tr>
</table>

---

## Hardware em operacao

<table>
<tr>
  <td><img src="images/Conv1D/08_Serial_Print.png" alt="Serial print previsoes"/></td>
  <td><img src="images/Conv1D/09_Predicao_Display.jpeg" alt="Display OLED previsoes"/></td>
</tr>
<tr>
  <td colspan="2" align="center"><img src="images/Conv1D/10_Circuito_Montado_BitDogLab.jpeg" alt="Circuito montado BitDogLab" width="60%"/></td>
</tr>
</table>

---

## Firmware - detalhes

**Modelo embarcado:** `temperature_model.h` gerado pelo `xxd -i` a partir do `.tflite`
**Arena TFLM:** 60 KB (~29 KB utilizados em runtime)
**Intervalo de coleta:** 31 s
**Normalizacao:** Z-score com `scaler_mean[]` e `scaler_scale[]` de `scaler_params.h`

**Pinagem:**

| Pino | Funcao |
|---|---|
| GP0 (SDA) / GP1 (SCL) | I2C0 - AHT20 + BMP280 @ 100 kHz |
| GP14 (SDA) / GP15 (SCL) | I2C1 - SSD1306 OLED @ 400 kHz |
