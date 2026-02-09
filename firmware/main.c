#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "tflm_wrapper.h"
#include "scaler_params.h"  // Parâmetros de normalização (mean e scale)
#include "ssd1306.h"
#include "font.h"
#include "aht20.h"
#include "bmp280.h"

// Configurações do modelo
#define WINDOW_SIZE 10      // Janela temporal: 10 amostras
#define NUM_FEATURES 4      // 4 features: Temp_AHT20, Umid_AHT20, Temp_BMP280, Press_BMP280
#define NUM_HORIZONS 3      // 3 previsões: 5min, 10min, 15min
#define SAMPLE_INTERVAL_MS 31000  // Intervalo de amostragem: 31 segundos

// Display OLED
ssd1306_t display;

// Buffer da janela temporal: 10 timesteps × 4 features
static float sensor_window[WINDOW_SIZE][NUM_FEATURES];
static int window_pos = 0;
static bool window_full = false;

// Parâmetros de calibração do BMP280 (globais)
static struct bmp280_calib_param bmp_params;

// ============================================================================
// FUNÇÕES DOS SENSORES (VOCÊ DEVE IMPLEMENTAR)
// ============================================================================

// Lê temperatura e umidade do sensor AHT20
// Retorna: 0 se OK, -1 se erro
int read_aht20(float* temp_c, float* humidity_pct) {
    AHT20_Data dados_aht;

    if (!aht20_read(i2c0, &dados_aht)) {  // Sensores usam i2c0
        return -1;  // Erro na leitura
    }

    *temp_c = dados_aht.temperature;
    *humidity_pct = dados_aht.humidity;
    return 0;
}

// Lê temperatura e pressão do sensor BMP280
// Retorna: 0 se OK, -1 se erro
int read_bmp280(float* temp_c, float* pressure_hpa) {
    int32_t temp_raw, press_raw;

    // Lê dados brutos do BMP280
    bmp280_read_raw(i2c0, &temp_raw, &press_raw);  // Sensores usam i2c0

    // Converte usando parâmetros de calibração
    int32_t temp_conv = bmp280_convert_temp(temp_raw, &bmp_params);
    int32_t press_conv = bmp280_convert_pressure(press_raw, temp_raw, &bmp_params);

    // Converte para float
    *temp_c = temp_conv / 100.0f;  // Divide por 100 para obter °C
    *pressure_hpa = press_conv / 100.0f;  // Converte Pa para hPa

    return 0;
}

// ============================================================================
// NORMALIZAÇÃO DOS DADOS
// ============================================================================

// Normaliza uma feature usando os parâmetros do scaler (Z-score normalization)
// Formula: (x - mean) / scale
void normalize_feature(float* feature, int feature_idx) {
    *feature = (*feature - scaler_mean[feature_idx]) / scaler_scale[feature_idx];
}

// ============================================================================
// INFERÊNCIA
// ============================================================================

// Executa a predição de temperatura
void run_temperature_prediction(void) {
    // Obtém ponteiros dos tensores de entrada e saída
    int in_size, out_size;
    float* input = tflm_input_ptr(&in_size);
    float* output = tflm_output_ptr(&out_size);

    if (!input || !output) {
        printf("ERRO: Tensores inválidos\n");
        return;
    }

    printf("\n=== Nova Predição ===\n");

    // Copia janela temporal normalizada para o tensor de entrada
    // Input shape: [1, 10, 4] = 40 floats
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            input[t * NUM_FEATURES + f] = sensor_window[t][f];
        }
    }

    // Executa inferência
    int rc = tflm_invoke();
    if (rc != 0) {
        printf("ERRO tflm_invoke: %d\n", rc);
        return;
    }

    // Resultados
    printf("Previsões de Temperatura (AHT20):\n");
    printf("  +5 min:  %.2f °C\n", output[0]);
    printf("  +10 min: %.2f °C\n", output[1]);
    printf("  +15 min: %.2f °C\n", output[2]);

    // TODO: Atualizar display OLED com as previsões
    ssd1306_fill(&display, false);
    char line[24];
    snprintf(line, sizeof(line), "Temp Prediction");
    ssd1306_draw_string(&display, line, 0, 0, false);

    snprintf(line, sizeof(line), "+5m:  %.1fC", output[0]);
    ssd1306_draw_string(&display, line, 0, 16, false);

    snprintf(line, sizeof(line), "+10m: %.1fC", output[1]);
    ssd1306_draw_string(&display, line, 0, 28, false);

    snprintf(line, sizeof(line), "+15m: %.1fC", output[2]);
    ssd1306_draw_string(&display, line, 0, 40, false);

    ssd1306_send_data(&display);
}

// ============================================================================
// COLETA DE DADOS
// ============================================================================

// Coleta uma amostra dos sensores e adiciona na janela temporal
int collect_sensor_sample(void) {
    float temp_aht20, humidity_aht20;
    float temp_bmp280, pressure_bmp280;

    // Lê sensores
    if (read_aht20(&temp_aht20, &humidity_aht20) != 0) {
        printf("ERRO: Falha ao ler AHT20\n");
        return -1;
    }

    if (read_bmp280(&temp_bmp280, &pressure_bmp280) != 0) {
        printf("ERRO: Falha ao ler BMP280\n");
        return -1;
    }

    printf("Sensores: AHT20=%.2f°C %.2f%% | BMP280=%.2f°C %.2fhPa\n",
           temp_aht20, humidity_aht20, temp_bmp280, pressure_bmp280);

    // Armazena valores brutos na janela
    sensor_window[window_pos][0] = temp_aht20;
    sensor_window[window_pos][1] = humidity_aht20;
    sensor_window[window_pos][2] = temp_bmp280;
    sensor_window[window_pos][3] = pressure_bmp280;

    // Normaliza cada feature
    for (int f = 0; f < NUM_FEATURES; f++) {
        normalize_feature(&sensor_window[window_pos][f], f);
    }

    // Avança posição da janela (circular buffer)
    window_pos = (window_pos + 1) % WINDOW_SIZE;

    // Marca janela como cheia após 10 amostras
    if (window_pos == 0 && !window_full) {
        window_full = true;
        printf("Janela temporal completa! Iniciando predições...\n");
    }

    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    stdio_init_all();
    sleep_ms(2000);

    printf("\n╔════════════════════════════════════════╗\n");
    printf("║  Temperature Prediction - CNN 1D      ║\n");
    printf("║  Raspberry Pi Pico + TFLite Micro    ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    // ========================================================================
    // Configurar I2C para sensores e display
    // ========================================================================
    printf("Inicializando I2C...\n");

    // I2C0: Sensores (AHT20 + BMP280) - GP0 (SDA) e GP1 (SCL)
    i2c_init(i2c0, 100 * 1000);  // 100kHz para sensores
    gpio_set_function(0, GPIO_FUNC_I2C);  // GP0 = SDA
    gpio_set_function(1, GPIO_FUNC_I2C);  // GP1 = SCL
    gpio_pull_up(0);
    gpio_pull_up(1);

    // I2C1: Display OLED - GP14 (SDA) e GP15 (SCL)
    i2c_init(i2c1, 400 * 1000);  // 400kHz para display
    gpio_set_function(14, GPIO_FUNC_I2C);  // GP14 = SDA
    gpio_set_function(15, GPIO_FUNC_I2C);  // GP15 = SCL
    gpio_pull_up(14);
    gpio_pull_up(15);

    // ========================================================================
    // Inicializar sensores AHT20 e BMP280
    // ========================================================================
    printf("Inicializando sensores...\n");

    // Inicializa AHT20
    if (!aht20_init(i2c0)) {
        printf("ERRO: Falha ao inicializar AHT20!\n");
    } else {
        printf("AHT20 inicializado com sucesso!\n");
    }

    // Inicializa BMP280
    bmp280_init(i2c0);
    bmp280_get_calib_params(i2c0, &bmp_params);
    printf("BMP280 inicializado com sucesso!\n");

    // ========================================================================
    // Inicializar Display OLED
    // ========================================================================
    printf("Inicializando display...\n");
    ssd1306_init(&display, 128, 64, false, 0x3C, i2c1);
    ssd1306_config(&display);
    ssd1306_fill(&display, false);
    ssd1306_draw_string(&display, "TinyML Temp", 0, 0, false);
    ssd1306_draw_string(&display, "CNN 1D Model", 0, 16, false);
    ssd1306_draw_string(&display, "Inicializando...", 0, 32, false);
    ssd1306_send_data(&display);

    // ========================================================================
    // Inicializar TensorFlow Lite Micro
    // ========================================================================
    printf("Inicializando TensorFlow Lite Micro...\n");
    int rc = tflm_init();
    if (rc != 0) {
        printf("ERRO tflm_init: %d\n", rc);

        // Mensagens de erro detalhadas
        switch(rc) {
            case 1: printf("Erro: Modelo nao encontrado!\n"); break;
            case 2: printf("Erro: Versao do schema incompativel!\n"); break;
            case 3: printf("Erro: AllocateTensors falhou (memoria?)!\n"); break;
            case 4: printf("Erro: Tensores input/output nulos!\n"); break;
            case 5: printf("Erro: Tipo do tensor de entrada incorreto!\n"); break;
            case 6: printf("Erro: Tipo do tensor de saida incorreto!\n"); break;
            default: printf("Erro desconhecido!\n"); break;
        }

        ssd1306_fill(&display, false);
        ssd1306_draw_string(&display, "ERROR!", 0, 0, false);
        char msg[32];
        snprintf(msg, sizeof(msg), "TFLM Init: %d", rc);
        ssd1306_draw_string(&display, msg, 0, 20, false);
        ssd1306_send_data(&display);
        while (1) tight_loop_contents();
    }

    printf("TFLM OK - Arena usado: %d bytes\n\n", tflm_arena_used_bytes());

    // ========================================================================
    // Display pronto
    // ========================================================================
    ssd1306_fill(&display, false);
    ssd1306_draw_string(&display, "PRONTO!", 0, 0, false);
    ssd1306_draw_string(&display, "Coletando dados", 0, 16, false);
    ssd1306_draw_string(&display, "janela temporal", 0, 28, false);
    ssd1306_draw_string(&display, "(10 amostras)", 0, 40, false);
    ssd1306_send_data(&display);

    printf("Sistema pronto!\n");
    printf("Coletando amostras a cada %d segundos...\n", SAMPLE_INTERVAL_MS/1000);
    printf("Aguardando janela temporal completa (10 amostras)...\n\n");

    // ========================================================================
    // Loop principal
    // ========================================================================
    absolute_time_t last_sample_time = get_absolute_time();

    while (1) {
        // Verifica se passou o intervalo de amostragem
        int64_t elapsed_ms = absolute_time_diff_us(last_sample_time, get_absolute_time()) / 1000;

        if (elapsed_ms >= SAMPLE_INTERVAL_MS) {
            // Coleta nova amostra
            if (collect_sensor_sample() == 0) {
                // Se janela está cheia, executa predição
                if (window_full) {
                    run_temperature_prediction();
                } else {
                    printf("Amostras coletadas: %d/10\n", window_pos);
                }
            }

            last_sample_time = get_absolute_time();
        }

        sleep_ms(100);  // Sleep pequeno para não travar o processador
    }

    return 0;
}
