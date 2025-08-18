#include <stdio.h>
#include <stdint.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/uart.h>
#include <esp_log.h>

#define UART_PORT UART_NUM_2
#define TX_PIN 25
#define RX_PIN 26

void init(void) {
    uart_config_t uart_config = {
        .baud_rate = 1000,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    
    uart_param_config(UART_PORT, &uart_config);
    uart_set_pin(UART_PORT, TX_PIN, RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_PORT, 256, 256, 0, NULL, 0);
    
    ESP_LOGI("SENSORS", "UART init - TX: %d, RX: %d", TX_PIN, RX_PIN);
}

void sensor_send(uint8_t data) {
    uart_write_bytes(UART_PORT, &data, 1);
    uart_wait_tx_done(UART_PORT, 100 / portTICK_PERIOD_MS);
    ESP_LOGI("SENSORS", "Poslano: %d", data);
}

uint8_t sensor_receive(void) {
    ESP_LOGI("SENSORS", "Cekam podatke...");
    uint8_t data = 0;
    
    int len = uart_read_bytes(UART_PORT, &data, 1, 2000 / portTICK_PERIOD_MS);
    
    if (len > 0) {
        ESP_LOGI("SENSORS", "Primljen podatak: %d", data);
        return data;
    } else {
        return 0;
    }
}
