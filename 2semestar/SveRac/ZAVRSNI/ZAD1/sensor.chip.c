//
// SPDX-License-Identifier: MIT
// Copyright (C) 2022 Uri Shaked / wokwi.com

#include "wokwi-api.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  uart_dev_t urt0;
  uint8_t x;
  timer_t timer;
  uint32_t attr_offset;
} chip_state_t;

static void on_rx_data(void *user_data, uint8_t byte);
static void on_write_done(void *user_data);

static void chip_timer_callback(void *user_data) {
  /* Called when the timer fires */
  chip_state_t *chip = (chip_state_t*)user_data;
  uint8_t x = chip->x;
  uint8_t offset = attr_read(chip->attr_offset);
  x = x + offset;
  printf("Sending data: %d\n", x);
  uart_write(chip->urt0, &x, 1);
}

void chip_init(void) {
  chip_state_t *chip = malloc(sizeof(chip_state_t));

  const uart_config_t u_config = {
    .tx = pin_init("TX", INPUT_PULLUP),
    .rx = pin_init("RX", INPUT),
    .baud_rate = 1000,
    .rx_data = on_rx_data,
    .write_done = on_write_done,
    .user_data = chip,
  };
  chip->urt0 = uart_init(&u_config);


  printf("SENSOR Chip initialized!\n");
  //Init Atribute
  chip->attr_offset = attr_init("offset", 5);

  //Init timer
  const timer_config_t timer_cfg= {
    .callback = chip_timer_callback,
    .user_data = chip
  };

  chip->timer = timer_init(&timer_cfg);
}

static void on_rx_data(void *user_data, uint8_t byte) {
  chip_state_t *chip = (chip_state_t*)user_data;
  printf("Incoming data: %d\n", byte);
  chip->x = byte;
  timer_start( chip->timer, 1000000, false);
}

static void on_write_done(void *user_data) {
  chip_state_t *chip = (chip_state_t*)user_data;
  printf("Send data successfully\n");
} 
