// Wokwi Custom Chip - For information and examples see:
// https://docs.wokwi.com/chips-api/getting-started
//
// Sveprisutno 2023 - Aritmetičko logička jedinica




#include "wokwi-api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


const int ADDRESS = 0x20;

#define OPERATION_NOP 0
#define OPERATION_ADD 1
#define OPERATION_SUB 2
#define OPERATION_READ_A 100
#define OPERATION_READ_B 101

union floatunion_t {
    float f;
    unsigned char a[sizeof (float) ];
} float_u;

typedef struct {
  uint8_t start_operation;
  uint8_t loc_adr;
  uint8_t loc_count;
  union floatunion_t par1;
  union floatunion_t par2;
  uint8_t operation;
  uint32_t threshold_attr;
} chip_state_t;

static bool on_i2c_connect(void *chip, uint32_t address, bool connect);
static uint8_t on_i2c_read(void *chip);
static bool on_i2c_write(void *chip, uint8_t data);
static void on_i2c_disconnect(void *chip);

#define MAX_INTERNAL_REG 3

// Generates random
int retRND3() 
{ 
   //rand() % (max_number + 1 - minimum_number) + minimum_number
   return 1;
   int num = rand() % 5;
   printf("%d\n", num);
    return num;
} 

void chip_init() {
  chip_state_t *chip = malloc(sizeof(chip_state_t));

  chip->par1.f = 0;
  chip->par2.f = 0;

  chip->loc_adr = 0;
  chip->loc_count = 0;
  chip->operation = 0;
  chip->start_operation = 0;

  const i2c_config_t i2c_config = {
    .user_data = chip,
    .address = ADDRESS,
    .scl = pin_init("SCL", INPUT),
    .sda = pin_init("SDA", INPUT),
    .connect = on_i2c_connect,
    .read = on_i2c_read,
    .write = on_i2c_write,
    .disconnect = on_i2c_disconnect, // Optional
  };
  i2c_init(&i2c_config);

  // This attribute can be edited by the user. It's defined in wokwi-custom-part.json:
  chip->threshold_attr = attr_init("threshold", 127);

  // The following message will appear in the browser's DevTools console:
  printf("Hello from I2C_1 chip!\n");
}

static void counter_updated(chip_state_t *chip) {
  /*const uint32_t threshold = attr_read(chip->threshold_attr);
  if (chip->counter > threshold) {
    pin_write(chip->pin_int, LOW);
    pin_mode(chip->pin_int, OUTPUT);
  } else {
    pin_mode(chip->pin_int, INPUT);
  }*/
}

bool on_i2c_connect(void *user_ctx, uint32_t address, bool read) {
  chip_state_t *chip = user_ctx;
  //if (read){
  //  if (retRND3() == 0){
  //    printf("Device not ready. Please repeat commad\n");
  //    return false; /* NotAck */
  //  }
  //}
  printf("I2C_1 Process connected\n");
  chip->start_operation = 1;
  return true; /* Ack */
  
}

uint8_t on_i2c_read(void *user_ctx) {
  union floatunion_t rez;
  chip_state_t *chip = user_ctx;
  
  if (chip->start_operation == 1){
      //lokalna adresa      
      chip->loc_count = 0;
      chip->start_operation = 0;
      //printf("Reading from MUL [%d]\n",chip->loc_adr);
  }else{
     chip->loc_count = (chip->loc_count + 1) % 4; 
  }
  
  if (chip->operation == OPERATION_ADD){
      rez.f = chip->par1.f + chip->par2.f;
  }else if (chip->operation == OPERATION_SUB){
      rez.f = chip->par1.f - chip->par2.f;
  }else{
    rez.f = -0.1010101;
  }


  printf("Sending rez [%f] [%d][%d]\n", rez.f, chip->loc_count, rez.a[chip->loc_count]);
  return (uint8_t)rez.a[chip->loc_count];

  //counter_updated(chip);
  //return chip->counter;
}

bool on_i2c_write(void *user_ctx, uint8_t data) {
  
  chip_state_t *chip = user_ctx;
 
  if (chip->start_operation == 1){
      //lokalna adresa      
      chip->loc_adr = data;
      chip->loc_count = 0;
      if (chip->loc_adr > MAX_INTERNAL_REG) chip->loc_adr =chip->loc_adr % MAX_INTERNAL_REG;
      chip->start_operation = 0;
      //printf("Write to local addres [%d]\n",chip->loc_adr);
  }else{
    //printf("Write to address [%d][%d] = %d\n",chip->loc_adr,chip->loc_count, data);
    if(chip->loc_adr == 0 ){ //operacija
        chip->operation = data ;       
        printf("Operation := %d\n",chip->operation);
    
    }else if(chip->loc_adr == 1 ){ //parametar 1
        //printf("Write to address [%d][%d] = %d\n",chip->loc_adr,chip->loc_count, data);
        chip->par1.a[chip->loc_count] = data;
        chip->loc_count = (chip->loc_count+1) % sizeof (float);
        printf("PAR1 := %f\n", chip->par1.f);
        //printf("PAR1 := [%d],[%d],[%d],[%d]\n", chip->par1.a[3], chip->par1.a[2],chip->par1.a[1],chip->par1.a[0]);
        //printf("PAR1 := %d\n", chip->loc_count);
        //printf("PAR1 := %lu\n",sizeof(float));
    
    }else if(chip->loc_adr == 2 ){ //parametar 2
        chip->par2.a[chip->loc_count] = data;        
        chip->loc_count = (chip->loc_count+1) % sizeof (float);
        printf("PAR2 := %f\n",chip->par2.f);
    }else{
      printf("Not allower addres");
    }
    
  }
  //counter_updated(chip);
  return true; // Ack
}

void on_i2c_disconnect(void *user_ctx) {
  // Do nothing
  printf("I2C_1 Process disconected!!!\n");
}
