#include <driver/i2c.h>
#include <esp_log.h>
#include <DS1307.hpp>

// ---------- DEFINE ---------- //
#define DEF_ADDR 0x68


// ---------- Class functions ---------- //
DS1307::DS1307(i2c_port_t port, gpio_num_t sda, gpio_num_t scl, uint16_t to)
{
    this->port = port;
    this->sda = sda;
    this->scl = scl;
    
    if (to > 0) {
        this->to = to;
    }
}

DS1307::~DS1307()
{
    if(i2c_driver_delete(port) != ESP_OK)
    {
        printf("Driver delete failed\n");
    }else
    {
        printf("Driver deleted!\n");
    }
    
}

int DS1307::init()
{
    i2c_config_t config = {};
    config.mode = I2C_MODE_MASTER;
    config.sda_io_num = sda;
    config.scl_io_num = scl;
    config.sda_pullup_en = GPIO_PULLUP_ENABLE;
    config.scl_pullup_en = GPIO_PULLUP_ENABLE;
    config.master.clk_speed = 100000; // Set I2C clock speed to 100kHz
    config.clk_flags = 0;

    if(i2c_param_config(port, &config) != ESP_OK) {
        printf("Configuration failed\n");
        return 1;
    }else
    {
        printf("configs!");
    }
    

    if(i2c_driver_install(port, config.mode, 0, 0, 0) != ESP_OK){
        printf("Drivers couldn't be installed\n");
        return 1;
    }else
    {
        printf("drivers!\n");
    }

    return 0;
}


uint8_t DS1307::get_timeout()
{
    return this->to;
}

uint8_t DS1307::bcd_dec(uint8_t bcd)
{
    return (bcd >> 4) * 10 + (bcd & 0x0F);
}

uint8_t DS1307::dec_bcd(uint8_t dec)
{
    return ((dec /10) << 4) | (dec % 10);
}

int DS1307::read_time()
{
    uint8_t data[3];
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, 0xD0, true); // I2C adresa komunikacije
    i2c_master_write_byte(cmd, 0x00, true); // adresa na RTC-u
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, 0xD1, true);
    i2c_master_read(cmd, data, 3, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);
    esp_err_t res = i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(1000));

    if (res != ESP_OK){
        return 1;
    }

    printf("%02d h : %02d min : %02d s\n\n", bcd_dec(data[2]), bcd_dec(data[1]), bcd_dec(data[0] & 0x7f));
return 0;
}

int DS1307::set_time(uint8_t h, uint8_t min, uint8_t s)
{
    uint8_t time[3] = {
        dec_bcd(s),
        dec_bcd(min),
        dec_bcd(h)  
    };

    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, 0xD0, true); // I2C adresa komunikacije
    i2c_master_write_byte(cmd, 0x00, true); // adresa na RTC-u
    i2c_master_write(cmd, time, sizeof(time), true);
    i2c_master_stop(cmd);

    if(i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(1000)) != ESP_OK){
        return 1;
    }

    return 0;
}

int DS1307::read_reg(uint8_t reg)
{
    uint8_t data;
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, 0xD0, true); // I2C adresa komunikacije
    i2c_master_write_byte(cmd, reg, true); // adresa na RTC-u
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, 0xD1, true);
    i2c_master_read(cmd, &data, 1, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);

    if(i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(1000)) != ESP_OK){
        printf("Nisam uspio pročitati, provjerite adresu i podatke!");
        return 1;
    }
    printf("Addr  :  Data\n");
    printf("0x%02X  : 0x%02X\n\n", reg, data);

    return 0;
}

int DS1307::write_reg(uint8_t reg, uint8_t data)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, 0xD0, true); // I2C adresa komunikacije
    i2c_master_write_byte(cmd, reg, true); // adresa na RTC-u
    i2c_master_write(cmd, &data, sizeof(data), true);
    i2c_master_stop(cmd);

    if(i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(1000)) != ESP_OK){
        return 1;
    }

    return 0;
}