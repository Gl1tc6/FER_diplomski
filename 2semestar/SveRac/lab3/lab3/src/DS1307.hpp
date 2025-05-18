#include<driver/i2c.h>

class DS1307
{
private:
    i2c_port_t port;
    gpio_num_t sda;
    gpio_num_t scl;
    uint16_t to = 30;

    uint8_t bcd_dec(uint8_t bcd_val);
    uint8_t dec_bcd(uint8_t dec_val);

    
public:
    DS1307(i2c_port_t port, gpio_num_t sda, gpio_num_t scl, uint16_t to);
    ~DS1307();
    int init();
    int read_time();
    int set_time(uint8_t h, uint8_t min, uint8_t s);
    int read_reg(uint8_t reg);
    int write_reg(uint8_t reg, uint8_t data);
    uint8_t get_timeout(); 
};
