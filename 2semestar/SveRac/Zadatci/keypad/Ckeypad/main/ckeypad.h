#ifndef CKEYPAD_H
#define CKEYPAD_H

#include <stdint.h>
#include <stdbool.h>

// Pin definitions
#define C1 32
#define C2 27
#define C3 26
#define R1 12
#define R2 14
#define R3 25
#define R4 33

// Keypad dimensions
#define KEYPAD_ROWS 4
#define KEYPAD_COLS 3

// Press types
typedef enum {
    PRESS_SINGLE = 0,
    PRESS_DOUBLE,
    PRESS_LONG
} press_type_t;

// Callback function type
typedef void (*keypad_callback_t)(char key, press_type_t press_type);

// Keypad state structure
typedef struct {
    // GPIO pins
    int row_pins[KEYPAD_ROWS];
    int col_pins[KEYPAD_COLS];
    
    // Keypad layout
    char key_map[KEYPAD_ROWS][KEYPAD_COLS];
    
    // Callback functions
    keypad_callback_t single_press_callback;
    keypad_callback_t double_press_callback;
    keypad_callback_t long_press_callback;
    
    // State tracking
    char last_key;
    char previous_key;  // For double press detection
    uint64_t last_press_time;
    uint64_t key_press_start_time;
    bool key_pressed;
    bool waiting_for_double;
    bool long_press_triggered;
    
    // Timing constants
    uint64_t long_press_threshold;  // 500ms
    uint64_t double_press_window;   // 300ms
    uint64_t scan_interval;         // 10ms
    
    // Task handle
    void* task_handle;
    bool running;
} CKeyPad;

// Function prototypes
void ckeypad_init(CKeyPad* keypad);
void ckeypad_set_single_press_callback(CKeyPad* keypad, keypad_callback_t callback);
void ckeypad_set_double_press_callback(CKeyPad* keypad, keypad_callback_t callback);
void ckeypad_set_long_press_callback(CKeyPad* keypad, keypad_callback_t callback);
void ckeypad_start(CKeyPad* keypad);
void ckeypad_stop(CKeyPad* keypad);
char ckeypad_scan(CKeyPad* keypad);

#endif // CKEYPAD_H