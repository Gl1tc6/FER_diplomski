// CButton.h

#ifndef _CButton_h
#define _CButton_h

 
// Pointer to event handling methods
extern "C" {
    typedef void (*ButtonEventHandler)(void);
}
// void my_singeClick_function(){}

class CButton{
    public:
        CButton(int port);
        void attachSingleClick(ButtonEventHandler method){singleClick = method;};
        void attachDoubleClick(ButtonEventHandler method){doubleClick = method;};
        void attachLongPress(ButtonEventHandler method){longPress = method;};

        void tick();

    private:
        ButtonEventHandler singleClick = NULL;
        ButtonEventHandler doubleClick = NULL;
        ButtonEventHandler longPress = NULL;

        gpio_num_t m_pinNumber;
        int m_lastState = 1;              // Last recorded button state
        int64_t m_lastStateChangeTime = 0; // Time of last state change
        int64_t m_lastPressTime = 0;      // Time of last button press
        int64_t m_lastReleaseTime = 0;    // Time of last button release
        int m_clickCount = 0;             // Counter for click detection
        bool m_isPressed = false;         // Current press state
        bool m_isLongPressDetected = false; // Long press flag
        bool m_processingClicks = false;   // Flag to track click processing
        
        const char *LogName = "CButton";   // For logging
};


#endif