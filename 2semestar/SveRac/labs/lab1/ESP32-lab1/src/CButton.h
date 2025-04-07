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
        ButtonEventHandler singleClick = nullptr;
        ButtonEventHandler doubleClick = nullptr;
        ButtonEventHandler longPress = nullptr;

        gpio_num_t m_pinNumber;
        int m_lastState = 1;              // prošlo stanje
        int64_t m_lastStateChangeTime = 0; // t od zadnje promjene stanja
        int64_t m_lastPressTime = 0;      // t od zadnjeg klika
        int64_t m_lastReleaseTime = 0;    // t od zadnjeg otpuštanja
        int m_clickCount = 0;
        bool m_isPressed = false;
        bool m_isLongPressDetected = false;
        bool m_processingClicks = false;   // blokirajuća zastava
};


#endif