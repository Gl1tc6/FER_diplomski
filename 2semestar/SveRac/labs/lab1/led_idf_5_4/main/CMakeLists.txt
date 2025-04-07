idf_component_register(SRCS "app_main.cpp" "CLed.cpp" "CButton.cpp"
                    PRIV_REQUIRES spi_flash
                    REQUIRES driver esp_timer
                    INCLUDE_DIRS ""
                    )