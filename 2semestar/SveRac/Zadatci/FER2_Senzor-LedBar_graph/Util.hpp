#pragma once

#include "esp_err.h"

#define self (*(this))

#define TIME_US_TO_ESP(...) (__VA_ARGS__)
#define TIME_MS_TO_ESP(...) TIME_US_TO_ESP((__VA_ARGS__) * 1000)
#define TIME_S_TO_ESP( ...) TIME_MS_TO_ESP((__VA_ARGS__) * 1000)

template <bool x>
class _MyEspErrorCheck {
	bool state = true;
public:
	operator bool() {
		bool state = self.state;
		self.state = false;
		return state;
	}
	void step() {
		if (unlikely(self.error != ESP_OK)) {
			if (x) {
				abort();
			} else {
				_esp_error_check_failed(self.error, __FILE__, __LINE__, "?", "?");
			}
		}
	}
	esp_err_t error;
};
#ifdef NDEBUG
	#define esp_check 
#elif defined(CONFIG_COMPILER_OPTIMIZATION_ASSERTIONS_SILENT)
	#define esp_check for (_MyEspErrorCheck<true> _my_meech; _my_meech; _my_meech.step()) _my_meech.error = 
#else
	#define esp_check for (_MyEspErrorCheck<false> _my_meech; _my_meech; _my_meech.step()) _my_meech.error = 
#endif
