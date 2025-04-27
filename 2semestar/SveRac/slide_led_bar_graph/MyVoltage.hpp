#pragma once

#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

#include "./Util.hpp"

struct MyAdc {
	adc_unit_t id;
	adc_oneshot_unit_handle_t handle;
	MyAdc(adc_unit_t id): id(id) {
		adc_oneshot_unit_init_cfg_t unit_cfg{ id, (adc_oneshot_clk_src_t) 0, ADC_ULP_MODE_DISABLE };
		esp_check adc_oneshot_new_unit(&unit_cfg, &self.handle);
	}
	~MyAdc() {
		adc_oneshot_del_unit(self.handle);
	}
};
struct MyChannel {
	MyAdc& adc;
	adc_channel_t id;
	adc_atten_t atten;
	adc_bitwidth_t bitwidth;
	MyChannel(MyAdc& adc, adc_channel_t id, adc_atten_t atten, adc_bitwidth_t bitwidth) : adc(adc), id(id), atten(atten), bitwidth(bitwidth) {
		adc_oneshot_chan_cfg_t channel_config{ atten, bitwidth, };
		esp_check adc_oneshot_config_channel(adc.handle, id, &channel_config);
	}
};
struct MyVoltage {
	adc_cali_handle_t handle;
	MyChannel& channel;
	MyVoltage(MyChannel& channel) : channel(channel) {
		adc_cali_line_fitting_config_t cali_cfg{ channel.adc.id, channel.atten, channel.bitwidth, 5000 };
		esp_check adc_cali_create_scheme_line_fitting(&cali_cfg, &self.handle);
	}
	int get_mV() {
		int output = 888888;
		esp_check adc_oneshot_read(self.channel.adc.handle, self.channel.id, &output);
		int voltage_mV;
		esp_check adc_cali_raw_to_voltage(self.handle, output, &voltage_mV);
		return voltage_mV;
	}
};
template <int pullup_Ohm, int source_mV>
int getResistance_Ohm(int voltage_mV) {
	return pullup_Ohm * voltage_mV / (source_mV - voltage_mV);
}
