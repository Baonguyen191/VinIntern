**DMP — telemetry keys theo nhóm thiết bị** (v1.13 · 132 telemetry + 16 attribute)

Đơn vị ghi trong ngoặc; không có ngoặc = bool / enum / counter không thứ nguyên.

**ICMP ping** (3)
reachable, packet_loss_pct (%), rtt_ms (ms)

**Host / gateway compute** (20)
cpu_usage_pct (%), cpu_iowait_pct (%), cpu_load_avg_1m, cpu_load_avg_5m, cpu_load_avg_15m, cpu_temp_c (°C), memory_usage_pct (%), memory_usage_mb (MB), memory_used_mb (MB), memory_free_mb (MB), swap_used_mb (MB), storage_free_gb (GB), storage_used_gb (GB), storage_health, process_count (count), thread_count (count), uptime_seconds (s), reboot_count_total (count), net_rx_bytes_total (bytes), net_tx_bytes_total (bytes)

**Kết nối: cellular / WLAN / gateway** (9)
cellular_signal_pct (%), cellular_csq, cellular_data_day_mb (MB), cellular_data_month_mb (MB), cellular_sim_active, wlan_signal_pct (%), wlan_signal_level_dbm (dBm), battery_low, child_count (count)

**GPS** (6)
gps_latitude (°), gps_longitude (°), gps_altitude_m (m), gps_speed_kmh (km/h), gps_course_deg (°), gps_satellite_count (count)

**Camera / NVR · Hik ISAPI** (8)
camera_run_seconds_total (s), channel_online_count (count), ptz_pan_rounds_total (count), ptz_tilt_rounds_total (count), ptz_zoom_steps_total (count), ptz_focus_steps_total (count), ircut_shifts_total (count), heater_state

**Access control · ZKTeco** (11)
online, door_state, alarm, device_time, user_count (count), user_capacity (count), fingerprint_count (count), fingerprint_capacity (count), face_count (count), record_count (count), record_capacity (count)

**Báo cháy · Notifier FAS** (9)
fire_alarm_state, fire_pre_alarm_state, fire_acked_state, fire_output_active_state, fire_panel_status, fire_event_code, fire_panel_event_time, smoke_state, ambient_temp_c (°C)

**Thang máy · Schindler PORT** (2)
lift_service_online, lift_service_state_code

**Điện năng / power meter** (13)
power_active_kw (kW), power_factor, frequency_hz (Hz), voltage_v (V), voltage_l1_v (V), voltage_l2_v (V), voltage_l3_v (V), current_a (A), current_l1_a (A), current_l2_a (A), current_l3_a (A), energy_active_kwh_total (kWh), energy_reactive_kvarh_total (kvarh)

**HVAC · chiller** (9)
chilled_water_supply_temp_c (°C), chilled_water_return_temp_c (°C), chilled_water_supply_pressure_bar (bar), chilled_water_return_pressure_bar (bar), chiller_load_pct (%), cooling_kw (kW), chiller_state, chiller_state_cmd, runtime_min (min)

**HVAC · air side (PAU / quạt / tháp giải nhiệt)** (16)
supply_air_temp_c (°C), supply_air_pressure_pa (Pa), fan_speed_pct (%), fan_speed_pct_sp (%), fan_state, fan_state_cmd, damper_state, damper_state_cmd, filter_state, water_valve_state, water_valve_state_cmd, cooling_tower_fan_state, diff_pressure_pa (Pa), air_flow_m3h (m³/h), co2_ppm (ppm), co_ppm (ppm)

**Nước / bơm / van / bồn** (20)
pump_speed_pct (%), pump_speed_pct_sp (%), pump_state, pump_state_cmd, pressure_supply_bar (bar), pressure_return_bar (bar), water_supply_temp_c (°C), water_return_temp_c (°C), water_flow_lpm (L/min), water_volume_m3_total (m³), supply_valve_open_limit, supply_valve_close_limit, supply_valve_state_cmd, return_valve_open_limit, return_valve_close_limit, return_valve_state_cmd, bypass_valve_state_cmd, tank_level_low, tank_level_medium, tank_level_high

**Dùng chung nhiều loại thiết bị** (6)
fault, mode, state_on_off, state_on_off_cmd, lighting_knx_state, lighting_knx_state_cmd

**Attribute — metadata, không phải time-series** (16)
cpu_count, device_label, device_model, device_type, device_type_isapi, firmware_release_date, firmware_version, imei, lift_service_state, mac_address, memory_total_mb, platform, serial_number, storage_capacity_gb, swap_total_mb, vendor_device_id

**§2.1 Siemens NORIS** (13)
active_energy_kwh (kWh), active_power_kw (kW), chw_return_temp (°C), chw_supply_temp (°C), compressor_status, cooling_load_kw (kW), evaporator_pressure (bar), op_mode, run_hours (h), voltage_l1 (V), voltage_l2 (V), voltage_l3 (V), water_volume_m3 (m³)

**§2.2 PHATTAI — trạm quan trắc không khí** (12)
aqi, co (µg/m³), humidity (%), no2 (µg/m³), o3 (µg/m³), pm1 (µg/m³), pm10 (µg/m³), pm25 (µg/m³), pm25_avg (µg/m³), so2 (µg/m³), temperature (°C), wind_speed (m/s)

---
 