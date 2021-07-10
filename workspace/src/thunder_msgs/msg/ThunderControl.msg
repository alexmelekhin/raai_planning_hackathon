Header header

uint32 CAN_CONTROL_ID = 528     # 0x210
uint32 CAN_PTZ_ID = 353         # 0x161 

uint8 CONTROL_MODE_NONE = 10
uint8 CONTROL_MODE_FAN = 11
uint8 CONTROL_MODE_AUTO = 12
uint8 CONTROL_MODE_PTZ = 13

uint8 FAN_PUMP_NONE = 20
uint8 FAN_PUMP_CLOSE = 22
uint8 FAN_PUMP_OPEN = 23

uint8 CONTROL_SWITCH_NONE = 30
uint8 CONTROL_SWITCH_CONTROL = 32
uint8 CONTROL_SWITCH_PTZ = 31

uint8 PTZ_SWITCH_NONE = 40
uint8 PTZ_SWITCH_MANUAL = 42
uint8 PTZ_SWITCH_AUTO = 43

# type of can message
uint32 can_id

# control command
uint8 control_mode

# ptz command
uint8 fan_pump
uint8 control_switch
uint8 ptz_switch
int16 ptz
