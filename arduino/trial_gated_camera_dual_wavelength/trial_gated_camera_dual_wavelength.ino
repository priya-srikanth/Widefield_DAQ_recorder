// Trial-gated widefield Teensy firmware.
//
// Behavior trial_start from behavior Arduino pin 6 into Teensy pin 20
// enables camera-trigger pulses.
// Behavior trial_stop from behavior Arduino pin 9 into Teensy pin 22
// disables camera-trigger pulses and LED gates.
//
// Wiring expectation:
//   pin 20: behavior trial_start TTL input, from behavior Arduino pin 6
//   pin 22: behavior trial_stop TTL input, from behavior Arduino pin 9
//   pin 18: Teensy camera trigger TTL output -> PCO SMA input #1
//   pin  3: PCO SMA output #4 Status Expos input -> gates LEDs during exposure
//   pin  5: 415 nm/violet LED TTL output
//   pin  6: 470 nm/blue LED TTL output
//   pin  7: optional frame/trigger mirror to DAQ
//   pin  4: optional global sync TTL input, logged as @T/#SYNC
//
// PCO must be configured for external trigger/frame-start mode for pin 18 to
// control exposure. PCO SMA output #4 should be configured as Status Expos,
// "Show common time of All lines", high polarity. LEDs turn on only while that
// status line is high.
//
// labcams CamStimInterface-compatible protocol:
//   @Q -> @Q_NCHANNELS_2_MODES_415nm:470nm:both
//   @C -> @C_1 or @C_2
//   @M_<mode>, @N, @S

const byte PIN_TRIAL_START = 20;
const byte PIN_TRIAL_STOP  = 22;
const byte PIN_PCO_STATUS_EXPOS = 3;
const byte PIN_GLOBAL_SYNC = 4;

const byte PIN_PCO_TRIGGER = 18;
const byte PIN_LED0_TRIGGER = 5;  // 415 nm/violet
const byte PIN_LED1_TRIGGER = 6;  // 470 nm/blue
const byte PIN_GPIO = 7;          // optional trigger mirror to DAQ

#define GPIO_MIMIC_TRIGGER

// Timing. Keep camera trigger pulse short. LEDs are gated by PCO Status Expos.
const uint32_t FRAME_PERIOD_US = 16000;   // 62.5 Hz raw camera trigger rate
const uint32_t CAMERA_TRIGGER_US = 1000;  // frame-start trigger pulse width

#define STX '@'
#define ETX '\n'
#define SEP "_"
#define CAP "NCHANNELS_2_MODES_415nm:470nm:both"

#define QUERY_NCHANNELS 'C'
#define QUERY_CAP       'Q'
#define START_LEDS      'N'
#define STOP_LEDS       'S'
#define FRAME           'F'
#define SYNC            'T'
#define SET_MODE        'M'
#define TRIAL_EVENT     'R'

volatile uint8_t mode = 3;       // 1: 415, 2: 470, 3: alternate
volatile uint8_t armed = 0;      // labcams says stimulation/camera control allowed
volatile uint8_t trial_active = 0;

volatile uint32_t pulse_count = 0;
volatile uint32_t last_pulse_count = 0;
volatile int32_t last_frame_ms = -1;
volatile byte last_led_pin = 0;
volatile byte pending_led_pin = 0;

volatile uint32_t sync_count = 0;
volatile uint32_t sync_frame_count = 0;
volatile int32_t last_sync_ms = -1;

volatile uint8_t last_trial_code = 0;  // 1=start, 2=stop
volatile uint32_t last_trial_frame = 0;
volatile int32_t last_trial_ms = -1;

uint32_t start_time_ms = 0;
uint32_t next_frame_us = 0;
uint32_t camera_low_us = 0;
byte active_led_pin = 0;
uint8_t camera_pulse_high = 0;
volatile uint8_t led_gate_high = 0;

#define MSGSIZE 64
char msg[MSGSIZE];
int cnt = 0;

int32_t elapsed_ms() {
  return (int32_t)(millis() - start_time_ms);
}

void all_outputs_low() {
  digitalWriteFast(PIN_PCO_TRIGGER, LOW);
  digitalWriteFast(PIN_LED0_TRIGGER, LOW);
  digitalWriteFast(PIN_LED1_TRIGGER, LOW);
#ifdef GPIO_MIMIC_TRIGGER
  digitalWriteFast(PIN_GPIO, LOW);
#endif
  camera_pulse_high = 0;
  led_gate_high = 0;
  active_led_pin = 0;
  pending_led_pin = 0;
}

void trial_start_received() {
  if (digitalReadFast(PIN_TRIAL_START) == HIGH) {
    trial_active = 1;
    next_frame_us = micros();
    last_trial_code = 1;
    last_trial_frame = pulse_count;
    last_trial_ms = elapsed_ms();
  }
}

void trial_stop_received() {
  if (digitalReadFast(PIN_TRIAL_STOP) == HIGH) {
    trial_active = 0;
    all_outputs_low();
    last_trial_code = 2;
    last_trial_frame = pulse_count;
    last_trial_ms = elapsed_ms();
  }
}

void sync_received() {
  if (digitalReadFast(PIN_GLOBAL_SYNC) == HIGH) {
    sync_count++;
    sync_frame_count = pulse_count;
    last_sync_ms = elapsed_ms();
  }
}

void exposure_status_changed() {
  if (digitalReadFast(PIN_PCO_STATUS_EXPOS) == HIGH) {
    led_gate_high = 1;
    if (armed && trial_active && pending_led_pin != 0) {
      active_led_pin = pending_led_pin;
      digitalWriteFast(PIN_LED0_TRIGGER, LOW);
      digitalWriteFast(PIN_LED1_TRIGGER, LOW);
      digitalWriteFast(active_led_pin, HIGH);
    }
  } else {
    led_gate_high = 0;
    digitalWriteFast(PIN_LED0_TRIGGER, LOW);
    digitalWriteFast(PIN_LED1_TRIGGER, LOW);
    active_led_pin = 0;
  }
}

byte led_for_next_frame() {
  switch (mode) {
    case 1:
      return PIN_LED0_TRIGGER;
    case 2:
      return PIN_LED1_TRIGGER;
    case 3:
      return ((pulse_count % 2) == 0) ? PIN_LED1_TRIGGER : PIN_LED0_TRIGGER;
    default:
      return PIN_LED0_TRIGGER;
  }
}

void trigger_frame_if_due() {
  uint32_t now = micros();
  if (armed && trial_active && (int32_t)(now - next_frame_us) >= 0) {
    pulse_count++;
    digitalWriteFast(PIN_LED0_TRIGGER, LOW);
    digitalWriteFast(PIN_LED1_TRIGGER, LOW);
    pending_led_pin = led_for_next_frame();

    digitalWriteFast(PIN_PCO_TRIGGER, HIGH);
#ifdef GPIO_MIMIC_TRIGGER
    digitalWriteFast(PIN_GPIO, HIGH);
#endif
    camera_pulse_high = 1;
    camera_low_us = now + CAMERA_TRIGGER_US;
    next_frame_us += FRAME_PERIOD_US;

    last_led_pin = pending_led_pin;
    last_pulse_count = pulse_count;
    last_frame_ms = elapsed_ms();
  }

  now = micros();
  if (camera_pulse_high && (int32_t)(now - camera_low_us) >= 0) {
    digitalWriteFast(PIN_PCO_TRIGGER, LOW);
    camera_pulse_high = 0;
  }
}

void setup() {
  pinMode(PIN_TRIAL_START, INPUT);
  pinMode(PIN_TRIAL_STOP, INPUT);
  pinMode(PIN_PCO_STATUS_EXPOS, INPUT);
  pinMode(PIN_GLOBAL_SYNC, INPUT);
  pinMode(PIN_PCO_TRIGGER, OUTPUT);
  pinMode(PIN_LED0_TRIGGER, OUTPUT);
  pinMode(PIN_LED1_TRIGGER, OUTPUT);
  pinMode(PIN_GPIO, OUTPUT);

  all_outputs_low();
  Serial.begin(2000000);

  attachInterrupt(digitalPinToInterrupt(PIN_TRIAL_START), trial_start_received, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_TRIAL_STOP),  trial_stop_received,  RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_PCO_STATUS_EXPOS), exposure_status_changed, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_SYNC), sync_received,        RISING);

  start_time_ms = millis();
  next_frame_us = micros();
}

void report_events() {
  noInterrupts();
  int32_t frame_ms = last_frame_ms;
  uint32_t frame_count = last_pulse_count;
  byte led = last_led_pin;
  int32_t sync_ms = last_sync_ms;
  uint32_t sc = sync_count;
  uint32_t sfc = sync_frame_count;
  int32_t trial_ms = last_trial_ms;
  uint8_t trial_code = last_trial_code;
  uint32_t trial_frame = last_trial_frame;
  interrupts();

  if (frame_ms > 0) {
    Serial.print(STX); Serial.print(FRAME); Serial.print(SEP);
    Serial.print((int)led); Serial.print(SEP);
    Serial.print(frame_count); Serial.print(SEP);
    Serial.print(frame_ms); Serial.print(ETX);
    noInterrupts(); last_frame_ms = -1; interrupts();
  }

  if (sync_ms > 0) {
    Serial.print(STX); Serial.print(SYNC); Serial.print(SEP);
    Serial.print(sfc); Serial.print(SEP);
    Serial.print(sc); Serial.print(SEP);
    Serial.print(sync_ms); Serial.print(ETX);
    noInterrupts(); last_sync_ms = -1; interrupts();
  }

  if (trial_ms > 0) {
    Serial.print(STX); Serial.print(TRIAL_EVENT); Serial.print(SEP);
    Serial.print((int)trial_code); Serial.print(SEP);
    Serial.print(trial_frame); Serial.print(SEP);
    Serial.print(trial_ms); Serial.print(ETX);
    noInterrupts(); last_trial_ms = -1; interrupts();
  }
}

void loop() {
  trigger_frame_if_due();
  report_events();
  serialEvent();
}

void serialEvent() {
  while (Serial.available()) {
    char ch = Serial.read();
    char* token;

    if (ch == STX || cnt > 0) {
      msg[cnt] = ch;
      cnt++;

      if (ch == ETX || cnt >= MSGSIZE) {
        cnt = 0;
        String reply = String(STX);

        switch (msg[1]) {
          case START_LEDS:
            start_time_ms = millis();
            pulse_count = 0;
            last_pulse_count = 0;
            sync_frame_count = 0;
            sync_count = 0;
            last_sync_ms = -1;
            last_frame_ms = -1;
            last_trial_ms = -1;
            trial_active = 0;  // wait for behavior trial_start pin
            next_frame_us = micros();
            armed = 1;
            reply += START_LEDS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(elapsed_ms()); Serial.print(ETX);
            break;

          case STOP_LEDS:
            armed = 0;
            trial_active = 0;
            all_outputs_low();
            reply += STOP_LEDS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(elapsed_ms()); Serial.print(ETX);
            break;

          case QUERY_CAP:
            reply += QUERY_CAP;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(CAP); Serial.print(ETX);
            break;

          case SET_MODE:
            token = strtok(msg, SEP);
            token = strtok(NULL, SEP);
            mode = token ? atoi(token) : 3;
            if (mode < 1 || mode > 3) {
              mode = 3;
            }
            reply += SET_MODE;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(mode); Serial.print(ETX);
            break;

          case QUERY_NCHANNELS:
            reply += QUERY_NCHANNELS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print((mode < 3) ? 1 : 2);
            Serial.print(ETX);
            break;

          default:
            reply += "E";
            reply += 1;
            reply += ETX;
            Serial.print(reply);
            break;
        }
      }
    }
  }
  Serial.flush();
}
