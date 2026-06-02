// Trial-gated widefield Teensy firmware.
//
// Behavior trial_start from behavior Arduino pin 6 into Teensy pin 20
// enables camera acquisition.
// Behavior trial_stop from behavior Arduino pin 9 into Teensy pin 19
// disables camera acquisition and LED gates.
//
// Wiring expectation:
//   pin 20: behavior trial_start TTL input, from behavior Arduino pin 6
//   pin 19: behavior trial_stop TTL input, from behavior Arduino pin 9
//   pin 18: Teensy acquire-enable TTL output -> PCO SMA input #2
//   pin  3: PCO SMA output #4 Status Expos input -> gates LEDs during exposure
//   pin  5: 415 nm/violet LED TTL output
//   pin  6: 470 nm/blue LED TTL output
//   pin  7: optional acquire-enable mirror to DAQ
//   pin  4: optional global sync TTL input, logged as @T/#SYNC
//
// PCO should be configured with Acquire Enable enabled/high on SMA input #2.
// PCO SMA output #4 should be configured as Status Expos, "Show common time
// of All lines", high polarity. LEDs turn on only while that status line is high.
//
// labcams CamStimInterface-compatible protocol:
//   @Q -> @Q_NCHANNELS_2_MODES_415nm:470nm:both
//   @C -> @C_1 or @C_2
//   @M_<mode>, @G_<0_or_1>, @D_<max_trial_ms>, @N, @S

const byte PIN_TRIAL_START = 20;
const byte PIN_TRIAL_STOP = 19;
const byte PIN_PCO_STATUS_EXPOS = 3;
const byte PIN_GLOBAL_SYNC = 4;

const byte PIN_PCO_ACQUIRE_ENABLE = 18;
const byte PIN_LED0_TRIGGER = 5;  // 415 nm/violet
const byte PIN_LED1_TRIGGER = 6;  // 470 nm/blue
const byte PIN_GPIO = 7;          // optional acquire-enable mirror to DAQ

#define GPIO_MIMIC_ACQUIRE_ENABLE

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
#define SET_TRIAL_GATE  'G'
#define SET_MAX_TRIAL_MS 'D'
#define TRIAL_EVENT     'R'

volatile uint8_t mode = 3;       // 1: 415, 2: 470, 3: alternate
volatile uint8_t armed = 0;      // labcams says camera/LED control allowed
volatile uint8_t trial_active = 0;
volatile uint8_t trial_gated = 0; // 0: preview/free-run while armed, 1: require trial_start/stop
volatile uint8_t last_trial_start_state = LOW;
volatile uint8_t last_trial_stop_state = LOW;
volatile uint32_t trial_start_ms = 0;
volatile uint32_t max_trial_ms = 5000; // safety stop if trial_stop TTL is missed; 0 disables

volatile uint32_t frame_count = 0;
volatile uint32_t last_frame_count = 0;
volatile int32_t last_frame_ms = -1;
volatile byte last_led_pin = 0;
volatile byte pending_led_pin = 0;

volatile uint32_t sync_count = 0;
volatile uint32_t sync_frame_count = 0;
volatile int32_t last_sync_ms = -1;

volatile uint8_t last_trial_code = 0;  // 1=start, 2=stop, 3=safety timeout
volatile uint32_t last_trial_frame = 0;
volatile int32_t last_trial_ms = -1;

uint32_t start_time_ms = 0;
byte active_led_pin = 0;
volatile uint8_t led_gate_high = 0;

#define MSGSIZE 64
char msg[MSGSIZE];
int cnt = 0;

int32_t elapsed_ms() {
  return (int32_t)(millis() - start_time_ms);
}

uint8_t acquisition_should_be_enabled() {
  return armed && (!trial_gated || trial_active);
}

byte led_for_next_frame() {
  switch (mode) {
    case 1:
      return PIN_LED0_TRIGGER;
    case 2:
      return PIN_LED1_TRIGGER;
    case 3:
      return ((frame_count % 2) == 0) ? PIN_LED1_TRIGGER : PIN_LED0_TRIGGER;
    default:
      return PIN_LED0_TRIGGER;
  }
}

void set_acquire_enabled(uint8_t enabled) {
  digitalWriteFast(PIN_PCO_ACQUIRE_ENABLE, enabled ? HIGH : LOW);
#ifdef GPIO_MIMIC_ACQUIRE_ENABLE
  digitalWriteFast(PIN_GPIO, enabled ? HIGH : LOW);
#endif
}

void all_outputs_low() {
  set_acquire_enabled(0);
  digitalWriteFast(PIN_LED0_TRIGGER, LOW);
  digitalWriteFast(PIN_LED1_TRIGGER, LOW);
  led_gate_high = 0;
  active_led_pin = 0;
  pending_led_pin = 0;
}

void note_frame_start_from_exposure() {
  frame_count++;
  pending_led_pin = led_for_next_frame();
  last_led_pin = pending_led_pin;
  last_frame_count = frame_count;
  last_frame_ms = elapsed_ms();
}

void trial_start_received() {
  if (digitalReadFast(PIN_TRIAL_START) == HIGH) {
    last_trial_start_state = HIGH;
    trial_active = 1;
    trial_start_ms = millis();
    if (armed && trial_gated) {
      set_acquire_enabled(1);
    }
    last_trial_code = 1;
    last_trial_frame = frame_count;
    last_trial_ms = elapsed_ms();
  }
}

void trial_stop_received() {
  if (digitalReadFast(PIN_TRIAL_STOP) == HIGH) {
    last_trial_stop_state = HIGH;
    trial_active = 0;
    all_outputs_low();
    last_trial_code = 2;
    last_trial_frame = frame_count;
    last_trial_ms = elapsed_ms();
  }
}

void sync_received() {
  if (digitalReadFast(PIN_GLOBAL_SYNC) == HIGH) {
    sync_count++;
    sync_frame_count = frame_count;
    last_sync_ms = elapsed_ms();
  }
}

void exposure_status_changed() {
  if (digitalReadFast(PIN_PCO_STATUS_EXPOS) == HIGH) {
    led_gate_high = 1;
    if (acquisition_should_be_enabled()) {
      note_frame_start_from_exposure();
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

void poll_trial_inputs() {
  uint8_t start_state = digitalReadFast(PIN_TRIAL_START);
  uint8_t stop_state = digitalReadFast(PIN_TRIAL_STOP);

  if (start_state == HIGH && last_trial_start_state == LOW) {
    trial_start_received();
  }
  if (stop_state == HIGH && last_trial_stop_state == LOW) {
    trial_stop_received();
  }

  last_trial_start_state = start_state;
  last_trial_stop_state = stop_state;

  if (trial_gated && trial_active && max_trial_ms > 0) {
    uint32_t elapsed = millis() - trial_start_ms;
    if (elapsed >= max_trial_ms) {
      trial_active = 0;
      all_outputs_low();
      last_trial_code = 3;
      last_trial_frame = frame_count;
      last_trial_ms = elapsed_ms();
    }
  }
}

void update_acquire_enable() {
  set_acquire_enabled(acquisition_should_be_enabled() ? 1 : 0);
}

void setup() {
  pinMode(PIN_TRIAL_START, INPUT);
  pinMode(PIN_TRIAL_STOP, INPUT);
  pinMode(PIN_PCO_STATUS_EXPOS, INPUT);
  pinMode(PIN_GLOBAL_SYNC, INPUT);
  pinMode(PIN_PCO_ACQUIRE_ENABLE, OUTPUT);
  pinMode(PIN_LED0_TRIGGER, OUTPUT);
  pinMode(PIN_LED1_TRIGGER, OUTPUT);
  pinMode(PIN_GPIO, OUTPUT);

  all_outputs_low();
  Serial.begin(2000000);

  attachInterrupt(digitalPinToInterrupt(PIN_TRIAL_START), trial_start_received, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_TRIAL_STOP), trial_stop_received, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_PCO_STATUS_EXPOS), exposure_status_changed, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_SYNC), sync_received, RISING);

  start_time_ms = millis();
  last_trial_start_state = digitalReadFast(PIN_TRIAL_START);
  last_trial_stop_state = digitalReadFast(PIN_TRIAL_STOP);
}

void report_events() {
  noInterrupts();
  int32_t frame_ms = last_frame_ms;
  uint32_t frame_number = last_frame_count;
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
    Serial.print(frame_number); Serial.print(SEP);
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
  poll_trial_inputs();
  update_acquire_enable();
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
        if (cnt >= MSGSIZE) {
          msg[MSGSIZE - 1] = '\0';
        } else {
          msg[cnt] = '\0';
        }
        cnt = 0;
        String reply = String(STX);

        switch (msg[1]) {
          case START_LEDS:
            start_time_ms = millis();
            frame_count = 0;
            last_frame_count = 0;
            sync_frame_count = 0;
            sync_count = 0;
            last_sync_ms = -1;
            last_frame_ms = -1;
            last_trial_ms = -1;
            trial_active = trial_gated ? 0 : 1;
            armed = 1;
            update_acquire_enable();
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

          case SET_TRIAL_GATE:
            token = strtok(msg, SEP);
            token = strtok(NULL, SEP);
            trial_gated = token ? (atoi(token) ? 1 : 0) : 0;
            trial_active = trial_gated ? 0 : 1;
            all_outputs_low();
            update_acquire_enable();
            reply += SET_TRIAL_GATE;
            Serial.print(reply); Serial.print(SEP);
            Serial.print((int)trial_gated); Serial.print(ETX);
            break;

          case SET_MAX_TRIAL_MS:
            token = strtok(msg, SEP);
            token = strtok(NULL, SEP);
            max_trial_ms = token ? atol(token) : 5000;
            reply += SET_MAX_TRIAL_MS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(max_trial_ms); Serial.print(ETX);
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
