// Trial-gated widefield Teensy firmware (Acquire Enable variant).
//
// Camera runs in PCO "auto sequence" trigger mode (internal self-pacing).
// The Teensy holds PCO SMA input #2 (Acquire Enable) HIGH during a trial so
// the camera emits frames only while the trial is active. Between trials,
// Acquire Enable is LOW, the camera produces no exposures, and the labcams
// writer therefore writes nothing.
//
// LEDs are gated by PCO Status Expos (SMA output #4) exactly as in
// constant_camera_dual_wavelength.ino: each rising edge on Status Expos lights
// the next LED in the alternation, so LED choice is driven off real exposures
// and cannot desynchronize from the camera frame stream.
//
// Deterministic pair boundaries: each trial starts on 415 (pulse_count reset on
// trial_start) and the stop is deferred until a 415/470 pair completes, so every
// trial emits an EVEN number of frames ending on 470. This keeps the running
// frame parity aligned with the LED across trials (channel 0 = 415, matching the
// analysis/storage convention), so the live preview and the saved frame stream
// show a consistent 415/470 split trial to trial. (A DAQ-TTL relabel before
// motion/SVD remains the ground-truth safety net.)
//
// Wiring expectation:
//   pin 20: behavior trial_start TTL input  (from behavior Arduino pin 6)
//   pin 19: behavior trial_stop  TTL input  (from behavior Arduino pin 9)
//   pin 18: Acquire Enable output           -> PCO SMA input #2
//   pin  3: PCO SMA output #4 Status Expos input -> drives LED alternation
//   pin  4: optional global sync TTL input, logged as @T/#SYNC
//   pin  5: 415 nm/violet LED TTL output
//   pin  6: 470 nm/blue   LED TTL output
//   pin  7: optional Acquire Enable mirror to DAQ
//
// PCO must be configured for "auto sequence" trigger mode and "external"
// acquire mode. PCO SMA output #4 should be configured as Status Expos,
// "Show common time of All lines", high polarity.
//
// labcams CamStimInterface-compatible protocol:
//   @Q -> @Q_NCHANNELS_2_MODES_415nm:470nm:both
//   @C -> @C_1 or @C_2
//   @M_<mode>, @G_<0_or_1>, @D_<max_trial_ms>, @N, @S
// Trial events are emitted as @R_<code>_<frame>_<ms> (code: 1=start, 2=stop,
// 3=safety_timeout).

const byte PIN_TRIAL_START       = 20;
const byte PIN_TRIAL_STOP        = 19;
const byte PIN_PCO_STATUS_EXPOS  = 3;
const byte PIN_GLOBAL_SYNC       = 4;

const byte PIN_ACQUIRE_ENABLE    = 18;  // -> PCO SMA input #2 (level signal)
const byte PIN_LED0_TRIGGER      = 5;   // 415 nm/violet
const byte PIN_LED1_TRIGGER      = 6;   // 470 nm/blue
const byte PIN_GPIO              = 7;   // optional Acquire Enable mirror

#define GPIO_MIMIC_ACQUIRE_ENABLE

#define STX '@'
#define ETX '\n'
#define SEP "_"
#define CAP "NCHANNELS_2_MODES_415nm:470nm:both"

#define QUERY_NCHANNELS   'C'
#define QUERY_CAP         'Q'
#define START_LEDS        'N'
#define STOP_LEDS         'S'
#define FRAME             'F'
#define SYNC              'T'
#define SET_MODE          'M'
#define SET_TRIAL_GATE    'G'
#define SET_MAX_TRIAL_MS  'D'
#define TRIAL_EVENT       'R'

volatile uint8_t  mode = 3;          // 1: 415, 2: 470, 3: alternate
volatile uint8_t  armed = 0;         // labcams armed (camera+LED control allowed)
volatile uint8_t  trial_active = 0;
volatile uint8_t  trial_gated = 0;   // 0: preview (acquire on while armed), 1: gate on trial state
// Deferred stop: when a trial_stop/timeout arrives mid-pair we keep acquiring
// until the current 470/415 pair finishes, so every trial ends on 415 and emits
// an EVEN number of frames. This keeps the running frame parity aligned with the
// LED across trials (consistent 415/470 in live preview and in the saved stream).
// 0 = none, 2 = trial_stop pending, 3 = safety-timeout pending.
volatile uint8_t  stop_pending = 0;
volatile uint8_t  last_trial_start_state = LOW;
volatile uint8_t  last_trial_stop_state  = LOW;
volatile uint32_t trial_start_ms = 0;
// safety stop if trial_stop TTL is missed. 0 disables.
volatile uint32_t max_trial_ms = 5000;

volatile uint32_t pulse_count = 0;
volatile uint32_t last_pulse_count = 0;
volatile int32_t  last_frame_ms = -1;
volatile byte     last_led_pin = 0;

volatile uint32_t sync_count = 0;
volatile uint32_t sync_frame_count = 0;
volatile int32_t  last_sync_ms = -1;

volatile uint8_t  last_trial_code = 0;  // 1=start, 2=stop, 3=safety timeout
volatile uint32_t last_trial_frame = 0;
volatile int32_t  last_trial_ms = -1;

uint32_t start_time_ms = 0;

#define MSGSIZE 64
char msg[MSGSIZE];
int  cnt = 0;

int32_t elapsed_ms() { return (int32_t)(millis() - start_time_ms); }

// Acquire Enable goes HIGH when the camera is allowed to acquire:
// armed AND (preview mode OR trial currently active).
inline void apply_acquire_enable() {
  uint8_t allow = armed && (!trial_gated || trial_active);
  digitalWriteFast(PIN_ACQUIRE_ENABLE, allow ? HIGH : LOW);
#ifdef GPIO_MIMIC_ACQUIRE_ENABLE
  digitalWriteFast(PIN_GPIO, allow ? HIGH : LOW);
#endif
}

inline void all_leds_low() {
  digitalWriteFast(PIN_LED0_TRIGGER, LOW);
  digitalWriteFast(PIN_LED1_TRIGGER, LOW);
}

// End the trial: drop Acquire Enable (camera stops after the current frame),
// LEDs off, and stamp the trial event for report_events(). Called only at a
// complete-pair boundary (pulse_count even => last captured frame was 470).
inline void finalize_trial_stop(uint8_t code) {
  trial_active = 0;
  stop_pending = 0;
  apply_acquire_enable();
  all_leds_low();
  last_trial_code = code;
  last_trial_frame = pulse_count;
  last_trial_ms = elapsed_ms();
}

void trial_start_received() {
  if (digitalReadFast(PIN_TRIAL_START) == HIGH) {
    last_trial_start_state = HIGH;
    trial_active = 1;
    stop_pending = 0;
    trial_start_ms = millis();
    // Reset alternation phase so each trial starts deterministically on LED0
    // (violet/415, isosbestic): the first Status Expos rise increments
    // pulse_count to 1 (odd) and exposure_status_changed() maps odd -> LED0
    // (415). Combined with the deferred stop (ends on 470), every trial is
    // 415,470,...,415,470 (even frame count, channel 0 = 415).
    pulse_count = 0;
    last_trial_code = 1;
    last_trial_frame = pulse_count;
    last_trial_ms = elapsed_ms();
    apply_acquire_enable();
  }
}

void trial_stop_received() {
  if (digitalReadFast(PIN_TRIAL_STOP) == HIGH) {
    last_trial_stop_state = HIGH;
    // Defer the stop until the current 415/470 pair completes so the trial ends
    // on 470 with an even frame count. If we are already at a pair boundary with
    // no exposure in progress, stop immediately (no extra frames).
    if (digitalReadFast(PIN_PCO_STATUS_EXPOS) == LOW && (pulse_count % 2 == 0)) {
      finalize_trial_stop(2);
    } else {
      stop_pending = 2;
    }
  }
}

void sync_received() {
  if (digitalReadFast(PIN_GLOBAL_SYNC) == HIGH) {
    sync_count++;
    sync_frame_count = pulse_count;
    last_sync_ms = elapsed_ms();
  }
}

// Closed-loop LED gating: every real PCO exposure (Status Expos HIGH)
// increments pulse_count and lights the next LED in the alternation. This
// cannot desynchronize from the camera because the camera is the master.
void exposure_status_changed() {
  if (digitalReadFast(PIN_PCO_STATUS_EXPOS) == HIGH) {
    if (armed && (!trial_gated || trial_active)) {
      pulse_count++;
      byte pin_out = PIN_LED0_TRIGGER;
      switch (mode) {
        case 1:
          pin_out = PIN_LED0_TRIGGER;
          break;
        case 2:
          pin_out = PIN_LED1_TRIGGER;
          break;
        case 3:
          // odd pulse_count -> LED0 (415), even -> LED1 (470). With pulse_count
          // reset to 0 on trial_start, the first exposure (pulse_count==1) is 415
          // (isosbestic). Channel 0 = 415 then matches the analysis/storage
          // convention; deferred stop ends each trial on 470 (complete pair).
          pin_out = (pulse_count % 2 == 0) ? PIN_LED1_TRIGGER : PIN_LED0_TRIGGER;
          break;
        default:
          pin_out = PIN_LED0_TRIGGER;
          break;
      }
      digitalWriteFast(PIN_LED0_TRIGGER, LOW);
      digitalWriteFast(PIN_LED1_TRIGGER, LOW);
      digitalWriteFast(pin_out, HIGH);
      last_led_pin = pin_out;
      last_pulse_count = pulse_count;
      last_frame_ms = elapsed_ms();
    }
  } else {
    // Exposure ended (Status Expos falling). LEDs off, and if a stop is pending
    // finalize once we have just completed a 470 frame (pulse_count even) so the
    // trial ends on a complete pair.
    all_leds_low();
    if (stop_pending && (pulse_count % 2 == 0)) {
      finalize_trial_stop(stop_pending);
    }
  }
}

void poll_trial_inputs() {
  uint8_t start_state = digitalReadFast(PIN_TRIAL_START);
  uint8_t stop_state  = digitalReadFast(PIN_TRIAL_STOP);

  if (start_state == HIGH && last_trial_start_state == LOW) {
    trial_start_received();
  }
  if (stop_state == HIGH && last_trial_stop_state == LOW) {
    trial_stop_received();
  }
  last_trial_start_state = start_state;
  last_trial_stop_state  = stop_state;

  if (trial_gated && trial_active && max_trial_ms > 0 && !stop_pending) {
    uint32_t elapsed = millis() - trial_start_ms;
    if (elapsed >= max_trial_ms) {
      // Safety timeout: end on a complete pair like a normal stop.
      if (digitalReadFast(PIN_PCO_STATUS_EXPOS) == LOW && (pulse_count % 2 == 0)) {
        finalize_trial_stop(3);
      } else {
        stop_pending = 3;
      }
    }
  }
}

void setup() {
  pinMode(PIN_TRIAL_START,      INPUT);
  pinMode(PIN_TRIAL_STOP,       INPUT);
  pinMode(PIN_PCO_STATUS_EXPOS, INPUT);
  pinMode(PIN_GLOBAL_SYNC,      INPUT);
  pinMode(PIN_ACQUIRE_ENABLE,   OUTPUT);
  pinMode(PIN_LED0_TRIGGER,     OUTPUT);
  pinMode(PIN_LED1_TRIGGER,     OUTPUT);
  pinMode(PIN_GPIO,             OUTPUT);

  digitalWriteFast(PIN_ACQUIRE_ENABLE, LOW);
  digitalWriteFast(PIN_LED0_TRIGGER,   LOW);
  digitalWriteFast(PIN_LED1_TRIGGER,   LOW);
  digitalWriteFast(PIN_GPIO,           LOW);

  Serial.begin(2000000);

  attachInterrupt(digitalPinToInterrupt(PIN_TRIAL_START),      trial_start_received,    RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_TRIAL_STOP),       trial_stop_received,     RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_PCO_STATUS_EXPOS), exposure_status_changed, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_GLOBAL_SYNC),      sync_received,           RISING);

  start_time_ms = millis();
  last_trial_start_state = digitalReadFast(PIN_TRIAL_START);
  last_trial_stop_state  = digitalReadFast(PIN_TRIAL_STOP);
}

void report_events() {
  noInterrupts();
  int32_t  frame_ms    = last_frame_ms;
  uint32_t frame_count = last_pulse_count;
  byte     led         = last_led_pin;
  int32_t  sync_ms     = last_sync_ms;
  uint32_t sc          = sync_count;
  uint32_t sfc         = sync_frame_count;
  int32_t  trial_ms    = last_trial_ms;
  uint8_t  trial_code  = last_trial_code;
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
  poll_trial_inputs();
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
            stop_pending = 0;
            trial_active = trial_gated ? 0 : 1;
            armed = 1;
            apply_acquire_enable();
            reply += START_LEDS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(elapsed_ms()); Serial.print(ETX);
            break;

          case STOP_LEDS:
            armed = 0;
            trial_active = 0;
            stop_pending = 0;
            apply_acquire_enable();
            all_leds_low();
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
            apply_acquire_enable();
            all_leds_low();
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
