// Imaging Teensy: gate dual-wavelength LEDs from PCO exposure pulses.
//
// This is a labcams-compatible variant of the upstream
// stim_camera_trigger_dual_wavelength sketch.  It keeps the serial protocol
// and capability string shape expected by labcams CamStimInterface:
//   @Q -> @Q_NCHANNELS_2_MODES_415nm:470nm:both
//   @C -> @C_1 or @C_2
//   @M_<mode>, @N, @S

const byte PIN_CAM_EXPOSURE = 3;   // PCO exposure TTL into Teensy interrupt pin
const byte PIN_SYNC0        = 4;   // optional behavior/global sync into Teensy
const byte PIN_SYNC1        = 2;   // unused by labcams, kept available for spare sync
const byte PIN_LED0_TRIGGER = 5;   // LED0 TTL out, advertised as 415nm/violet by labcams
const byte PIN_LED1_TRIGGER = 6;   // LED1 TTL out, advertised as 470nm/blue by labcams
const byte PIN_GPIO         = 7;   // optional exposure mirror for DAQ/camera timing

#define GPIO_MIMIC_EXPOSURE

// ---------- Protocol ----------
#define STX '@'
#define ETX '\n'
#define SEP "_"
#define CAP "NCHANNELS_2_MODES_415nm:470nm:both"

#define QUERY_NCHANNELS 'C'
#define QUERY_CAP       'Q'
#define START_LEDS      'N'   // ARM
#define STOP_LEDS       'S'   // DISARM
#define FRAME           'F'   // Teensy -> PC (frame event)
#define SYNC            'T'   // Teensy -> PC (sync event)
#define SET_MODE        'M'   // PC -> Teensy (1=415/violet, 2=470/blue, 3=alternate/both)

// ---------- State ----------
volatile uint32_t current_time = 0;
volatile uint32_t start_time   = 0;

volatile uint32_t pulse_count      = 0;
volatile uint32_t last_pulse_count = 0;
volatile int32_t  last_rise_ms     = -1;
volatile byte     last_led_pin     = 0;

volatile int32_t  last_sync_ms     = -1;
volatile uint32_t sync_count       = 0;
volatile uint32_t sync_frame_count = 0;

volatile uint8_t mode  = 3;     // 1: 415/violet, 2: 470/blue, 3: alternate
volatile uint8_t armed = 0;

// ---------- Serial RX buffer ----------
#define MSGSIZE 64
char msg[MSGSIZE];
int  cnt = 0;

void camera_triggered() {
  if (digitalReadFast(PIN_CAM_EXPOSURE) == LOW) {
    digitalWriteFast(PIN_LED0_TRIGGER, LOW);
    digitalWriteFast(PIN_LED1_TRIGGER, LOW);
#ifdef GPIO_MIMIC_EXPOSURE
    digitalWriteFast(PIN_GPIO, LOW);
#endif
  } else if (armed) {
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
        pin_out = (pulse_count % 2 == 0) ? PIN_LED1_TRIGGER : PIN_LED0_TRIGGER;
        break;
      default:
        pin_out = PIN_LED0_TRIGGER;
        break;
    }

    digitalWriteFast(pin_out, HIGH);
#ifdef GPIO_MIMIC_EXPOSURE
    digitalWriteFast(PIN_GPIO, HIGH);
#endif
    last_rise_ms = millis() - start_time;
    last_led_pin = pin_out;
    last_pulse_count = pulse_count;
  }
}

void sync_received() {
  if (digitalReadFast(PIN_SYNC0) == HIGH) {
    sync_count++;
  }
  sync_frame_count = pulse_count;
  last_sync_ms = millis() - start_time;
}

void setup() {
  pinMode(PIN_LED0_TRIGGER, OUTPUT);
  pinMode(PIN_LED1_TRIGGER, OUTPUT);
  pinMode(PIN_GPIO, OUTPUT);
  pinMode(PIN_SYNC0, INPUT);
  pinMode(PIN_SYNC1, INPUT);
  pinMode(PIN_CAM_EXPOSURE, INPUT);

  digitalWriteFast(PIN_LED0_TRIGGER, LOW);
  digitalWriteFast(PIN_LED1_TRIGGER, LOW);
  digitalWriteFast(PIN_GPIO, LOW);

  Serial.begin(2000000);

  attachInterrupt(digitalPinToInterrupt(PIN_CAM_EXPOSURE), camera_triggered, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_SYNC0),        sync_received,   CHANGE);

  start_time = millis();
}

void loop() {
  current_time = millis() - start_time;

  noInterrupts();
  int32_t lr = last_rise_ms;
  byte led = last_led_pin;
  uint32_t pc = last_pulse_count;
  interrupts();

  if (lr > 0 && abs((int32_t)current_time - lr) > 10) {
    Serial.print(STX); Serial.print(FRAME); Serial.print(SEP);
    Serial.print((int)led); Serial.print(SEP);
    Serial.print(pc); Serial.print(SEP);
    Serial.print(lr); Serial.print(ETX);
    noInterrupts(); last_rise_ms = -1; interrupts();
  }

  noInterrupts();
  int32_t lsync = last_sync_ms;
  uint32_t sc = sync_count;
  uint32_t sfc = sync_frame_count;
  interrupts();

  if (lsync > 0) {
    Serial.print(STX); Serial.print(SYNC); Serial.print(SEP);
    Serial.print(sfc); Serial.print(SEP);
    Serial.print(sc); Serial.print(SEP);
    Serial.print(lsync); Serial.print(ETX);
    noInterrupts(); last_sync_ms = -1; interrupts();
  }

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
            last_sync_ms = -1;
            last_rise_ms = -1;
            start_time = millis();
            pulse_count = 0;
            last_pulse_count = 0;
            sync_frame_count = 0;
            sync_count = 0;
            armed = 1;
            reply += START_LEDS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(millis() - start_time); Serial.print(ETX);
            break;

          case STOP_LEDS:
            armed = 0;
            digitalWriteFast(PIN_LED0_TRIGGER, LOW);
            digitalWriteFast(PIN_LED1_TRIGGER, LOW);
            digitalWriteFast(PIN_GPIO, LOW);
            reply += STOP_LEDS;
            Serial.print(reply); Serial.print(SEP);
            Serial.print(millis() - start_time); Serial.print(ETX);
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
