#define IN1 9  // Động cơ trái, tiến
#define IN2 10 // Động cơ trái, lùi
#define ENA 11 // PWM tốc độ động cơ trái
#define IN3 5  // Động cơ phải, tiến
#define IN4 6  // Động cơ phải, lùi
#define ENB 3  // PWM tốc độ động cơ phải

void setup()
{
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    pinMode(ENA, OUTPUT);
    pinMode(IN3, OUTPUT);
    pinMode(IN4, OUTPUT);
    pinMode(ENB, OUTPUT);
    Serial.begin(9600); // Khởi tạo giao tiếp Serial
}

void loop()
{
    if (Serial.available() > 0)
    {
        char command = Serial.read();
        if (command == 'F')
        { // Tốc độ bình thường
            digitalWrite(IN1, HIGH);
            digitalWrite(IN2, LOW);
            analogWrite(ENA, 255); // Tốc độ max
            digitalWrite(IN3, HIGH);
            digitalWrite(IN4, LOW);
            analogWrite(ENB, 255);
        }
        else if (command == 'S')
        { // Giảm tốc độ (1/3)
            digitalWrite(IN1, HIGH);
            digitalWrite(IN2, LOW);
            analogWrite(ENA, 85); // Tốc độ 1/3
            digitalWrite(IN3, HIGH);
            digitalWrite(IN4, LOW);
            analogWrite(ENB, 85);
        }
    }
}