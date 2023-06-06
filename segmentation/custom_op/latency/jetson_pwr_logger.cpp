#include <fstream>
#include <thread>
#include <chrono>
using namespace std;

#define to_us(t) std::chrono::duration_cast<std::chrono::microseconds>(t)

int main(int argc, char** argv) {
    char read_path[] = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input";
    std::chrono::time_point<std::chrono::system_clock> timeStamp;
    if (argc != 3) {
        printf("Usage: %s log_file sample_time[ms]\n", argv[0]);
        return -1;
    }
    ofstream log(argv[1]);
    if (!log) {
        printf("Unable to open log file %s\n", argv[1]);
        return -1;
    }
    int sampleTime = atoi(argv[2]);

    ifstream pwr_file;
    int read;
    while (1) {
        timeStamp = std::chrono::high_resolution_clock::now();
        log << to_us(timeStamp.time_since_epoch()).count() << " ";
        for (int i = 0; i < 3; i++) {
            read_path[57] = '0' + i;
            pwr_file.open(read_path, std::ios_base::in);
            if (!pwr_file) {
                printf("Unable to open the power log %s\n", read_path);
                return -1;
            }
            pwr_file >> read;
            log << read << ' ';
            pwr_file.close();
        }
        log << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(sampleTime));
    }
    return 0;
}