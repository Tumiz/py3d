#include <fstream>
#include <cmath>
#include <map>
#include <unistd.h>

class Scope{
    public:
        Scope(const char* path):f(path),t(0){
            f<<"[\n";
        }
        static Scope& instance() {
            static Scope s("log.json");
            return s;
        }
        void tick(double dt){
            f << "{\"t\":" << t;
            for(auto e:frame){
                f<<",\""<<e.first<<"\":"<<e.second;
            }
            f<<"},"<<std::endl;
            frame.clear();
            t += dt;
        }
        template <typename T>
        Scope& w(const char* key, T value){
            frame[key]=std::to_string(value);
            return *this;
        }
        ~Scope(){
            f.seekp(-2L,std::ios::end);
            f<<"]"<<std::endl;
            f.close();
        }
    private:
    double t;
    std::ofstream f;
    std::map<const char*, std::string> frame;   
};